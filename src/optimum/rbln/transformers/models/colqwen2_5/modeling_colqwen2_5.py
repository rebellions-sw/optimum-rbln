# Copyright 2025 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING, Any, List, Optional, Union

import rebel
import torch
from rebel.compile_context import CompileContext
from transformers import (
    AutoModelForVision2Seq,
    PretrainedConfig,
    PreTrainedModel,
    Qwen2_5_VLForConditionalGeneration,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLRotaryEmbedding,
)

from ....configuration_utils import RBLNCompileConfig
from ....modeling import RBLNModel
from ....utils.logging import get_logger
from ..decoderonly.modeling_decoderonly import (
    RBLNRuntimeModel,
    set_default_values,
    validate_attention_method,
)
from ..qwen2_5_vl.modeling_qwen2_5_vl import RBLNQwen2_5_VLForConditionalGeneration
from .colqwen2_5_architecture import ColQwen2_5_LanguageModelWrapper
from .configuration_colqwen2_5 import RBLNColQwen2_5ForConditionalGenerationConfig


logger = get_logger(__name__)

if TYPE_CHECKING:
    from transformers import (
        AutoFeatureExtractor,
        AutoProcessor,
        AutoTokenizer,
        PretrainedConfig,
    )


class RBLNRuntimeModelForColQwen2_5(RBLNRuntimeModel):
    def __init__(
        self,
        runtime: rebel.Runtime,
        batch_size: int,
        rbln_config: RBLNColQwen2_5ForConditionalGenerationConfig,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            runtime,
            "prefill",
            batch_size,
            None,
            None,
            None,
            rbln_config,
            **kwargs,
        )

    def prefill_forward(
        self,
        inputs: torch.Tensor,
        cache_position: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        batch_idx: int = None,
        block_tables: torch.Tensor = None,
        is_external_block_tables: bool = None,
        position_embed: Optional[torch.Tensor] = None,
        local_block_tables: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        """
        Performs chunked prefill for efficient KV-cache updates and memory optimization.
        Instead of processing the entire sequence at once, the input is divided into chunks of size `prefill_chunk_size`,
        and each chunk is processed sequentially. This allows for better memory utilization and compatibility with continuous batching.
        """
        (
            inputs,
            cache_position,
            chunked_attention_mask,
            _,
            _,
            position_embed,
            _,
            query_length,
        ) = self._prepare_prefill_inputs(
            inputs, cache_position, attention_mask, position_embed, token_type_ids=token_type_ids
        )

        projs = []
        # Process input in chunks of size `prefill_chunk_size`
        for step in range(0, query_length, self.rbln_config.prefill_chunk_size):
            # Extract the current chunk of inputs and cache positions
            input_chunk = inputs[:, step : step + self.rbln_config.prefill_chunk_size]
            cache_pos_chunk = cache_position[:, step : step + self.rbln_config.prefill_chunk_size]
            if position_embed is not None:
                position_embed_chunk = position_embed[:, :, :, step : step + self.rbln_config.prefill_chunk_size, :]

            if self.rbln_config.use_attention_mask and not self.rbln_config.use_position_ids:
                # Update attention mask to ensure proper causal behavior
                if step >= self.rbln_config.prefill_chunk_size:
                    chunked_attention_mask[:, :, :, step - self.rbln_config.prefill_chunk_size : step] = 1
                chunked_attention_mask[:, :, :, step : step + self.rbln_config.prefill_chunk_size] = self.causal_mask

            # Forward pass for the current chunk
            proj = self.runtime(
                inputs_embeds=input_chunk,
                cache_position=cache_pos_chunk,
                block_tables=block_tables,
                position_emb=position_embed_chunk if position_embed is not None else None,
            )
            projs.append(proj)

        return torch.concat(projs, dim=-2)


class RBLNColQwen2_5ForConditionalGeneration(RBLNQwen2_5_VLForConditionalGeneration):
    auto_model_class = AutoModelForVision2Seq
    _rbln_submodules = [
        {"name": "visual"},
    ]
    _decoder_wrapper_cls = ColQwen2_5_LanguageModelWrapper
    _use_rotary_emb = False

    def __post_init__(self, **kwargs):
        main_input_name = self.main_input_name

        main_input_name = "inputs_embeds"
        artifacts = torch.load(self.model_save_dir / self.subfolder / "torch_artifacts.pth", weights_only=False)
        self.embed_tokens = self._create_embedding_layer()
        self.embed_tokens.load_state_dict(artifacts["embed_tokens"])

        self.prefill = RBLNRuntimeModelForColQwen2_5(
            runtime=self.model[0],
            main_input_name=main_input_name,
            embed_tokens=self.embed_tokens,
            batch_size=self.rbln_config.batch_size,
            rbln_config=self.rbln_config,
            vocab_size=self.config.vocab_size,
        )

        self.visual = self.rbln_submodules[0]
        self.mrope_section = self.config.rope_scaling["mrope_section"]
        self.rotary_emb = Qwen2_5_VLRotaryEmbedding(self.config)
        self.rope_deltas = torch.zeros(self.rbln_config.batch_size)

    def can_generate(self):
        return False

    @classmethod
    @torch.inference_mode()
    def get_compiled_model(cls, model: "PreTrainedModel", rbln_config: RBLNColQwen2_5ForConditionalGenerationConfig):
        wrapped_model = cls.wrap_model_if_needed(model, rbln_config)

        rbln_compile_configs = rbln_config.compile_cfgs
        prefill_compile_config = rbln_compile_configs[0]

        context = CompileContext(use_weight_sharing=False)

        # Here we use meta tensor, for the memory efficiency.
        meta_tensor_names = [name for name, _, _ in prefill_compile_config.input_info if "past_key_values" in name]
        prefill_example_inputs = prefill_compile_config.get_dummy_inputs(fill=0, meta_tensor_names=meta_tensor_names)

        # Mark static tensors (self kv states)
        static_tensors = {}
        for (name, _, _), tensor in zip(prefill_compile_config.input_info, prefill_example_inputs):
            if "past_key_values" in name:
                static_tensors[name] = tensor
                context.mark_static_address(tensor)

        def compile_model(wrapped_model, compile_config, example_inputs, compile_context, quantization):
            try:
                if quantization:
                    quantization.maybe_set_quantization_env()
                original_linear = torch.nn.functional.linear
                torch.nn.functional.linear = torch.ops.rbln_custom_ops.linear
                compiled_model = RBLNModel.compile(
                    wrapped_model,
                    compile_config,
                    example_inputs=example_inputs,
                    compile_context=compile_context,
                )
                return compiled_model
            finally:
                torch.nn.functional.linear = original_linear
                if quantization:
                    quantization.maybe_reset_quantization_env()

        wrapped_model.phase = "prefill"
        compiled_prefill = compile_model(
            wrapped_model, prefill_compile_config, prefill_example_inputs, context, rbln_config.quantization
        )

        compiled_models = {"prefill": compiled_prefill}
        return compiled_models

    @classmethod
    def get_input_info(
        cls,
        batch_size: int,
        query_length: int,
        rbln_config: RBLNColQwen2_5ForConditionalGenerationConfig,
        model_config: PretrainedConfig,
    ):
        input_info = super().get_input_info(
            batch_size,
            query_length,
            rbln_config=rbln_config,
            model_config=model_config,
        )
        query_position = input_info.pop(4)  # remove query postion
        assert query_position[0] == "query_position", print(query_position[0], "is deleted.")
        return input_info

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Optional[Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"]] = None,
        model: Optional["PreTrainedModel"] = None,
        model_config: Optional["PretrainedConfig"] = None,
        rbln_config: Optional[RBLNColQwen2_5ForConditionalGenerationConfig] = None,
    ) -> RBLNColQwen2_5ForConditionalGenerationConfig:
        if rbln_config.max_seq_len is None:
            rbln_config.max_seq_len = getattr(model_config, "max_position_embeddings", None) or getattr(
                model_config, "n_positions", None
            )
        if rbln_config.max_seq_len is None:
            raise ValueError("`max_seq_len` should be specified.")

        rbln_config.attn_impl, rbln_config.kvcache_partition_len, rbln_config.kvcache_block_size = set_default_values(
            attn_impl=rbln_config.attn_impl,
            kvcache_partition_len=rbln_config.kvcache_partition_len,
            kvcache_block_size=rbln_config.kvcache_block_size,
            max_seq_len=rbln_config.max_seq_len,
        )

        validate_attention_method(
            attn_impl=rbln_config.attn_impl,
            kvcache_partition_len=rbln_config.kvcache_partition_len,
            kvcache_block_size=rbln_config.kvcache_block_size,
            max_seq_len=rbln_config.max_seq_len,
        )

        required_num_blocks = (rbln_config.max_seq_len // rbln_config.kvcache_block_size) * rbln_config.batch_size
        max_num_blocks = required_num_blocks

        if rbln_config.kvcache_num_blocks is None:
            rbln_config.kvcache_num_blocks = max_num_blocks

        prefill_input_info = cls.get_input_info(
            batch_size=1,
            query_length=rbln_config.prefill_chunk_size,
            rbln_config=rbln_config,
            model_config=model_config,
        )

        prefill_compile_config = RBLNCompileConfig(compiled_model_name="prefill", input_info=prefill_input_info)
        rbln_config.set_compile_cfgs([prefill_compile_config])

        return rbln_config

    @classmethod
    def _create_runtimes(
        cls,
        compiled_models: List[rebel.RBLNCompiledModel],
        rbln_config: RBLNColQwen2_5ForConditionalGenerationConfig,
    ) -> List[rebel.Runtime]:
        expected_model_names = ["prefill"]
        if any(model_name not in rbln_config.device_map for model_name in expected_model_names):
            cls._raise_missing_compiled_file_error(expected_model_names)

        return [
            rebel.Runtime(
                compiled_models[0],
                tensor_type="pt",
                device=rbln_config.device_map["prefill"],
                activate_profiler=rbln_config.activate_profiler,
            ),
        ]

    def get_rope_index(self, *args, **kwargs):
        return Qwen2_5_VLForConditionalGeneration.get_rope_index(self, *args, **kwargs)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        generate_idx: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> torch.Tensor:
        
        # Handle the custom "pixel_values" input obtained with `ColQwen2Processor` through unpadding
        if pixel_values is not None and image_grid_thw is not None:
            offsets = image_grid_thw[:, 1] * image_grid_thw[:, 2]  # (batch_size,)
            pixel_values = torch.cat(
                [pixel_sequence[:offset] for pixel_sequence, offset in zip(pixel_values, offsets)],
                dim=0,
            )

        inputs_embeds, position_embed, rope_deltas = self._preprocess_prefill(
            input_ids,
            attention_mask,
            pixel_values,
            pixel_values_videos,
            image_grid_thw,
            video_grid_thw,
            second_per_grid_ts,
        )

        self.rope_deltas = rope_deltas
        batch_size = inputs_embeds.shape[0]

        projs = []
        max_size = self.rbln_config.prefill_chunk_size * (
            inputs_embeds.shape[1] // self.rbln_config.prefill_chunk_size + 1
        )

        block_tables = torch.arange(
            0,
            self.rbln_config.max_seq_len // self.rbln_config.kvcache_block_size,
            dtype=torch.int16,
        )

        for b_idx in range(batch_size):
            cache_position = torch.arange(0, inputs_embeds.shape[1], dtype=torch.int32).unsqueeze(0)

            proj = self.prefill(
                inputs_embeds=inputs_embeds[b_idx : b_idx + 1],
                attention_mask=attention_mask[b_idx] if attention_mask is not None else None,
                cache_position=cache_position,
                block_tables=block_tables,
                position_embed=position_embed[:, b_idx : b_idx + 1],
            )
            pad_size = (0, 0, 0, max_size - proj.shape[1], 0, 0)
            padded_proj = torch.nn.functional.pad(
                proj, pad_size, "constant", 1e-8
            )  # For normaliztion, fill non-zero value
            projs.append(padded_proj)

        # post process
        projs = torch.cat(projs, dim=0)
        projs = projs[:, : inputs_embeds.shape[1]]
        projs = projs / projs.norm(dim=-1, keepdim=True)  # (batch_size, sequence_length, dim)

        fliped_attention_mask_batch = torch.flip(attention_mask, dims=[-1])
        projs = projs * fliped_attention_mask_batch.unsqueeze(-1)  # (batch_size, sequence_length, dim)
        return projs
