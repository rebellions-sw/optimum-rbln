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

from typing import TYPE_CHECKING, List, Optional, Union

import rebel
import torch
from rebel.compile_context import CompileContext
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    Qwen2_5_VLForConditionalGeneration,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLRotaryEmbedding,
)

from ....configuration_utils import RBLNCompileConfig
from ....modeling import RBLNModel
from ..decoderonly.modeling_decoderonly import (
    set_default_values,
    validate_attention_method,
)
from ..qwen2_5_vl.modeling_qwen2_5_vl import RBLNQwen2_5_VLForConditionalGeneration
from .colqwen2_5_architecture import ColQwen2_5_LanguageModelWrapper
from .configuration_colqwen2_5 import RBLNColQwen2_5ForRetrievalConfig


if TYPE_CHECKING:
    from transformers import (
        AutoFeatureExtractor,
        AutoProcessor,
        AutoTokenizer,
        PretrainedConfig,
    )


class RBLNColQwen2_5ForRetrieval(RBLNQwen2_5_VLForConditionalGeneration):
    main_input_name = "inputs_embeds"
    auto_model_class = None
    _rbln_submodules = [
        {"name": "visual"},
    ]
    _decoder_wrapper_cls = ColQwen2_5_LanguageModelWrapper
    _use_rotary_emb = False

    def __post_init__(self, **kwargs):
        artifacts = torch.load(self.model_save_dir / self.subfolder / "torch_artifacts.pth", weights_only=False)
        self.embed_tokens = self._create_embedding_layer()
        self.embed_tokens.load_state_dict(artifacts["embed_tokens"])
        self.visual = self.rbln_submodules[0]
        self.prefill_runtime = self.model[0]
        self.mrope_section = self.config.rope_scaling["mrope_section"]
        self.rotary_emb = Qwen2_5_VLRotaryEmbedding(self.config)
        self.block_tables = torch.arange(self.rbln_config.kvcache_num_blocks, dtype=torch.int16)

    @classmethod
    @torch.inference_mode()
    def get_compiled_model(cls, model: "PreTrainedModel", rbln_config: RBLNColQwen2_5ForRetrievalConfig):
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

        def compile_model(wrapped_model, compile_config, example_inputs, compile_context):
            try:
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

        wrapped_model.phase = "prefill"
        compiled_prefill = compile_model(wrapped_model, prefill_compile_config, prefill_example_inputs, context)

        compiled_models = {"prefill": compiled_prefill}
        return compiled_models

    @classmethod
    def get_input_info(
        cls,
        batch_size: int,
        query_length: int,
        rbln_config: RBLNColQwen2_5ForRetrievalConfig,
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
        rbln_config: Optional[RBLNColQwen2_5ForRetrievalConfig] = None,
    ) -> RBLNColQwen2_5ForRetrievalConfig:
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

        required_num_blocks = rbln_config.max_seq_len // rbln_config.kvcache_block_size
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
        rbln_config: RBLNColQwen2_5ForRetrievalConfig,
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

    def _preprocess_chunked_prefill(self, inputs_embeds, attention_mask, position_embed):
        query_length = inputs_embeds.shape[1] if attention_mask is None else torch.sum(attention_mask.view(-1)).item()
        # extract valid inputs
        inputs_embeds = inputs_embeds[:, attention_mask.bool()] if attention_mask is not None else inputs_embeds
        position_embed = (
            position_embed[:, :, :, attention_mask.bool(), :] if attention_mask is not None else position_embed
        )
        # adding padding for chunked prefill
        padding_size = (
            self.rbln_config.prefill_chunk_size - (query_length % self.rbln_config.prefill_chunk_size)
        ) % self.rbln_config.prefill_chunk_size
        padded_len = query_length + padding_size

        inputs_embeds = torch.nn.functional.pad(inputs_embeds, (0, 0, 0, padding_size))
        position_embed = torch.nn.functional.pad(position_embed, (0, 0, 0, padding_size))
        cache_position = torch.arange(padded_len, dtype=torch.int32).unsqueeze(0)

        return inputs_embeds, position_embed, cache_position, query_length

    def _chunked_prefill_forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embed: Optional[torch.Tensor] = None,
    ):
        padded_inputs_embeds, padded_position_embed, cache_position, query_length = self._preprocess_chunked_prefill(
            inputs_embeds, attention_mask, position_embed
        )
        
        # chunked prefill
        projs = []
        for step in range(0, query_length, self.rbln_config.prefill_chunk_size):
            # Extract the current chunk of inputs and cache positions
            input_chunk = padded_inputs_embeds[:, step : step + self.rbln_config.prefill_chunk_size]
            cache_pos_chunk = cache_position[:, step : step + self.rbln_config.prefill_chunk_size]
            position_embed_chunk = padded_position_embed[:, :, :, step : step + self.rbln_config.prefill_chunk_size, :]

            # Forward pass for the current chunk
            proj = self.prefill_runtime(
                inputs_embeds=input_chunk,
                cache_position=cache_pos_chunk,
                block_tables=self.block_tables,
                position_emb=position_embed_chunk,
            )
            projs.append(proj)
        projs = torch.concat(projs, dim=-2)[:, :query_length]

        return self._postprocess_chunked_prefill(projs, attention_mask)

    def _postprocess_chunked_prefill(self, projs, attention_mask):
        if attention_mask is not None:
            embedding = torch.full(
                (1, attention_mask.shape[-1], projs.shape[-1]),
                fill_value=1e-10,
                dtype=projs.dtype,
            )
            mask_indices = torch.nonzero(attention_mask, as_tuple=True)[0]
            embedding.index_copy_(dim=-2, index=mask_indices, source=projs)
        else:
            embedding = projs
        return embedding

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        # Handle the custom "pixel_values" input obtained with `ColQwen2Processor` through unpadding
        if pixel_values is not None and image_grid_thw is not None:
            offsets = image_grid_thw[:, 1] * image_grid_thw[:, 2]  # (batch_size,)
            pixel_values = torch.cat(
                [pixel_sequence[:offset] for pixel_sequence, offset in zip(pixel_values, offsets)],
                dim=0,
            )

        # Preprocess of Qwen2_5_VLForConditionalGeneration
        inputs_embeds, position_embed, _ = self._preprocess_prefill(
            input_ids,
            attention_mask,
            pixel_values,
            pixel_values_videos,
            image_grid_thw,
            video_grid_thw,
            second_per_grid_ts,
        )
        batch_size = inputs_embeds.shape[0]

        projs = []
        for b_idx in range(batch_size):
            proj = self._chunked_prefill_forward(
                inputs_embeds[b_idx : b_idx + 1],
                attention_mask[b_idx] if attention_mask is not None else None,
                position_embed[:, b_idx : b_idx + 1],
            )
            projs.append(proj)

        # post process
        projs = torch.cat(projs, dim=0)
        projs = projs / projs.norm(dim=-1, keepdim=True)  # (batch_size, sequence_length, dim)
        projs = projs * attention_mask.unsqueeze(-1)  # (batch_size, sequence_length, dim)

        return projs
