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

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union

import rebel
import torch
from rebel.compile_context import CompileContext
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import no_init_weights

from ....configuration_utils import RBLNCompileConfig
from ....modeling import RBLNModel
from ....utils import logging
from ...models.decoderonly import RBLNDecoderOnlyModelForCausalLM, RBLNDecoderOnlyModelForCausalLMConfig
from ..decoderonly.modeling_decoderonly import set_default_values, validate_attention_method
from .configuration_qwen3 import RBLNQwen3ModelConfig
from .qwen3_architecture import Qwen3ModelWrapper, Qwen3Wrapper


logger = logging.get_logger(__name__)

if TYPE_CHECKING:
    from transformers import (
        AutoFeatureExtractor,
        AutoProcessor,
        AutoTokenizer,
        PretrainedConfig,
    )


class RBLNQwen3ForCausalLM(RBLNDecoderOnlyModelForCausalLM):
    _decoder_wrapper_cls = Qwen3Wrapper

    @classmethod
    def _update_sliding_window_config(
        cls, model_config: PretrainedConfig, rbln_config: RBLNDecoderOnlyModelForCausalLMConfig
    ):
        # https://github.com/huggingface/transformers/issues/35896
        # There seems to be a bug in transformers(v4.52.4). Therefore, similar to when attn_implementation is eager,
        # we set all layers to use sliding window in this version. This should be updated once the bug is fixed.

        rbln_config.cache_impl = "sliding_window"
        rbln_config.sliding_window = model_config.sliding_window
        rbln_config.sliding_window_layers = list(range(model_config.num_hidden_layers))
        return rbln_config

    def forward(self, *args, **kwargs):
        kwargs["return_dict"] = True
        return super().forward(*args, **kwargs)


class RBLNQwen3Model(RBLNModel):
    _decoder_wrapper_cls = Qwen3ModelWrapper
    _use_rotary_emb = True

    def __post_init__(self, **kwargs):
        artifacts = torch.load(self.model_save_dir / self.subfolder / "torch_artifacts.pth", weights_only=False)
        self.embed_tokens = self._create_embedding_layer()
        self.embed_tokens.load_state_dict(artifacts["embed_tokens"])
        self.block_tables = torch.arange(
            self.rbln_config.max_seq_len / self.rbln_config.kvcache_block_size, dtype=torch.int16
        )
        self.causal_mask = 1 - torch.triu(
            torch.ones(1, 1, self.rbln_config.prefill_chunk_size, self.rbln_config.prefill_chunk_size), diagonal=1
        )

    @classmethod
    def save_torch_artifacts(
        cls,
        model: PreTrainedModel,
        save_dir_path: Path,
        subfolder: str,
        rbln_config: RBLNQwen3ModelConfig,
    ):
        save_dict = {}
        save_dict["embed_tokens"] = model.get_input_embeddings().state_dict()
        torch.save(save_dict, save_dir_path / subfolder / "torch_artifacts.pth")

    def _create_embedding_layer(self):
        with no_init_weights():
            embed_tokens = torch.nn.Embedding(
                self.config.vocab_size,
                self.config.hidden_size,
                self.config.pad_token_id,
            )
        return embed_tokens

    def get_input_embeddings(self):
        return self.embed_tokens

    @classmethod
    def wrap_model_if_needed(cls, model: PreTrainedModel, rbln_config: "RBLNQwen3ModelConfig"):
        wrapper_cfg = {
            "max_seq_len": rbln_config.max_seq_len,
            "attn_impl": rbln_config.attn_impl,
            "kvcache_partition_len": rbln_config.kvcache_partition_len,
            "kvcache_block_size": rbln_config.kvcache_block_size,
            "use_rotary_emb": cls._use_rotary_emb,
            "use_attention_mask": rbln_config.use_attention_mask,
            "cache_impl": rbln_config.cache_impl,
            "sliding_window": rbln_config.sliding_window,
            "sliding_window_layers": rbln_config.sliding_window_layers,
        }
        return cls._decoder_wrapper_cls(model, **wrapper_cfg).eval()

    @classmethod
    @torch.inference_mode()
    def get_compiled_model(cls, model: "PreTrainedModel", rbln_config: RBLNQwen3ModelConfig):
        wrapped_model = cls.wrap_model_if_needed(model, rbln_config)

        rbln_compile_configs = rbln_config.compile_cfgs
        prefill_compile_config = rbln_compile_configs[0]

        context = CompileContext(use_weight_sharing=False)

        meta_tensor_names = [name for name, _, _ in prefill_compile_config.input_info if "past_key_values" in name]
        prefill_example_inputs = prefill_compile_config.get_dummy_inputs(fill=0, meta_tensor_names=meta_tensor_names)

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
                    create_runtimes=rbln_config.create_runtimes,
                    device=rbln_config.device,
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
        rbln_config: RBLNQwen3ModelConfig,
        model_config: PretrainedConfig,
    ):
        input_info = RBLNDecoderOnlyModelForCausalLM.get_input_info(
            batch_size,
            query_length,
            rbln_config=rbln_config,
            model_config=model_config,
        )

        if rbln_config.sliding_window is None:
            # remove query position
            input_info.pop(3)

        return input_info

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Optional[Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"]] = None,
        model: Optional["PreTrainedModel"] = None,
        model_config: Optional["PretrainedConfig"] = None,
        rbln_config: Optional[RBLNQwen3ModelConfig] = None,
    ) -> RBLNQwen3ModelConfig:
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

        # only compile prefill cb -> always batch_size 1
        required_num_blocks = rbln_config.max_seq_len // rbln_config.kvcache_block_size
        max_num_blocks = required_num_blocks

        if rbln_config.attn_impl == "flash_attn":
            estimated_max_num_blocks = RBLNDecoderOnlyModelForCausalLM.get_maximum_num_blocks(
                config=model_config,
                tensor_parallel_size=rbln_config.tensor_parallel_size or 1,
                kvcache_block_size=rbln_config.kvcache_block_size,
                nbits_per_param=16 if not rbln_config.quantization else 4,
                n_model_params=sum(p.numel() for p in model.parameters()),
                num_runtimes=1 + len(rbln_config.decoder_batch_sizes),
            )

            max_num_blocks = min(max_num_blocks, estimated_max_num_blocks)

            flash_min_blocks = rbln_config.max_seq_len // rbln_config.kvcache_block_size + 1
            if max_num_blocks < flash_min_blocks:
                max_num_blocks = flash_min_blocks

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
        rbln_config: RBLNQwen3ModelConfig,
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

    def _preprocess_chunked_prefill(
        self,
        inputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embed: Optional[torch.Tensor] = None,
    ):
        # valid sequence length of inputs_embeds
        query_length = inputs.shape[1] if attention_mask is None else torch.sum(attention_mask.view(-1)).item()

        # extract valid inputs
        inputs = inputs[:, attention_mask.bool()] if attention_mask is not None else inputs
        if position_embed is not None:
            position_embed = (
                position_embed[:, :, :, attention_mask.bool(), :] if attention_mask is not None else position_embed
            )

        if self.rbln_config.use_attention_mask:
            chunked_attention_mask = (
                torch.zeros(
                    1, 1, self.rbln_config.prefill_chunk_size, self.rbln_config.max_seq_len, dtype=torch.float32
                )
                if self.rbln_config.use_attention_mask
                else None
            )
        else:
            chunked_attention_mask = None

        # padding for chunked prefill
        padding_size = (
            self.rbln_config.prefill_chunk_size - (query_length % self.rbln_config.prefill_chunk_size)
        ) % self.rbln_config.prefill_chunk_size
        padded_len = query_length + padding_size

        inputs = torch.nn.functional.pad(inputs, (0, padding_size))
        position_embed = (
            None if position_embed is None else torch.nn.functional.pad(position_embed, (0, 0, 0, padding_size))
        )
        cache_position = torch.arange(padded_len, dtype=torch.int32).unsqueeze(0)

        return inputs, chunked_attention_mask, position_embed, cache_position, query_length

    def _chunked_prefill_forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embed: Optional[torch.Tensor] = None,
    ):
        padded_input, chunked_attention_mask, padded_position_embed, cache_position, query_length = (
            self._preprocess_chunked_prefill(inputs_embeds, attention_mask, position_embed)
        )

        # chunked prefill
        last_hidden_states = []
        for step in range(0, query_length, self.rbln_config.prefill_chunk_size):
            # Extract the current chunk of inputs and cache positions
            input_chunk = padded_input[:, step : step + self.rbln_config.prefill_chunk_size]
            cache_pos_chunk = cache_position[:, step : step + self.rbln_config.prefill_chunk_size]

            model_args = {
                "input_ids": input_chunk,
                "cache_position": cache_pos_chunk,
                "block_tables": self.block_tables,
            }

            if chunked_attention_mask is not None:
                if step >= self.rbln_config.prefill_chunk_size:
                    chunked_attention_mask[:, :, :, step - self.rbln_config.prefill_chunk_size : step] = 1
                chunked_attention_mask[:, :, :, step : step + self.rbln_config.prefill_chunk_size] = self.causal_mask
                model_args["attention_mask"] = chunked_attention_mask

            last_hidden_states_chunk = self.model[0](**model_args)
            last_hidden_states.append(last_hidden_states_chunk)

        last_hidden_states = torch.concat(last_hidden_states, dim=-2)[:, :query_length]

        return self._postprocess_chunked_prefill(last_hidden_states, attention_mask)

    def _postprocess_chunked_prefill(
        self, last_hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ):
        # index copy for attention mask
        if attention_mask is not None:
            new_last_hidden_states = torch.full(
                (1, attention_mask.shape[-1], last_hidden_states.shape[-1]),
                fill_value=1e-10,
                dtype=last_hidden_states.dtype,
            )
            mask_indices = torch.nonzero(attention_mask, as_tuple=True)[0]
            new_last_hidden_states.index_copy_(dim=-2, index=mask_indices, source=last_hidden_states)
        else:
            new_last_hidden_states = last_hidden_states
        return new_last_hidden_states

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        position_embed: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        inputs = inputs_embeds if inputs_embeds is not None else input_ids
        batch_size = inputs.shape[0]
        all_last_hidden_states = []
        for b_idx in range(batch_size):
            last_hidden_states = self._chunked_prefill_forward(
                inputs[b_idx : b_idx + 1],
                attention_mask[b_idx] if attention_mask is not None else None,
                position_embed[b_idx : b_idx + 1] if position_embed is not None else None,
            )
            all_last_hidden_states.append(last_hidden_states)

        return BaseModelOutputWithPast(last_hidden_state=torch.concat(all_last_hidden_states, dim=0))
