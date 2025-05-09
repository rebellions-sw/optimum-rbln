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
from typing import TYPE_CHECKING, Optional, Union

from transformers import PretrainedConfig, PreTrainedModel

from ....configuration_utils import RBLNCompileConfig
from ....utils.logging import get_logger
from ..decoderonly.decoderonly_architecture import (
    set_default_values,
    validate_attention_method,
)
from ..decoderonly.modeling_decoderonly import RBLNDecoderOnlyModelForCausalLM
from .configuration_gemma3 import RBLNGemma3ForCausalLMConfig
from .gemma3_architecture import Gemma3ForCausalLMWrapper


logger = get_logger()


if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer


class RBLNGemma3ForCausalLM(RBLNDecoderOnlyModelForCausalLM):
    """
    The Gemma3 Model transformer with a language modeling head (linear layer) on top.
    This model inherits from [`RBLNDecoderOnlyModelForCausalLM`]. Check the superclass documentation for the generic methods the library implements for all its models.

    A class to convert and run pre-trained transformers based Gemma3ForCausalLM model on RBLN devices.
    It implements the methods to convert a pre-trained transformers Gemma3ForCausalLM model into a RBLN transformer model by:
    - transferring the checkpoint weights of the original into an optimized RBLN graph,
    - compiling the resulting graph using the RBLN compiler.
    """

    _decoder_wrapper_cls = Gemma3ForCausalLMWrapper

    # @classmethod
    # def get_input_info(
    #     cls,
    #     batch_size: int,
    #     query_length: int,
    #     use_inputs_embeds: bool,
    #     use_attention_mask: bool,
    #     max_seq_len: int,
    #     kvcache_block_size: int,
    #     kvcache_num_blocks: int,
    #     num_key_value_heads: int,
    #     num_hidden_layers: int,
    #     hidden_size: int,
    #     head_dim: int,
    #     sliding_window: int,
    #     sliding_window_pattern: int,
    # ):
    #     if use_inputs_embeds:
    #         main_input = ("inputs_embeds", [batch_size, query_length, hidden_size], "float32")
    #     else:
    #         main_input = ("input_ids", [batch_size, query_length], "int64")

    #     input_info = [
    #         main_input,
    #         (
    #             "cache_position",
    #             [batch_size, query_length],
    #             "int32",
    #         ),
    #     ]

    #     if use_attention_mask:
    #         input_info.extend(
    #             [
    #                 ("attention_mask", [batch_size, 1, query_length, max_seq_len], "float32"),
    #             ]
    #         )

    #     if query_length > 1:
    #         input_info.extend(
    #             [
    #                 ("query_position", [], "int16"),
    #             ]
    #         )

    #     # different from the RBLNDecoderOnlyModelForCausalLM
    #     # local_kvcache_block_size = 1024
    #     # max_local_block_cnt = model_config.sliding_window // local_kvcache_block_size + 1
    #     max_global_block_cnt = max_seq_len // kvcache_block_size
    #     if query_length > 1:
    #         input_info.extend([("block_tables", [max_global_block_cnt], "int16")])
    #         input_info.extend([("local_block_tables", [1], "int16")])
    #     else:
    #         input_info.extend([("block_tables", [batch_size, max_global_block_cnt], "int16")])
    #         input_info.extend([("local_block_tables", [batch_size, 1], "int16")])

    #     def is_sliding(layer_idx: int) -> bool:
    #         return bool((layer_idx + 1) % sliding_window_pattern)

    #     local_kvcache_shape = [batch_size, num_key_value_heads, sliding_window, head_dim]
    #     global_kvcache_shape = [kvcache_num_blocks, num_key_value_heads, kvcache_block_size, head_dim]
    #     input_info.extend(
    #         [
    #             (
    #                 f"past_key_values_{i}",
    #                 local_kvcache_shape if is_sliding(i // 2) else global_kvcache_shape,
    #                 "float32",
    #             )
    #             for i in range(num_hidden_layers * 2)
    #         ]
    #     )

    #     return input_info

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Optional[Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"]] = None,
        model: Optional["PreTrainedModel"] = None,
        model_config: Optional["PretrainedConfig"] = None,
        rbln_config: Optional[RBLNGemma3ForCausalLMConfig] = None,
    ) -> RBLNGemma3ForCausalLMConfig:
        if rbln_config.max_seq_len is None:
            rbln_config.max_seq_len = getattr(model_config, "max_position_embeddings", None)
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

        if rbln_config.attn_impl == "flash_attn":
            # TODO(taehoon): override the get_maximum_num_blocks function
            estimated_max_num_blocks = cls.get_maximum_num_blocks(
                config=model_config,
                tensor_parallel_size=rbln_config.tensor_parallel_size or 1,
                kvcache_block_size=rbln_config.kvcache_block_size,
                nbits_per_param=16 if not rbln_config.quantization else 4,  # TODO(jongho): FIX Ad-hoc
                n_model_params=sum(p.numel() for p in model.parameters()),
            )

            max_num_blocks = min(max_num_blocks, estimated_max_num_blocks)

            flash_min_blocks = rbln_config.max_seq_len // rbln_config.kvcache_block_size + 1
            if max_num_blocks < flash_min_blocks:
                max_num_blocks = flash_min_blocks

            if max_num_blocks < rbln_config.batch_size:
                raise RuntimeError(
                    f"Batch size ({rbln_config.batch_size}) exceeds available KV cache blocks ({max_num_blocks}). "
                    "Ensure the number of blocks is at least equal to the batch size."
                )

        if rbln_config.kvcache_num_blocks is None:
            rbln_config.kvcache_num_blocks = max_num_blocks
        elif rbln_config.kvcache_num_blocks > max_num_blocks:
            logger.warning(
                f"The set `kvcache_num_blocks` ({rbln_config.kvcache_num_blocks}) is greater"
                f" than the estimated maximum number of blocks ({max_num_blocks})."
                "This can cause a failure during model compilation."
            )
        logger.info(f"[KVCache] Compiling with num_blocks: {rbln_config.kvcache_num_blocks}")
        num_attention_heads = getattr(model_config, "n_head", None) or getattr(model_config, "num_attention_heads")
        num_key_value_heads = getattr(model_config, "num_key_value_heads", None) or num_attention_heads
        num_hidden_layers = getattr(model_config, "n_layer", None) or getattr(model_config, "num_hidden_layers")
        hidden_size = getattr(model_config, "n_embd", None) or getattr(model_config, "hidden_size")
        head_dim = getattr(model_config, "head_dim", None) or hidden_size // num_attention_heads
        sliding_window = getattr(model_config, "sliding_window", None)
        sliding_window_pattern = getattr(model_config, "sliding_window_pattern", None)

        prefill_input_info = cls.get_input_info(
            batch_size=1,
            query_length=rbln_config.prefill_chunk_size,
            use_inputs_embeds=rbln_config.use_inputs_embeds,
            use_attention_mask=rbln_config.use_attention_mask,
            max_seq_len=rbln_config.max_seq_len,
            kvcache_block_size=rbln_config.kvcache_block_size,
            kvcache_num_blocks=rbln_config.kvcache_num_blocks,
            num_key_value_heads=num_key_value_heads,
            num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_size,
            head_dim=head_dim,
            sliding_window=sliding_window,
            sliding_window_pattern=sliding_window_pattern,
        )
        dec_input_info = cls.get_input_info(
            batch_size=rbln_config.batch_size,
            query_length=1,
            use_inputs_embeds=rbln_config.use_inputs_embeds,
            use_attention_mask=rbln_config.use_attention_mask,
            max_seq_len=rbln_config.max_seq_len,
            kvcache_block_size=rbln_config.kvcache_block_size,
            kvcache_num_blocks=rbln_config.kvcache_num_blocks,
            num_key_value_heads=num_key_value_heads,
            num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_size,
            head_dim=head_dim,
            sliding_window=sliding_window,
            sliding_window_pattern=sliding_window_pattern,
        )

        prefill_compile_config = RBLNCompileConfig(compiled_model_name="prefill", input_info=prefill_input_info)
        dec_compile_config = RBLNCompileConfig(compiled_model_name="decoder", input_info=dec_input_info)

        rbln_config.set_compile_cfgs([prefill_compile_config, dec_compile_config])

        return rbln_config