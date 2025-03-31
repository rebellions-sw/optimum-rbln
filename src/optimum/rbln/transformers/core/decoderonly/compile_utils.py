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

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import rebel
import torch
from rebel.compile_context import CompileContext
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.modeling_utils import no_init_weights

from ....modeling import RBLNModel
from ....modeling_config import RBLNCompileConfig, RBLNConfig
from ....utils.logging import get_logger
from ...utils.rbln_quantization import QuantizationManager
from .decoderonly_architecture import (
    validate_attention_method,
)


logger = get_logger()

if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer


class DecoderOnlyCompileUtils:
    @classmethod
    def get_pytorch_model(cls, *args, **kwargs) -> "PreTrainedModel":
        logger.debug("Loading the LLM model to the CPU.")  # TODO(jongho): Remove.

        rbln_kwargs = kwargs.get("rbln_kwargs", {})
        rbln_quantization = rbln_kwargs.get("quantization", None)
        if rbln_quantization is not None and rbln_quantization["format"] == "rbln":
            model = cls.get_quantized_model(*args, **kwargs)
        else:
            model = super().get_pytorch_model(*args, **kwargs)

        logger.debug("Loaded the LLM model to the CPU.")
        return model

    @classmethod
    def wrap_model_if_needed(cls, model: "PreTrainedModel", rbln_config: "RBLNConfig"):
        wrapper_cfg = {"max_seq_len": rbln_config.model_cfg["max_seq_len"]}
        wrapper_cfg["attn_impl"] = rbln_config.model_cfg.get("attn_impl")
        wrapper_cfg["kvcache_partition_len"] = rbln_config.model_cfg.get("kvcache_partition_len")
        wrapper_cfg["kvcache_block_size"] = rbln_config.model_cfg.get("kvcache_block_size")
        wrapper_cfg["use_rotary_emb"] = cls._use_rotary_emb
        wrapper_cfg["use_attention_mask"] = rbln_config.model_cfg.get("use_attention_mask")

        return cls._decoder_wrapper_cls(model, **wrapper_cfg).eval()

    @classmethod
    @torch.inference_mode()
    def get_compiled_model(cls, model: "PreTrainedModel", rbln_config: RBLNConfig):
        wrapped_model = cls.wrap_model_if_needed(model, rbln_config)

        rbln_compile_configs = rbln_config.compile_cfgs
        prefill_compile_config = rbln_compile_configs[0]
        dec_compile_config = rbln_compile_configs[1]

        context = CompileContext(use_weight_sharing=True)

        # Here we use meta tensor, for the memory efficiency.
        meta_tensor_names = [name for name, _, _ in prefill_compile_config.input_info if "past_key_values" in name]
        prefill_example_inputs = prefill_compile_config.get_dummy_inputs(fill=0, meta_tensor_names=meta_tensor_names)

        # Mark static tensors (self kv states)
        static_tensors = {}
        for (name, _, _), tensor in zip(prefill_compile_config.input_info, prefill_example_inputs):
            if "past_key_values" in name:
                static_tensors[name] = tensor
                context.mark_static_address(tensor)

        dec_example_inputs = dec_compile_config.get_dummy_inputs(fill=0, static_tensors=static_tensors)

        quantize_config = rbln_config.model_cfg.get("quantization", None)

        @QuantizationManager.with_quantization_env
        def compile_model(*args, **kwargs):
            wrapped_model.phase = "prefill"
            compiled_prefill = RBLNModel.compile(
                wrapped_model,
                prefill_compile_config,
                example_inputs=prefill_example_inputs,
                compile_context=context,
            )

            wrapped_model.phase = "decode"
            compiled_decoder = RBLNModel.compile(
                wrapped_model,
                dec_compile_config,
                example_inputs=dec_example_inputs,
                compile_context=context,
            )
            return {"prefill": compiled_prefill, "decoder": compiled_decoder}

        return compile_model(quantize_config=quantize_config)

    @classmethod
    def _get_rbln_config(
        cls,
        preprocessors: Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"],
        model_config: "PretrainedConfig",
        rbln_kwargs: Dict[str, Any] = {},
    ) -> RBLNConfig:
        rbln_max_seq_len = rbln_kwargs.get("max_seq_len", None)
        rbln_batch_size = rbln_kwargs.get("batch_size", None)
        rbln_use_inputs_embeds = rbln_kwargs.get("use_inputs_embeds", None)
        rbln_use_attention_mask = rbln_kwargs.get("use_attention_mask", None)
        rbln_attn_impl = rbln_kwargs.get("attn_impl", None)
        rbln_kvcache_partition_len = rbln_kwargs.get("kvcache_partition_len", None)
        rbln_kvcache_block_size = rbln_kwargs.get("kvcache_block_size", None)
        rbln_quantization = QuantizationManager.validate_quantization_config(rbln_kwargs.get("quantization", None))
        rbln_prefill_chunk_size = rbln_kwargs.get("prefill_chunk_size", None)

        if rbln_use_attention_mask is None:
            rbln_use_attention_mask = False
            rbln_npu = rbln_kwargs.get("npu", None) or rebel.get_npu_name()
            if rbln_npu == "RBLN-CA02":
                rbln_use_attention_mask = True

        if rbln_prefill_chunk_size is None:
            rbln_prefill_chunk_size = 128
        elif rbln_prefill_chunk_size % 64 != 0 or rbln_prefill_chunk_size == 0:
            raise ValueError(
                f"Invalid rbln_prefill_chunk_size: {rbln_prefill_chunk_size}. It must be a nonzero multiple of 64."
            )

        if rbln_max_seq_len is None:
            rbln_max_seq_len = getattr(model_config, "max_position_embeddings", None) or getattr(
                model_config, "n_positions", None
            )
        if rbln_max_seq_len is None:
            raise ValueError("`rbln_max_seq_len` should be specified.")

        rbln_batch_size = 1 if rbln_batch_size is None else rbln_batch_size
        rbln_use_inputs_embeds = False if rbln_use_inputs_embeds is None else rbln_use_inputs_embeds

        rbln_attn_impl, rbln_kvcache_partition_len, rbln_kvcache_block_size = validate_attention_method(
            rbln_attn_impl=rbln_attn_impl,
            rbln_kvcache_partition_len=rbln_kvcache_partition_len,
            rbln_kvcache_block_size=rbln_kvcache_block_size,
            rbln_max_seq_len=rbln_max_seq_len,
        )

        if rbln_kvcache_block_size is None:
            if rbln_attn_impl == "eager":
                rbln_kvcache_block_size = rbln_max_seq_len
            else:
                rbln_kvcache_block_size = rbln_kvcache_partition_len

        rbln_kvcache_num_blocks = (rbln_max_seq_len // rbln_kvcache_block_size) * rbln_batch_size
        if rbln_attn_impl == "flash_attn":
            max_num_blocks, _ = cls.get_maximum_num_blocks(
                config=model_config,
                tensor_parallel_size=rbln_kwargs.get("tensor_parallel_size", 1),
                kvcache_block_size=rbln_kvcache_block_size,
                nbits_per_param=16 if rbln_quantization is None else 4,  # TODO(jongho): FIX Ad-hoc
                n_model_params=rbln_kwargs["n_model_params"],
            )
            rbln_kvcache_num_blocks = min(rbln_kvcache_num_blocks, max_num_blocks)

            required_blocks = rbln_max_seq_len // rbln_kvcache_block_size + 1
            if rbln_kvcache_num_blocks < required_blocks:
                rbln_kvcache_num_blocks = required_blocks

            logger.info(f"[KVCache] Compiling with num_blocks: {rbln_kvcache_num_blocks}")

            if rbln_kvcache_num_blocks < rbln_batch_size:
                raise RuntimeError(
                    f"Batch size ({rbln_batch_size}) exceeds available KV cache blocks ({rbln_kvcache_num_blocks}). "
                    "Ensure the number of blocks is at least equal to the batch size."
                )

        num_attention_heads = getattr(model_config, "n_head", None) or getattr(model_config, "num_attention_heads")
        num_key_value_heads = getattr(model_config, "num_key_value_heads", None) or num_attention_heads
        num_hidden_layers = getattr(model_config, "n_layer", None) or getattr(model_config, "num_hidden_layers")
        head_dim = getattr(model_config, "head_dim", None) or model_config.hidden_size // num_attention_heads
        hidden_size = getattr(model_config, "n_embd", None) or getattr(model_config, "hidden_size")

        def get_input_info(
            batch_size,
            query_length,
            use_inputs_embeds,
            hidden_size,
        ):
            if use_inputs_embeds:
                main_input = ("inputs_embeds", [batch_size, query_length, hidden_size], "float32")
            else:
                main_input = ("input_ids", [batch_size, query_length], "int64")

            input_info = [
                main_input,
                (
                    "cache_position",
                    [batch_size, query_length],
                    "int32",
                ),
            ]

            if rbln_use_attention_mask:
                input_info.extend(
                    [
                        ("attention_mask", [batch_size, 1, query_length, rbln_max_seq_len], "float32"),
                    ]
                )

            if query_length > 1:
                input_info.extend(
                    [
                        ("query_position", [], "int16"),
                    ]
                )

            max_block_cnt = rbln_max_seq_len // rbln_kvcache_block_size

            if query_length > 1:
                input_info.extend([("block_tables", [max_block_cnt], "int16")])
            else:
                input_info.extend([("block_tables", [batch_size, max_block_cnt], "int16")])

            input_info.extend(
                [
                    (
                        f"past_key_values_{i}",
                        [
                            rbln_kvcache_num_blocks,
                            num_key_value_heads,
                            rbln_kvcache_block_size,
                            head_dim,
                        ],
                        "float32",
                    )
                    for i in range(num_hidden_layers * 2)
                ]
            )

            return input_info

        prefill_input_info = get_input_info(
            batch_size=1,
            query_length=rbln_prefill_chunk_size,
            use_inputs_embeds=rbln_use_inputs_embeds,
            hidden_size=hidden_size,
        )
        dec_input_info = get_input_info(
            batch_size=rbln_batch_size,
            query_length=1,
            use_inputs_embeds=rbln_use_inputs_embeds,
            hidden_size=hidden_size,
        )

        prefill_compile_config = RBLNCompileConfig(compiled_model_name="prefill", input_info=prefill_input_info)
        dec_compile_config = RBLNCompileConfig(compiled_model_name="decoder", input_info=dec_input_info)

        rbln_config = RBLNConfig(
            rbln_cls=cls.__name__,
            compile_cfgs=[prefill_compile_config, dec_compile_config],
            rbln_kwargs=rbln_kwargs,
        )

        rbln_config.model_cfg.update(
            {
                "max_seq_len": rbln_max_seq_len,
                "batch_size": rbln_batch_size,
                "prefill_chunk_size": rbln_prefill_chunk_size,
                "use_attention_mask": rbln_use_attention_mask,
                "use_inputs_embeds": rbln_use_inputs_embeds,
                "kvcache_partition_len": rbln_kvcache_partition_len,
                "kvcache_block_size": rbln_kvcache_block_size,
                "attn_impl": rbln_attn_impl,
                "kvcache_num_blocks": rbln_kvcache_num_blocks,
            }
        )

        if rbln_quantization is not None:
            rbln_config.model_cfg.update({"quantization": rbln_quantization})

        return rbln_config

    def get_input_embeddings(self):
        return self.embed_tokens

    @classmethod
    def save_torch_artifacts(
        cls,
        model: "PreTrainedModel",
        save_dir_path: Path,
        subfolder: str,
        rbln_config: RBLNConfig,
    ):
        """
        If you are unavoidably running on a CPU rather than an RBLN device,
        store the torch tensor, weight, etc. in this function.
        """
        if rbln_config.model_cfg["use_inputs_embeds"]:
            save_dict = {}
            save_dict["embed_tokens"] = model.get_input_embeddings().state_dict()
            torch.save(save_dict, save_dir_path / subfolder / "torch_artifacts.pth")

    @classmethod
    def get_quantized_model(
        cls,
        model_id: str,
        config: Optional["PretrainedConfig"] = None,
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        subfolder: str = "",
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        **kwargs,
    ):
        from ...utils.rbln_quantization import prepare_model_for_quantization

        kwargs = cls.update_kwargs(kwargs)

        if config is None:
            config = AutoConfig.from_pretrained(
                model_id,
                use_auth_token=use_auth_token,
                revision=revision,
                force_download=force_download,
                cache_dir=cache_dir,
                trust_remote_code=trust_remote_code,
                **kwargs,
            )

        with no_init_weights():
            model = AutoModelForCausalLM.from_config(config)

        prepare_model_for_quantization(model, model_id, kwargs.get("num_hidden_layers"))

        return model

    @classmethod
    def get_maximum_num_blocks(
        cls,
        config: PretrainedConfig,
        tensor_parallel_size: int,
        kvcache_block_size: int,
        nbits_per_param: int,
        n_model_params: int,
    ) -> int:
        def align(x: int, nbytes: int) -> int:
            return int(math.ceil(x / nbytes) * nbytes)

        def align_2MB(x: int) -> int:
            return align(x, 2 * 1024 * 1024)

        num_attention_heads = getattr(config, "n_head", None) or getattr(config, "num_attention_heads")
        num_layers = getattr(config, "n_layer", None) or getattr(config, "num_hidden_layers")
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // num_attention_heads
        vocab_size = config.vocab_size
        hidden_size = getattr(config, "n_embd", None) or getattr(config, "hidden_size")
        num_key_value_heads = getattr(config, "num_key_value_heads", None) or num_attention_heads

        # TODO(jongho): Update if target npu is REBEL.
        ATOM_DRAM_NBYTES = 16 * 2**30
        ATOM_SYS_DRAM_NBYTES = 288 * 2**20
        available_dram = tensor_parallel_size * (ATOM_DRAM_NBYTES - ATOM_SYS_DRAM_NBYTES)

        # Get estimated kernel size (approximated)
        lm_heads_params = align(vocab_size, 64) * hidden_size
        lm_heads_nbytes = (
            align_2MB(lm_heads_params * nbits_per_param // 8 / tensor_parallel_size) * tensor_parallel_size
        )
        params = n_model_params - lm_heads_params
        layer_nbytes = (
            align_2MB(params * nbits_per_param // 8 / num_layers / tensor_parallel_size)
            * num_layers
            * tensor_parallel_size
        )
        kernel_size = layer_nbytes + lm_heads_nbytes

        available_dram -= kernel_size

        # TODO: Accurate buffer estimation
        buffer = 2**30  # 1GB Buffer
        if tensor_parallel_size <= 4:
            buffer /= 4

        available_dram -= buffer

        # Estimate nbytes per a single kvcache block
        nbytes_per_block = (
            align_2MB(
                kvcache_block_size
                * head_dim
                * math.ceil(num_key_value_heads / tensor_parallel_size)  # Shard
                * 2  # (fp16)
            )
            * num_layers
            * 2  # (k, v)
            * tensor_parallel_size
        )
        n_blocks = available_dram // nbytes_per_block

        return n_blocks, nbytes_per_block
