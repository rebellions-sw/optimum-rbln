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

import inspect
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple, Union

import rebel
import torch
from rebel.compile_context import CompileContext
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import no_init_weights

from ....configuration_utils import RBLNCompileConfig
from ....modeling import RBLNModel
from ....utils.logging import get_logger
from ...modeling_attention_utils import (
    RBLNDecoderOnlyFlashAttentionMixin,
    set_default_values,
    validate_attention_method,
    validate_sliding_window,
)
from ...modeling_outputs import RBLNDecoderOnlyOutput
from ...utils.rbln_quantization import prepare_model_for_quantization
from .configuration_decoderonly import RBLNDecoderOnlyModelConfig, RBLNDecoderOnlyModelForCausalLMConfig
from .decoderonly_architecture import DecoderOnlyWrapper
from .decoderonly_runtime_utils import RBLNPageTableManager, RBLNRuntimeModel
from .generation_decoderonly import RBLNDecoderOnlyGenerationMixin


logger = get_logger()

if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer


class RBLNDecoderOnlyModel(RBLNModel, RBLNDecoderOnlyFlashAttentionMixin):
    """
    A base class for decoder-only transformer models outputting raw hidden-states without any specific head on top.
    This class is used for RBLN-optimized models that are not causal language models.
    This class serves as the foundation for various decoder-only architectures like GPT, LLaMA, etc.

    The class provides core functionality for:

    1. Converting pre-trained transformer models to RBLN-optimized format
    2. Handling the compilation process for RBLN devices
    3. Managing inference operations for decoder-only architectures

    This class inherits from RBLNModel and implements specific methods required for
    decoder-only architectures.

    Note:
        - This class is designed to be subclassed by specific model implementations
          (e.g., RBLNLlamaModel, RBLNQwen2Model)
        - Subclasses should implement model-specific conversion logic.
        - The class handles RBLN-specific optimizations automatically during compilation
    """

    main_input_name = "input_ids"
    auto_model_class = AutoModel
    _decoder_wrapper_cls = DecoderOnlyWrapper
    _use_rotary_emb = True

    def __post_init__(self, **kwargs):
        if self.rbln_config.use_inputs_embeds:
            artifacts = torch.load(self.model_save_dir / self.subfolder / "torch_artifacts.pth", weights_only=False)
            self.embed_tokens = self._create_embedding_layer()
            self.embed_tokens.load_state_dict(artifacts["embed_tokens"])
        else:
            self.embed_tokens = None

        self.setup_runtime()

    def setup_runtime(self):
        # Initialize resources to be used across Runtime instances (prefill and decode phases)
        page_table_manager = RBLNPageTableManager(self.rbln_config)
        dec_attn_mask = torch.zeros(
            self.rbln_config.batch_size, 1, 1, self.rbln_config.max_seq_len, dtype=torch.float32
        )
        out_buffers = [torch.empty(self.prefill_output_size, dtype=torch.float32, device="cpu")]

        common_kwargs = {
            "main_input_name": "inputs_embeds" if self.rbln_config.use_inputs_embeds else "input_ids",
            "embed_tokens": self.embed_tokens,
            "dec_attn_mask": dec_attn_mask,
            "page_table_manager": page_table_manager,
            "rbln_config": self.rbln_config,
        }
        self.prefill_decoder = RBLNRuntimeModel(
            runtime=self.model[0],
            phase="prefill",
            batch_size=self.rbln_config.batch_size,
            out_buffers=out_buffers,
            **common_kwargs,
        )
        if self.can_generate():
            self.decoders = {}
            for i, batch_size in enumerate(self.rbln_config.decoder_batch_sizes):
                self.decoders[batch_size] = RBLNRuntimeModel(
                    runtime=self.model[i + 1],
                    phase="decode",
                    batch_size=batch_size,
                    **common_kwargs,
                )

            # NOTE(eunji): Use a decoder whose batch size matches the model's main batch size for compatibility.
            self.decoder = self.decoders[self.rbln_config.batch_size]

    @property
    def prefill_output_size(self):
        return (
            1,
            self.rbln_config.prefill_chunk_size if self.rbln_config.logits_to_keep == 0 else 1,
            self.config.hidden_size,
        )

    @classmethod
    def get_quantized_model(
        cls,
        model_id: str,
        config: Optional[PretrainedConfig] = None,
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        subfolder: str = "",
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        rbln_config: Optional[RBLNDecoderOnlyModelConfig] = None,
        **kwargs,
    ):
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
            if config.torch_dtype == torch.bfloat16:
                # FIXME: bfloat16 is not supported by rebel-compiler
                config.torch_dtype = torch.float32

        with no_init_weights():
            model = cls.auto_model_class.from_config(config)

        model = prepare_model_for_quantization(
            model,
            model_id,
            kwargs.get("num_hidden_layers"),
            use_auth_token=use_auth_token,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            rbln_quantization=rbln_config.quantization,
        )
        return model

    def __getattr__(self, __name: str) -> Any:
        # Special method to delegate attribute access to the original Huggingface LM class.
        # This method is called when an attribute is not found in the current instance's dictionary.
        # It enables transparent access to the original model's attributes and methods while maintaining
        # proper method binding.

        # The method implements a delegation pattern that:

        # 1. For methods: Creates a wrapper that properly binds 'self' to method calls
        # 2. For other attributes: Returns them directly from the original class

        def redirect(func):
            return lambda *pargs, **kwargs: func(self, *pargs, **kwargs)

        val = getattr(self.get_hf_class(), __name, None) or getattr(PreTrainedModel, __name)
        if isinstance(val, Callable) and "self" in set(inspect.signature(val).parameters):
            return redirect(val)
        return val

    @classmethod
    def save_torch_artifacts(
        cls,
        model: PreTrainedModel,
        save_dir_path: Path,
        subfolder: str,
        rbln_config: RBLNDecoderOnlyModelForCausalLMConfig,
    ):
        # If you are unavoidably running on a CPU rather than an RBLN device,
        # store the torch tensor, weight, etc. in this function.
        if rbln_config.use_inputs_embeds:
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

    def get_decoder(self):
        if not self.can_generate():
            raise ValueError("Decode stage is not supported in this model.")
        return self.decoder

    def can_generate(self):
        return self.rbln_config.can_generate

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_attn_impl(self) -> str:
        return self.rbln_config.attn_impl

    def get_kvcache_num_blocks(self) -> int:
        return self.rbln_config.kvcache_num_blocks

    @classmethod
    def wrap_model_if_needed(cls, model: PreTrainedModel, rbln_config: "RBLNDecoderOnlyModelConfig"):
        return cls._decoder_wrapper_cls(model, rbln_config, cls._use_rotary_emb).eval()

    @classmethod
    def _compile_model(
        cls,
        wrapped_model,
        compile_config,
        example_inputs,
        compile_context,
        rbln_config: RBLNDecoderOnlyModelForCausalLMConfig,
        quantization=None,
        phase: str = "prefill",
    ):
        try:
            wrapped_model.phase = phase
            if quantization:
                quantization.maybe_set_quantization_env()
            original_linear = torch.nn.functional.linear
            torch.nn.functional.linear = torch.ops.rbln_custom_ops.linear
            compiled_model = cls.compile(
                wrapped_model,
                compile_config,
                create_runtimes=rbln_config.create_runtimes,
                device=rbln_config.device,
                example_inputs=example_inputs,
                compile_context=compile_context,
            )
            return compiled_model
        finally:
            torch.nn.functional.linear = original_linear
            if quantization:
                quantization.maybe_reset_quantization_env()

    @classmethod
    def _get_compile_context(
        cls,
        compile_config: RBLNCompileConfig,
        example_inputs: List[torch.Tensor],
    ):
        context = CompileContext(use_weight_sharing=True)

        # Mark static tensors (self kv states)
        static_tensors = {}
        for (name, _, _), tensor in zip(compile_config.input_info, example_inputs):
            if "past_key_values" in name:
                static_tensors[name] = tensor
                context.mark_static_address(tensor)

        return context, static_tensors

    @classmethod
    @torch.inference_mode()
    def get_compiled_model(cls, model: PreTrainedModel, rbln_config: RBLNDecoderOnlyModelForCausalLMConfig):
        wrapped_model = cls.wrap_model_if_needed(model, rbln_config)
        prefill_compile_config = rbln_config.compile_cfgs[0]

        # Here we use meta tensor, for the memory efficiency.
        meta_tensor_names = [name for name, _, _ in prefill_compile_config.input_info if "past_key_values" in name]
        prefill_example_inputs = prefill_compile_config.get_dummy_inputs(fill=0, meta_tensor_names=meta_tensor_names)
        context, static_tensors = cls._get_compile_context(prefill_compile_config, prefill_example_inputs)

        compiled_models = {}
        compiled_models["prefill"] = cls._compile_model(
            wrapped_model,
            prefill_compile_config,
            prefill_example_inputs,
            context,
            rbln_config,
            rbln_config.quantization,
            phase="prefill",
        )

        if rbln_config.can_generate:
            wrapped_model.phase = "decode"
            for batch_size, dec_compile_config in zip(rbln_config.decoder_batch_sizes, rbln_config.compile_cfgs[1:]):
                dec_example_inputs = dec_compile_config.get_dummy_inputs(fill=0, static_tensors=static_tensors)
                compiled_decoder = cls._compile_model(
                    wrapped_model,
                    dec_compile_config,
                    dec_example_inputs,
                    context,
                    rbln_config,
                    rbln_config.quantization,
                    phase="decode",
                )
                compiled_models[f"decoder_batch_{batch_size}"] = compiled_decoder

            # check if the memory is enough to have additional blocks
            required_num_blocks = (rbln_config.max_seq_len // rbln_config.kvcache_block_size) * rbln_config.batch_size
            if rbln_config.kvcache_num_blocks < required_num_blocks:
                cls.maybe_suggest_kvcache_num_blocks(
                    compiled_models=compiled_models,
                    model_config=model.config,
                    rbln_config=rbln_config,
                )

        return compiled_models

    @classmethod
    def get_pytorch_model(
        cls, *args, rbln_config: Optional[RBLNDecoderOnlyModelConfig] = None, **kwargs
    ) -> PreTrainedModel:
        if rbln_config and rbln_config.quantization:
            model = cls.get_quantized_model(*args, rbln_config=rbln_config, **kwargs)
        else:
            model = super().get_pytorch_model(*args, **kwargs)

        return model

    @classmethod
    def use_query_position(cls, use_local_attention: bool, is_prefill: bool = True):
        return use_local_attention

    @classmethod
    def get_input_info(
        cls,
        batch_size: int,
        query_length: int,
        rbln_config: RBLNDecoderOnlyModelForCausalLMConfig,
        model_config: PretrainedConfig,
    ):
        num_attention_heads = getattr(model_config, "n_head", None) or getattr(model_config, "num_attention_heads")
        num_key_value_heads = getattr(model_config, "num_key_value_heads", None) or num_attention_heads
        num_hidden_layers = getattr(model_config, "n_layer", None) or getattr(model_config, "num_hidden_layers")
        hidden_size = getattr(model_config, "n_embd", None) or getattr(model_config, "hidden_size")
        head_dim = getattr(model_config, "head_dim", None) or hidden_size // num_attention_heads
        is_prefill = query_length > 1

        input_info = []
        if rbln_config.use_inputs_embeds:
            input_info.append(("inputs_embeds", [batch_size, query_length, hidden_size], "float32"))
        else:
            input_info.append(("input_ids", [batch_size, query_length], "int64"))

        input_info.append(("cache_position", [batch_size, query_length], "int32"))

        if rbln_config.use_global_attention:
            max_block_cnt = rbln_config.max_seq_len // rbln_config.kvcache_block_size
            input_info.append(
                ("block_tables", [max_block_cnt] if is_prefill else [batch_size, max_block_cnt], "int16")
            )
        if rbln_config.use_local_attention:
            input_info.append(("local_block_tables", [1] if is_prefill else [batch_size, 1], "int16"))

        if cls.use_query_position(rbln_config.use_local_attention, is_prefill):
            input_info.append(("query_position", [], "int16"))

        if rbln_config.use_attention_mask:
            if rbln_config.use_position_ids:
                input_info.append(("attention_mask", [batch_size, rbln_config.max_seq_len], "float32"))
            else:
                input_info.append(
                    ("attention_mask", [batch_size, 1, query_length, rbln_config.max_seq_len], "float32")
                )

        if rbln_config.use_position_ids:
            input_info.append(("position_ids", [batch_size, query_length], "int32"))

        kvcache_dtype = "float32"
        if rbln_config.quantization and rbln_config.quantization.kv_caches == "fp8":
            kvcache_dtype = "float8_e4m3fn"

        global_kvcache_shape = [
            rbln_config.kvcache_num_blocks,
            num_key_value_heads,
            rbln_config.kvcache_block_size,
            head_dim,
        ]
        local_kvcache_shape = [rbln_config.batch_size, num_key_value_heads, rbln_config.sliding_window, head_dim]
        input_info.extend(
            [
                (
                    f"past_key_values_{i}",
                    local_kvcache_shape
                    if rbln_config.sliding_window is not None and ((i // 2) in rbln_config.sliding_window_layers)
                    else global_kvcache_shape,
                    kvcache_dtype,
                )
                for i in range(num_hidden_layers * 2)
            ]
        )

        return input_info

    @classmethod
    def _update_sliding_window_config(
        cls, model_config: PretrainedConfig, rbln_config: RBLNDecoderOnlyModelForCausalLMConfig
    ):
        # Update the sliding window configuration for the RBLN model.

        # This method must be implemented by subclasses to handle their specific sliding window configurations,
        # as Hugging Face models use different configuration keys to represent sliding window layers.

        # Args:
        #     model_config (PretrainedConfig): The model configuration from Hugging Face.
        #     rbln_config (RBLNDecoderOnlyModelForCausalLMConfig): The RBLN model configuration.

        # Notes:
        #     Required configuration settings:
        #     - `cache_impl`: Must be one of:
        #         - "static": All layers use global attention (no sliding window)
        #         - "sliding_window": All layers use sliding window attention
        #         - "hybrid": A mix of global and sliding window attention layers
        #     - `sliding_window`: Width of the sliding window (required if cache_impl is "sliding_window" or "hybrid")
        #     - `sliding_window_layers`: List of layer indices using sliding window attention (required if cache_impl is "hybrid")

        #     Example implementation for a 'sliding_window' model:
        #     ```python
        #     rbln_config.cache_impl = "sliding_window"
        #     rbln_config.sliding_window = model_config.sliding_window
        #     rbln_config.sliding_window_layers = [i for i in range(model_config.num_hidden_layers)]
        #     return rbln_config
        #     ```

        # Returns:
        #     RBLNDecoderOnlyModelConfig: The updated RBLN model configuration.

        raise NotImplementedError(
            "Subclasses must implement _update_sliding_window_config to configure sliding window attention settings. "
            "See method docstring for required configuration details."
        )

    @classmethod
    def _update_attention_config(
        cls, model: PreTrainedModel, model_config: PretrainedConfig, rbln_config: RBLNDecoderOnlyModelForCausalLMConfig
    ):
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

        num_full_blocks = (rbln_config.max_seq_len // rbln_config.kvcache_block_size) * rbln_config.batch_size

        # Update kvcache_num_blocks based on the attention implementation.
        if rbln_config.attn_impl == "flash_attn":
            estimated_max_num_blocks = cls.get_maximum_num_blocks(
                config=model_config,
                tensor_parallel_size=rbln_config.tensor_parallel_size or 1,
                kvcache_block_size=rbln_config.kvcache_block_size,
                nbits_per_param=16 if not rbln_config.quantization else 4,  # TODO(jongho): FIX Ad-hoc
                n_model_params=sum(p.numel() for p in model.parameters()),
                num_runtimes=1 if not rbln_config.can_generate else 1 + len(rbln_config.decoder_batch_sizes),
            )

            if rbln_config.kvcache_num_blocks is None:
                if estimated_max_num_blocks < num_full_blocks:
                    # lower bound of the number of blocks for flash attention.
                    min_blocks_for_flash = min(
                        rbln_config.max_seq_len // rbln_config.kvcache_block_size + 1, num_full_blocks
                    )
                    if min_blocks_for_flash > estimated_max_num_blocks:
                        # NOTE: Just try to compile with lower bound of blocks for flash attention.
                        # Even if it's larger than the estimated maximum number of blocks.
                        rbln_config.kvcache_num_blocks = min_blocks_for_flash
                    else:
                        logger.info(f"[KVCache] Compiling with num_blocks: {rbln_config.kvcache_num_blocks}")
                        rbln_config.kvcache_num_blocks = estimated_max_num_blocks

                    if rbln_config.kvcache_num_blocks < rbln_config.batch_size:
                        raise RuntimeError(
                            f"Batch size ({rbln_config.batch_size}) exceeds num_blocks ({rbln_config.kvcache_num_blocks}). "
                            "Ensure the number of blocks is at least equal to the batch size."
                        )
                else:
                    rbln_config.kvcache_num_blocks = num_full_blocks
            elif rbln_config.kvcache_num_blocks > estimated_max_num_blocks:
                logger.warning(
                    f"The set `kvcache_num_blocks` ({rbln_config.kvcache_num_blocks}) is greater"
                    f" than the estimated maximum number of blocks ({estimated_max_num_blocks})."
                    "This can cause a failure during model compilation."
                )
        else:
            if rbln_config.kvcache_num_blocks is None:
                rbln_config.kvcache_num_blocks = num_full_blocks
            elif rbln_config.kvcache_num_blocks > num_full_blocks:
                logger.warning(
                    f"The set `kvcache_num_blocks` ({rbln_config.kvcache_num_blocks}) is greater"
                    f" than the required number of blocks ({num_full_blocks})."
                    "This can cause a failure during model compilation."
                )

        logger.info(f"[KVCache] Compiling with num_blocks: {rbln_config.kvcache_num_blocks}")

        return rbln_config

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Optional[Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"]] = None,
        model: Optional[PreTrainedModel] = None,
        model_config: Optional[PretrainedConfig] = None,
        rbln_config: Optional[RBLNDecoderOnlyModelForCausalLMConfig] = None,
    ) -> RBLNDecoderOnlyModelForCausalLMConfig:
        if rbln_config.max_seq_len is None:
            rbln_config.max_seq_len = getattr(model_config, "max_position_embeddings", None) or getattr(
                model_config, "n_positions", None
            )
        if rbln_config.max_seq_len is None:
            raise ValueError("`max_seq_len` should be specified.")

        if getattr(model_config, "sliding_window", None) is not None and getattr(
            model_config, "use_sliding_window", True
        ):
            rbln_config = cls._update_sliding_window_config(model_config, rbln_config)
            if rbln_config.sliding_window is not None:
                validate_sliding_window(rbln_config)

        rbln_config = cls._update_attention_config(model, model_config, rbln_config)

        prefill_input_info = cls.get_input_info(
            batch_size=1,
            query_length=rbln_config.prefill_chunk_size,
            rbln_config=rbln_config,
            model_config=model_config,
        )

        prefill_compile_config = RBLNCompileConfig(compiled_model_name="prefill", input_info=prefill_input_info)
        compile_cfgs = [prefill_compile_config]

        if rbln_config.can_generate:
            for batch_size in rbln_config.decoder_batch_sizes:
                dec_input_info = cls.get_input_info(
                    batch_size=batch_size,
                    query_length=1,
                    rbln_config=rbln_config,
                    model_config=model_config,
                )
                compile_cfgs.append(
                    RBLNCompileConfig(compiled_model_name=f"decoder_batch_{batch_size}", input_info=dec_input_info)
                )
        rbln_config.set_compile_cfgs(compile_cfgs)

        return rbln_config

    @classmethod
    def _create_runtimes(
        cls,
        compiled_models: List[rebel.RBLNCompiledModel],
        rbln_config: RBLNDecoderOnlyModelForCausalLMConfig,
    ) -> List[rebel.Runtime]:
        expected_model_names = ["prefill"]
        if rbln_config.can_generate:
            expected_model_names.extend(
                [f"decoder_batch_{batch_size}" for batch_size in rbln_config.decoder_batch_sizes]
            )
        if any(model_name not in rbln_config.device_map for model_name in expected_model_names):
            cls._raise_missing_compiled_file_error(expected_model_names)

        ret_val = [
            rebel.Runtime(
                compiled_models[0],
                tensor_type="pt",
                device=rbln_config.device_map["prefill"],
                activate_profiler=rbln_config.activate_profiler,
                timeout=rbln_config.timeout,
            )
        ]
        if rbln_config.can_generate:
            ret_val.extend(
                [
                    rebel.Runtime(
                        compiled_models[i + 1],
                        tensor_type="pt",
                        device=rbln_config.device_map[f"decoder_batch_{batch_size}"],
                        activate_profiler=rbln_config.activate_profiler,
                        timeout=rbln_config.timeout,
                    )
                    for i, batch_size in enumerate(rbln_config.decoder_batch_sizes)
                ]
            )
        return ret_val

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        position_embed: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor]:
        inputs = inputs_embeds if inputs_embeds is not None else input_ids
        batch_size = inputs.shape[0]

        if batch_size != self.rbln_config.batch_size:
            raise ValueError(
                f"Batch size ({batch_size}) must be equal to the batch size of the model ({self.rbln_config.batch_size})."
            )

        all_last_hidden_states = []
        for b_idx in range(self.rbln_config.batch_size):
            query_length = (
                attention_mask[b_idx].sum(dim=-1).int().item() if attention_mask is not None else inputs.shape[1]
            )
            cache_position = torch.arange(query_length, dtype=torch.int32).unsqueeze(0)
            last_hidden_states = self.prefill_decoder(
                inputs[b_idx : b_idx + 1],
                attention_mask=attention_mask[b_idx] if attention_mask is not None else None,
                position_embed=position_embed[b_idx : b_idx + 1] if position_embed is not None else None,
                cache_position=cache_position,
                batch_idx=b_idx,
            ).logits
            all_last_hidden_states.append(last_hidden_states)

        last_hidden_states = torch.concat(all_last_hidden_states, dim=0)
        return BaseModelOutputWithPast(last_hidden_state=last_hidden_states)


class RBLNDecoderOnlyModelForCausalLM(RBLNDecoderOnlyModel, RBLNDecoderOnlyGenerationMixin):
    """
    A base class for decoder-only transformer models optimized for causal language modeling tasks on RBLN devices.
    This class serves as the foundation for various decoder-only architectures like GPT, LLaMA, etc.

    The class provides core functionality for:

    1. Converting pre-trained transformer models to RBLN-optimized format
    2. Handling the compilation process for RBLN devices
    3. Managing inference operations for causal language modeling

    This class inherits from RBLNModel and implements specific methods required for
    decoder-only architectures and causal language modeling tasks.

    Note:
        - This class is designed to be subclassed by specific model implementations
          (e.g., RBLNLlamaForCausalLM, RBLNGPT2LMHeadModel)
        - Subclasses should implement model-specific conversion logic.
        - The class handles RBLN-specific optimizations automatically during compilation
    """

    auto_model_class = AutoModelForCausalLM

    @property
    def prefill_output_size(self):
        return (
            1,
            self.rbln_config.prefill_chunk_size if self.rbln_config.logits_to_keep == 0 else 1,
            self.config.vocab_size,
        )

    @classmethod
    def use_query_position(cls, use_local_attention: bool, is_prefill: bool = True):
        return is_prefill

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        generate_idx: Optional[torch.Tensor] = None,
        padded_cache_lengths: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        return_dict: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor]:
        # Forward method for the RBLN-optimized model, designed for integration with the HuggingFace generate API.
        # For continuous batching, the prefill stage processes one batch at a time and updates the KV cache using batch_idx.
        # A for-loop ensures synchronization with the HuggingFace generate API.
        # The decoder stage operates as usual, processing inputs in batch mode.

        # for only use forward
        if generate_idx is None:
            generate_idx = (
                attention_mask.sum(dim=-1, keepdim=True).int()
                if attention_mask is not None
                else torch.full((input_ids.shape[0], 1), input_ids.shape[1], dtype=torch.int32)
            )
            padded_cache_lengths = torch.zeros_like(generate_idx)

        # Prefill
        if cache_position is None:
            logits = []
            inputs = inputs_embeds if inputs_embeds is not None else input_ids
            batch_size = inputs.shape[0]
            for b_idx in range(batch_size):
                cache_position = torch.arange(0, generate_idx[b_idx].item(), dtype=torch.int32).unsqueeze(0)
                output = self.prefill_decoder(
                    input_ids=inputs[b_idx : b_idx + 1] if inputs_embeds is None else None,
                    inputs_embeds=inputs[b_idx : b_idx + 1] if inputs_embeds is not None else None,
                    attention_mask=attention_mask[b_idx] if attention_mask is not None else None,
                    cache_position=cache_position,
                    batch_idx=b_idx,
                    token_type_ids=token_type_ids[b_idx : b_idx + 1] if token_type_ids is not None else None,
                )
                padded_cache_lengths[b_idx] += output.padded_cache_lengths
                logits.append(output.logits)
            logits = torch.cat(logits, dim=0)
        # Decoder
        else:
            inputs = inputs_embeds if inputs_embeds is not None else input_ids
            batch_size = inputs.shape[0]
            if batch_size not in self.decoders:
                raise ValueError(
                    f"No decoder runtime available for batch size {batch_size}. "
                    f"Available batch sizes are: {list(self.decoders.keys())}. "
                    f"Please run your model with one of these batch sizes or add support for batch size {batch_size}."
                )
            logits = self.decoders[batch_size](
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                position_ids=position_ids if self.rbln_config.use_position_ids else None,
            ).logits

        if not return_dict:
            return logits, generate_idx, padded_cache_lengths
        else:
            return RBLNDecoderOnlyOutput(
                logits=logits, generate_idx=generate_idx, padded_cache_lengths=padded_cache_lengths
            )
