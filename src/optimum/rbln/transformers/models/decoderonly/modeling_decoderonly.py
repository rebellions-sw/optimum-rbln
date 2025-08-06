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
import torch.nn.functional as F
from rebel.compile_context import CompileContext
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import no_init_weights

from ....configuration_utils import RBLNCompileConfig
from ....modeling import RBLNModel
from ....utils.logging import get_logger
from ....utils.runtime_utils import RBLNPytorchRuntime
from ...modeling_attention_utils import (
    RBLNDecoderOnlyFlashAttentionMixin,
    set_default_values,
    validate_attention_method,
    validate_sliding_window,
)
from ...modeling_outputs import RBLNDecoderOnlyForCausalLMOutput
from ...utils.rbln_quantization import prepare_model_for_quantization
from .configuration_decoderonly import RBLNDecoderOnlyModelConfig, RBLNDecoderOnlyModelForCausalLMConfig
from .decoderonly_architecture import DecoderOnlyWrapper
from .generation_decoderonly import (
    RBLNDecoderOnlyGenerationMixin,
)
from .page_table_manager import RBLNPageTableManager


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
        self.prefill_runtime = RBLNPytorchRuntime(runtime=self.model[0])
        self.setup_forward_components()

    def setup_forward_components(self):
        # setup attributes for prefill inference
        self.block_tables = (
            torch.arange(self.rbln_config.kvcache_num_blocks, dtype=torch.int16)
            if self.rbln_config.use_global_attention
            else None
        )
        self.local_block_tables = (
            torch.tensor([0], dtype=torch.int16) if self.rbln_config.use_local_attention else None
        )
        if self.rbln_config.use_attention_mask:
            self.causal_mask = 1 - torch.triu(
                torch.ones(1, 1, self.rbln_config.prefill_chunk_size, self.rbln_config.prefill_chunk_size), diagonal=1
            )
        output_size = (
            1,
            self.rbln_config.prefill_chunk_size if self.rbln_config.logits_to_keep == 0 else 1,
            self.config.hidden_size,
        )
        # Buffer for storing output logits
        self.out_buffers = [torch.empty(output_size, dtype=torch.float32, device="cpu")]

        self.page_table_manager = RBLNPageTableManager(self.rbln_config)

        # # FIXME: this is a hack to keep backward compatibility with the old generation API
        # self.prefill_decoder = self._prefill_forward
        # self.decoder = self._decode_forward
        # if self.can_generate():
        #     self.decoders = {}
        #     for batch_size in self.rbln_config.decoder_batch_sizes:
        #         self.decoders[batch_size] = self._decode_forward

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
            original_linear = F.linear
            F.linear = torch.ops.rbln_custom_ops.linear
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
            F.linear = original_linear
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
            model = cls.get_quantized_model(*args, **kwargs)
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
                    "float32",
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

        required_num_blocks = (rbln_config.max_seq_len // rbln_config.kvcache_block_size) * rbln_config.batch_size
        max_num_blocks = required_num_blocks

        if rbln_config.attn_impl == "flash_attn":
            estimated_max_num_blocks = cls.get_maximum_num_blocks(
                config=model_config,
                tensor_parallel_size=rbln_config.tensor_parallel_size or 1,
                kvcache_block_size=rbln_config.kvcache_block_size,
                nbits_per_param=16 if not rbln_config.quantization else 4,  # TODO(jongho): FIX Ad-hoc
                n_model_params=sum(p.numel() for p in model.parameters()),
                num_runtimes=1 if not rbln_config.can_generate else 1 + len(rbln_config.decoder_batch_sizes),
            )

            max_num_blocks = min(max_num_blocks, estimated_max_num_blocks)

            flash_min_blocks = rbln_config.max_seq_len // rbln_config.kvcache_block_size + 1
            if rbln_config.batch_size > 1 and max_num_blocks < flash_min_blocks:
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

    def _preprocess_chunked_prefill(
        self,
        inputs: torch.Tensor,
        cache_position: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_embed: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ):
        """
        Prepare inputs for prefill phase.
        """
        # Handle continuous batching in a compiled graph by extracting valid inputs
        # If an attention mask is provided, select only the valid (non-masked) inputs
        if attention_mask is not None:
            inputs = inputs[:, attention_mask.bool()]
            position_embed = None if position_embed is None else position_embed[:, :, :, attention_mask.bool(), :]
            token_type_ids = None if token_type_ids is None else token_type_ids[:, attention_mask.bool()]

        query_length = inputs.shape[1]
        if query_length > self.rbln_config.max_seq_len:
            raise ValueError(
                f"Input length ({query_length}) exceeds the maximum allowed sequence length ({self.rbln_config.max_seq_len})."
            )

        # Initialize attention mask for chunked processing
        chunked_attention_mask = (
            torch.zeros(1, 1, self.rbln_config.prefill_chunk_size, self.rbln_config.max_seq_len, dtype=torch.float32)
            if self.rbln_config.use_attention_mask
            else None
        )

        # Pad input and cache_position if the last chunk is smaller than `prefill_chunk_size`
        cache_position = (
            torch.arange(query_length, dtype=torch.int32).unsqueeze(0) if cache_position is None else cache_position
        )
        padding_size = (self.rbln_config.prefill_chunk_size - query_length) % self.rbln_config.prefill_chunk_size
        if padding_size > 0:
            inputs = (
                F.pad(inputs, (0, 0, 0, padding_size))
                if self.rbln_config.use_inputs_embeds
                else F.pad(inputs, (0, padding_size))
            )
            position_embed = F.pad(position_embed, (0, 0, 0, padding_size)) if position_embed is not None else None
            token_type_ids = F.pad(token_type_ids, (0, padding_size), value=-1) if token_type_ids is not None else None
            cache_position = F.pad(cache_position, (0, padding_size))

        # Overwrite position_ids and padded_cache_lengths
        position_ids = cache_position.clone() if self.rbln_config.use_position_ids else None
        padded_cache_lengths = 0

        return (
            inputs,
            cache_position,
            chunked_attention_mask,
            position_ids,
            position_embed,
            padded_cache_lengths,
            query_length,
            token_type_ids,
        )

    def _chunked_prefill_forward(
        self,
        inputs: torch.Tensor,
        cache_position: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        batch_idx: Optional[int] = None,
        block_tables: Optional[torch.Tensor] = None,
        is_external_block_tables: Optional[bool] = None,
        position_embed: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        local_block_tables: Optional[torch.Tensor] = None,
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
            position_ids,
            position_embed,
            padded_cache_lengths,
            query_length,
            token_type_ids,
        ) = self._preprocess_chunked_prefill(
            inputs, cache_position, attention_mask, position_embed, token_type_ids=token_type_ids
        )

        # Process input in chunks of size `prefill_chunk_size`
        output_logits = []
        for step in range(0, query_length, self.rbln_config.prefill_chunk_size):
            s, e = step, step + self.rbln_config.prefill_chunk_size
            # Extract the current chunk of inputs, cache positions, position ids, and position embeddings
            input_chunk = inputs[:, s:e]
            cache_pos_chunk = cache_position[:, s:e]
            position_ids_chunk = position_ids[:, s:e] if self.rbln_config.use_position_ids else None
            position_embed_chunk = position_embed[:, :, :, s:e, :] if position_embed is not None else None

            # Update attention mask to ensure proper causal behavior
            if self.rbln_config.use_attention_mask and not self.rbln_config.use_position_ids:
                if step > 0:  # update previous chunk
                    chunked_attention_mask[
                        :, :, :, s - self.rbln_config.prefill_chunk_size : e - self.rbln_config.prefill_chunk_size
                    ] = 1
                chunked_attention_mask[:, :, :, s:e] = self.causal_mask

            # Calculate query position if needed
            if self.rbln_config.use_local_attention or self.rbln_config.logits_to_keep > 0:
                query_position = (
                    torch.tensor((query_length - 1) % self.rbln_config.prefill_chunk_size, dtype=torch.int16)
                    if e >= query_length
                    else torch.tensor(self.rbln_config.prefill_chunk_size - 1, dtype=torch.int16)
                )
            else:
                query_position = None

            # Forward pass for the current chunk
            kwargs = {}
            if hasattr(self, "out_buffers"):
                kwargs["out"] = self.out_buffers

            output_logit = self.prefill_runtime(
                input_chunk,
                cache_pos_chunk,
                block_tables,
                local_block_tables,
                position_embed_chunk,
                query_position,
                chunked_attention_mask if self.rbln_config.use_attention_mask else None,
                position_ids_chunk,
                **kwargs,
            )
            output_logits.append(output_logit)

        # Aggregate output_logits
        output_logits = torch.concat(output_logits, dim=-2)
        if self.rbln_config.logits_to_keep > 0:
            output_logits = output_logits[:, -self.rbln_config.logits_to_keep :, :]
        else:
            output_logits = output_logits[:, :query_length, :]

        return self._postprocess_chunked_prefill(
            output_logits,
            attention_mask=attention_mask,
            batch_idx=batch_idx,
            is_external_block_tables=is_external_block_tables,
            padded_cache_lengths=padded_cache_lengths,
        )

    def _postprocess_chunked_prefill(
        self, last_hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs
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
    ) -> Tuple[torch.FloatTensor]:
        inputs = inputs_embeds if inputs_embeds is not None else input_ids
        batch_size = inputs.shape[0]
        all_last_hidden_states = []
        for b_idx in range(batch_size):
            last_hidden_states = self._chunked_prefill_forward(
                inputs[b_idx : b_idx + 1],
                attention_mask=attention_mask[b_idx] if attention_mask is not None else None,
                batch_idx=b_idx,
                position_embed=position_embed[b_idx : b_idx + 1] if position_embed is not None else None,
                block_tables=self.block_tables,
                local_block_tables=self.local_block_tables,
            )
            all_last_hidden_states.append(last_hidden_states)

        last_hidden_states = torch.concat(all_last_hidden_states, dim=0)
        return BaseModelOutputWithPast(last_hidden_state=last_hidden_states)


class RBLNDecoderOnlyModelForCausalLM(RBLNDecoderOnlyGenerationMixin, RBLNDecoderOnlyModel):
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

    def __post_init__(self, **kwargs):
        if self.rbln_config.use_inputs_embeds:
            self.main_input_name = "inputs_embeds"
            artifacts = torch.load(self.model_save_dir / self.subfolder / "torch_artifacts.pth", weights_only=False)
            self.embed_tokens = self._create_embedding_layer()
            self.embed_tokens.load_state_dict(artifacts["embed_tokens"])
        else:
            self.embed_tokens = None

        self.prefill_runtime = RBLNPytorchRuntime(runtime=self.model[0])
        if self.can_generate():
            self.decoders_runtime = {}
            for i, batch_size in enumerate(self.rbln_config.decoder_batch_sizes):
                self.decoders_runtime[batch_size] = RBLNPytorchRuntime(runtime=self.model[i + 1])
            # NOTE(eunji): Use a decoder whose batch size matches the model's main batch size for compatibility.
            self.decoder_runtime = self.decoders_runtime[self.rbln_config.batch_size]

        self.setup_forward_components()

    def setup_forward_components(self):
        super().setup_forward_components()
        if self.can_generate():
            self.dec_attn_mask = torch.zeros(
                self.rbln_config.batch_size, 1, 1, self.rbln_config.max_seq_len, dtype=torch.float32
            )

        output_size = (
            [1, self.rbln_config.prefill_chunk_size, self.config.vocab_size]
            if self.rbln_config.logits_to_keep == 0
            else [1, 1, self.config.vocab_size]
        )
        # Buffer for storing prefill output logits
        self.out_buffers = [torch.empty(output_size, dtype=torch.float32, device="cpu")]

    @classmethod
    def use_query_position(cls, use_local_attention: bool, is_prefill: bool = True):
        return is_prefill

    def _postprocess_chunked_prefill(
        self,
        logits: List[torch.Tensor],
        query_length: Optional[int] = None,
        batch_idx: Optional[int] = None,
        is_external_block_tables: Optional[bool] = None,
        padded_cache_lengths: Optional[int] = None,
        **kwargs,
    ):
        # Update decoder attention mask with processed KV-cache length from prefill phase
        if self.can_generate() and not is_external_block_tables and self.rbln_config.use_attention_mask:
            self.dec_attn_mask[batch_idx].fill_(0)
            self.dec_attn_mask[batch_idx, :, :, :query_length] = 1

        return RBLNDecoderOnlyForCausalLMOutput(logits=logits, padded_cache_lengths=padded_cache_lengths)

    def inputs_embeddings_if_needed(
        self, input_ids: Optional[torch.Tensor] = None, inputs_embeds: Optional[torch.Tensor] = None
    ):
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Either `input_ids` or `inputs_embeds` must be provided.")

        if self.rbln_config.use_inputs_embeds:
            return self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        else:
            return input_ids

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
        block_tables: Optional[torch.Tensor] = None,
        local_block_tables: Optional[torch.Tensor] = None,
        position_embed: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor]:
        # Forward method for the RBLN-optimized model, designed for integration with the HuggingFace generate API.
        # For continuous batching, the prefill stage processes one batch at a time and updates the KV cache using batch_idx.
        # A for-loop ensures synchronization with the HuggingFace generate API.
        # The decoder stage operates as usual, processing inputs in batch mode.

        # for only use forward, not for generate API
        # FIXME(taehoon): remove generate_idx if you can.
        if generate_idx is None:
            generate_idx = (
                attention_mask.sum(dim=-1, keepdim=True).int()
                if attention_mask is not None
                else torch.full((input_ids.shape[0], 1), input_ids.shape[1], dtype=torch.int32)
            )
            padded_cache_lengths = torch.zeros_like(generate_idx)

        inputs = inputs_embeds if inputs_embeds is not None else input_ids
        batch_size = inputs.shape[0]


        # Prefill
        if cache_position is None or cache_position.shape[1] > 1:
            logits = []
            for b_idx in range(batch_size):
                input_ids = inputs[b_idx : b_idx + 1] if input_ids is not None else None
                inputs_embeds = inputs[b_idx : b_idx + 1] if inputs_embeds is not None else None
                attention_mask = attention_mask[b_idx] if attention_mask is not None else None
                token_type_ids = token_type_ids[b_idx : b_idx + 1] if token_type_ids is not None else None
                position_embed = position_embed[b_idx : b_idx + 1] if position_embed is not None else None
                cache_position = (
                    cache_position[b_idx : b_idx + 1, attention_mask.bool()]
                    if cache_position is not None
                    else torch.arange(0, generate_idx[b_idx].item(), dtype=torch.int32)
                )

                inputs = self.inputs_embeddings_if_needed(input_ids, inputs_embeds)
                block_tables, local_block_tables, is_external_block_tables = (
                    self.page_table_manager.get_block_tables_if_needed(
                        inputs.shape[0],
                        cache_position,
                        batch_idx=b_idx,
                        phase="prefill",
                        block_tables=block_tables,
                        local_block_tables=local_block_tables,
                    )
                )
                output = self._chunked_prefill_forward(
                    inputs,
                    cache_position,
                    attention_mask.to(torch.float32),
                    b_idx,
                    block_tables=block_tables,
                    is_external_block_tables=is_external_block_tables,
                    token_type_ids=token_type_ids,
                    local_block_tables=local_block_tables,
                    position_embed=position_embed,
                )
                padded_cache_lengths[b_idx] += output.padded_cache_lengths
                logits.append(output.logits)
            logits = torch.cat(logits, dim=0)
        # Decoder
        else:
            logits = self._decode(
                input_ids,
                inputs_embeds,
                cache_position=cache_position,
                block_tables=block_tables,
                local_block_tables=local_block_tables,
                position_ids=position_ids if self.rbln_config.use_position_ids else None,
                attention_mask=attention_mask.to(torch.float32),
            ).logits

        if not return_dict:
            return logits, generate_idx, padded_cache_lengths
        else:
            return RBLNDecoderOnlyForCausalLMOutput(
                logits=logits, generate_idx=generate_idx, padded_cache_lengths=padded_cache_lengths
            )

    def _validate_decoder_batch_size(self, inputs: torch.Tensor, **kwargs):
        batch_size = inputs.shape[0]
        if batch_size not in self.rbln_config.decoder_batch_sizes:
            raise ValueError(
                f"No decoder runtime available for batch size {batch_size}. "
                f"Available batch sizes are: {list(self.decoders.keys())}. "
                f"Please run your model with one of these batch sizes or add support for batch size {batch_size}."
            )

        for arg_name, arg_value in kwargs.items():
            if arg_value is not None and arg_value.shape[0] != batch_size:
                raise ValueError(f"{arg_name} batch size mismatch: got {arg_value.shape[0]}, expected {batch_size}.")

    def _decode(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        cache_position: torch.Tensor = None,
        block_tables: torch.Tensor = None,
        is_external_block_tables: bool = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_embed: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        local_block_tables: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:

        inputs = self.inputs_embeddings_if_needed(input_ids, inputs_embeds)
        block_tables, local_block_tables, is_external_block_tables = (
            self.page_table_manager.get_block_tables_if_needed(
                inputs.shape[0],
                cache_position,
                phase="decode",
                block_tables=block_tables,
                local_block_tables=local_block_tables,
            )
        )
        self._validate_decoder_batch_size(
            inputs,
            cache_position=cache_position,
            block_tables=block_tables,
            attention_mask=attention_mask,
            position_embed=position_embed,
            position_ids=position_ids,
        )

        batch_size = inputs.shape[0]
        if self.rbln_config.use_attention_mask and (attention_mask is None or attention_mask.dim() < 4):
            for b_idx in range(batch_size):
                decoding_step = cache_position[b_idx].item()
                if not (0 <= decoding_step < self.dec_attn_mask.shape[-1]):
                    raise ValueError(
                        f"Decoding step {decoding_step} out of bounds for attention mask with shape {self.dec_attn_mask.shape}."
                    )

                if is_external_block_tables:
                    self.dec_attn_mask[b_idx].fill_(0)
                    self.dec_attn_mask[b_idx, :, :, : decoding_step + 1] = 1
                else:
                    self.dec_attn_mask[b_idx, :, :, decoding_step] = 1

            attention_mask = self.dec_attn_mask

        logits = self.decoders_runtime[batch_size](
            inputs,
            cache_position,
            block_tables,
            local_block_tables,
            position_embed,
            attention_mask if self.rbln_config.use_attention_mask else None,
            position_ids if self.rbln_config.use_position_ids else None,
        )

        return RBLNDecoderOnlyForCausalLMOutput(logits=logits)
