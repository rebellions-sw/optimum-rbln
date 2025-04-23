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
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Deque, Dict, List, Optional, Tuple, Union

import rebel
import torch
from rebel.compile_context import CompileContext
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.modeling_utils import no_init_weights
from transformers.utils import ModelOutput

from ....configuration_utils import RBLNCompileConfig
from ....modeling import RBLNModel
from ....utils.logging import get_logger
from ....utils.runtime_utils import RBLNPytorchRuntime
from ...utils.rbln_quantization import QuantizationManager
from .configuration_decoderonly import RBLNDecoderOnlyModelForCausalLMConfig
from .decoderonly_architecture import (
    DecoderOnlyWrapper,
    set_default_values,
    validate_attention_method,
)


logger = get_logger()

if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer


class RBLNRuntimeModel(RBLNPytorchRuntime):
    mandatory_members = ["main_input_name", "embed_tokens"]

    def __init__(
        self,
        runtime: rebel.Runtime,
        phase: str,
        batch_size: int,
        dec_attn_mask: torch.Tensor,
        block_tables: torch.Tensor,
        free_block_pool: Deque,
        kvcache_block_size: int,
        use_attention_mask: bool,
        attn_impl: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(runtime, **kwargs)
        self.phase = phase
        self.batch_size = batch_size

        # shared data structures between prefill and decode phase
        self.use_attention_mask = use_attention_mask

        # shared tensor between prefill and decode phase
        self.dec_attn_mask = dec_attn_mask
        self.block_tables = block_tables
        self.free_block_pool = free_block_pool

        self.kvcache_block_size = kvcache_block_size
        self.empty_block = -1
        self.attn_impl = attn_impl

        if self.phase == "prefill":
            vocab_size = kwargs.pop("vocab_size")
            self.max_seq_len = kwargs.pop("max_seq_len")
            self.prefill_chunk_size = kwargs.pop("prefill_chunk_size")
            self.output_size = [1, 1, vocab_size]
            self.causal_mask = 1 - torch.triu(
                torch.ones(1, 1, self.prefill_chunk_size, self.prefill_chunk_size), diagonal=1
            )

    def get_block_tables(self, cache_position: torch.Tensor, batch_idx: int = None):
        """
        Manages and returns the KV cache block tables.
        Updates the block tables based on the given cache_position, allocating new blocks or reusing existing ones as needed.

        Args:
            cache_position (torch.Tensor): Tensor containing cache position information, indicating positions within the cache for each batch item.
            batch_idx (int, optional): Specific batch index, used when phase is 'prefill'.

        Returns:
            torch.Tensor: Updated block tables.
        """

        NO_BLOCKS_ERROR = (
            "No memory blocks are available for allocation. "
            "The generate() API cannot complete this inference task because Paged Attention is not fully supported by optimum-rbln. "
            "This is supported by vllm-rbln (see: https://docs.rbln.ai/software/model_serving/vllm_support/vllm-rbln.html). "
            "Using vllm-rbln should fix this issue and enhance inference performance."
        )

        def update_block(batch_idx: int, block_idx: int):
            """
            If the block is empty (empty_block), allocates a block from the free_block_pool.
            """
            if self.block_tables[batch_idx][block_idx] == self.empty_block:
                if self.free_block_pool:
                    block = self.free_block_pool.popleft()
                    self.block_tables[batch_idx][block_idx] = block
                else:
                    raise RuntimeError(NO_BLOCKS_ERROR)

        def replace_empty_block(block_tables: torch.Tensor):
            """
            Replaces all occurrences of `self.empty_block` in `block_tables` with a dummy block from `self.free_block_pool`.
            """
            if not torch.any(block_tables == self.empty_block):
                return block_tables.clone()
            elif self.free_block_pool:
                _free_block = self.free_block_pool[0]
                return torch.where(block_tables == self.empty_block, _free_block, block_tables)
            else:
                raise RuntimeError(NO_BLOCKS_ERROR)

        if self.phase == "prefill":
            # Track previously used blocks and return them to the free_block_pool and
            # reset the current batch's block table to empty blocks
            prev_blocks = self.block_tables[batch_idx][self.block_tables[batch_idx] != self.empty_block].tolist()
            self.free_block_pool.extend(prev_blocks)
            self.block_tables[batch_idx].fill_(self.empty_block)

            # Get the start (s) and end (e) positions from cache_position and
            # iterate over the cache positions to allocate necessary blocks
            s, e = cache_position[0][0].item(), cache_position[0][-1].item()
            for position in range(s, e + 1, self.kvcache_block_size):
                block_idx = position // self.kvcache_block_size
                if batch_idx >= len(self.block_tables) or block_idx >= len(self.block_tables[batch_idx]):
                    raise IndexError(f"Invalid index: batch_idx={batch_idx}, block_idx={block_idx}")
                update_block(batch_idx, block_idx)

            return replace_empty_block(self.block_tables[batch_idx])
        # Case for 'decoder' phase, iterate over the cache positions to allocate necessary blocks
        else:
            for b_idx in range(self.batch_size):
                position = cache_position[b_idx][0].item()
                block_idx = position // self.kvcache_block_size
                update_block(b_idx, block_idx)

            return replace_empty_block(self.block_tables)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        cache_position: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        batch_idx: Optional[int] = None,
        block_tables: Optional[torch.Tensor] = None,
        position_embed: Optional[torch.Tensor] = None,
    ):
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Either `input_ids` or `inputs_embeds` must be provided.")

        if inputs_embeds is None:
            inputs = input_ids
            if self.embed_tokens is not None:
                inputs = self.embed_tokens(inputs)
        else:
            inputs = inputs_embeds

        if block_tables is None:
            block_tables = self.get_block_tables(cache_position, batch_idx=batch_idx)
            is_external_block_tables = False
        else:
            is_external_block_tables = True

        if self.phase == "decode":
            return self.decode_forward(
                inputs,
                cache_position,
                block_tables,
                is_external_block_tables,
                attention_mask=attention_mask,
                position_embed=position_embed,
            )
        else:
            return self.prefill_forward(
                inputs, cache_position, attention_mask, batch_idx, block_tables, position_embed=position_embed
            )

    def decode_forward(
        self,
        inputs: torch.Tensor,
        cache_position: torch.Tensor = None,
        block_tables: torch.Tensor = None,
        is_external_block_tables: bool = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_embed: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        batch_size = inputs.shape[0]
        if batch_size != self.batch_size:
            raise RuntimeError(
                f"Batch size mismatch: got {batch_size}, expected {self.batch_size} (compiled batch size)."
            )

        if batch_size != cache_position.shape[0]:
            raise RuntimeError(f"Cache position size mismatch: got {cache_position.shape[0]}, expected {batch_size}.")

        if self.use_attention_mask and attention_mask is None:
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

        logits = super().forward(
            inputs,
            cache_position,
            attention_mask if self.use_attention_mask else None,
            block_tables,
            position_embed,
        )

        return logits

    def prefill_forward(
        self,
        inputs: torch.Tensor,
        cache_position: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        batch_idx: int = None,
        block_tables: torch.Tensor = None,
        is_external_block_tables: bool = None,
        position_embed: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        """
        Performs chunked prefill for efficient KV-cache updates and memory optimization.
        Instead of processing the entire sequence at once, the input is divided into chunks of size `prefill_chunk_size`,
        and each chunk is processed sequentially. This allows for better memory utilization and compatibility with continuous batching.
        """

        # Handle continuous batching in a compiled graph by extracting valid inputs
        # If an attention mask is provided, select only the valid (non-masked) inputs
        inputs = inputs[:, attention_mask.bool()] if attention_mask is not None else inputs
        if position_embed is not None:
            position_embed = (
                position_embed[:, :, :, attention_mask.bool(), :] if attention_mask is not None else position_embed
            )

        query_length = inputs.shape[1]
        if query_length > self.max_seq_len:
            raise ValueError(
                f"Input length ({query_length}) exceeds the maximum allowed sequence length ({self.max_seq_len})."
            )

        # Initialize attention mask for chunked processing
        if self.use_attention_mask:
            chunked_attention_mask = torch.zeros(1, 1, self.prefill_chunk_size, self.max_seq_len, dtype=torch.float32)

        # Buffer for storing output logits
        out_buffers = [
            torch.empty(
                size=self.output_size,
                dtype=torch.float32,
                device="cpu",
            )
        ]

        # Process input in chunks of size `prefill_chunk_size`
        for step in range(0, query_length, self.prefill_chunk_size):
            # Pad input and cache_position if the last chunk is smaller than `prefill_chunk_size`
            if (step + self.prefill_chunk_size) > query_length:
                padding_size = step + self.prefill_chunk_size - query_length
                # inputs_embeds
                if inputs.dim() == 3:
                    inputs = torch.nn.functional.pad(inputs, (0, 0, 0, padding_size))
                # inputs_ids
                else:
                    inputs = torch.nn.functional.pad(inputs, (0, padding_size))

                cache_position = torch.cat(
                    [
                        cache_position,
                        torch.arange(
                            query_length,
                            step + self.prefill_chunk_size,
                            dtype=torch.int32,
                        ).unsqueeze(0),
                    ],
                    dim=-1,
                )

                if position_embed is not None:
                    position_embed = torch.nn.functional.pad(position_embed, (0, 0, 0, padding_size))

            # Extract the current chunk of inputs and cache positions
            input_chunk = inputs[:, step : step + self.prefill_chunk_size]
            cache_pos_chunk = cache_position[:, step : step + self.prefill_chunk_size]
            if position_embed is not None:
                position_embed_chunk = position_embed[:, :, :, step : step + self.prefill_chunk_size, :]

            if self.use_attention_mask:
                # Update attention mask to ensure proper causal behavior
                if step >= self.prefill_chunk_size:
                    chunked_attention_mask[:, :, :, step - self.prefill_chunk_size : step] = 1
                chunked_attention_mask[:, :, :, step : step + self.prefill_chunk_size] = self.causal_mask

            # Define query position
            query_position = torch.tensor((query_length - 1) % self.prefill_chunk_size, dtype=torch.int16)

            # Forward pass for the current chunk
            logits = super().forward(
                input_chunk,
                cache_pos_chunk,
                chunked_attention_mask if self.use_attention_mask else None,
                query_position,
                block_tables,
                position_embed_chunk if position_embed is not None else None,
                out=out_buffers,
            )

        # Update decoder attention mask with processed KV-cache length from prefill phase
        if not is_external_block_tables and self.use_attention_mask:
            self.dec_attn_mask[batch_idx].fill_(0)
            self.dec_attn_mask[batch_idx, :, :, :query_length] = 1

        return logits


@dataclass
class RBLNDecoderOnlyOutput(ModelOutput):
    logits: torch.FloatTensor = None
    generate_idx: torch.Tensor = None


class RBLNDecoderOnlyModelForCausalLM(RBLNModel):
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

    main_input_name = "input_ids"
    auto_model_class = AutoModelForCausalLM
    _decoder_wrapper_cls = DecoderOnlyWrapper
    _use_rotary_emb = True

    def __post_init__(self, **kwargs):
        main_input_name = self.main_input_name

        if self.rbln_config.use_inputs_embeds:
            main_input_name = "inputs_embeds"
            artifacts = torch.load(self.model_save_dir / self.subfolder / "torch_artifacts.pth", weights_only=False)
            with no_init_weights():
                self.embed_tokens = torch.nn.Embedding(
                    self.config.vocab_size,
                    self.config.hidden_size,
                    self.config.pad_token_id,
                )
            self.embed_tokens.load_state_dict(artifacts["embed_tokens"])
        else:
            self.embed_tokens = None

        # Initialize shared resources to be used across Runtime instances (prefill and decode phases)
        dec_attn_mask = torch.zeros(
            self.rbln_config.batch_size, 1, 1, self.rbln_config.max_seq_len, dtype=torch.float32
        )
        block_tables = torch.zeros(
            self.rbln_config.batch_size,
            self.rbln_config.max_seq_len // self.rbln_config.kvcache_block_size,
            dtype=torch.int16,
        ).fill_(-1)
        free_block_pool = deque(x for x in range(self.rbln_config.kvcache_num_blocks))

        self.prefill_decoder = RBLNRuntimeModel(
            runtime=self.model[0],
            main_input_name=main_input_name,
            embed_tokens=self.embed_tokens,
            phase="prefill",
            batch_size=self.rbln_config.batch_size,
            dec_attn_mask=dec_attn_mask,
            block_tables=block_tables,
            free_block_pool=free_block_pool,
            kvcache_block_size=self.rbln_config.kvcache_block_size,
            vocab_size=self.config.vocab_size,
            prefill_chunk_size=self.rbln_config.prefill_chunk_size,
            max_seq_len=self.rbln_config.max_seq_len,
            use_attention_mask=self.rbln_config.use_attention_mask,
            attn_impl=self.rbln_config.attn_impl,
        )
        self.decoder = RBLNRuntimeModel(
            runtime=self.model[1],
            main_input_name=main_input_name,
            embed_tokens=self.embed_tokens,
            phase="decode",
            batch_size=self.rbln_config.batch_size,
            dec_attn_mask=dec_attn_mask,
            block_tables=block_tables,
            free_block_pool=free_block_pool,
            kvcache_block_size=self.rbln_config.kvcache_block_size,
            use_attention_mask=self.rbln_config.use_attention_mask,
            attn_impl=self.rbln_config.attn_impl,
        )

    @classmethod
    def save_torch_artifacts(
        cls,
        model: "PreTrainedModel",
        save_dir_path: Path,
        subfolder: str,
        rbln_config: RBLNDecoderOnlyModelForCausalLMConfig,
    ):
        """
        If you are unavoidably running on a CPU rather than an RBLN device,
        store the torch tensor, weight, etc. in this function.
        """
        if rbln_config.use_inputs_embeds:
            save_dict = {}
            save_dict["embed_tokens"] = model.get_input_embeddings().state_dict()
            torch.save(save_dict, save_dir_path / subfolder / "torch_artifacts.pth")

    def get_input_embeddings(self):
        return self.embed_tokens

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

    def __getattr__(self, __name: str) -> Any:
        """
        Special method to delegate attribute access to the original Huggingface LM class.
        This method is called when an attribute is not found in the current instance's dictionary.
        It enables transparent access to the original model's attributes and methods while maintaining
        proper method binding.

        The method implements a delegation pattern that:
        1. For methods: Creates a wrapper that properly binds 'self' to method calls
        2. For other attributes: Returns them directly from the original class
        """

        def redirect(func):
            return lambda *pargs, **kwargs: func(self, *pargs, **kwargs)

        val = getattr(self.get_hf_class(), __name, None) or getattr(PreTrainedModel, __name)
        if isinstance(val, Callable) and "self" in set(inspect.signature(val).parameters):
            return redirect(val)
        return val

    @classmethod
    def get_pytorch_model(
        cls, *args, rbln_config: Optional[RBLNDecoderOnlyModelForCausalLMConfig] = None, **kwargs
    ) -> "PreTrainedModel":
        if (
            rbln_config is not None
            and "format" in rbln_config.quantization
            and rbln_config.quantization["format"] == "rbln"
        ):
            model = cls.get_quantized_model(*args, **kwargs)
        else:
            model = super().get_pytorch_model(*args, **kwargs)

        return model

    @classmethod
    def wrap_model_if_needed(cls, model: "PreTrainedModel", rbln_config: "RBLNDecoderOnlyModelForCausalLMConfig"):
        wrapper_cfg = {
            "max_seq_len": rbln_config.max_seq_len,
            "attn_impl": rbln_config.attn_impl,
            "kvcache_partition_len": rbln_config.kvcache_partition_len,
            "kvcache_block_size": rbln_config.kvcache_block_size,
            "use_rotary_emb": cls._use_rotary_emb,
            "use_attention_mask": rbln_config.use_attention_mask,
        }
        return cls._decoder_wrapper_cls(model, **wrapper_cfg).eval()

    @classmethod
    @torch.inference_mode()
    def get_compiled_model(cls, model: "PreTrainedModel", rbln_config: RBLNDecoderOnlyModelForCausalLMConfig):
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

        @QuantizationManager.with_quantization_env
        def compile_model(*args, **kwargs):
            try:
                original_linear = torch.nn.functional.linear
                torch.nn.functional.linear = torch.ops.rbln_custom_ops.linear
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
            finally:
                torch.nn.functional.linear = original_linear

        compiled_models = compile_model(quantize_config=rbln_config.quantization)

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
    def maybe_suggest_kvcache_num_blocks(
        cls,
        compiled_models: Dict[str, rebel.RBLNCompiledModel],
        model_config: PretrainedConfig,
        rbln_config: RBLNDecoderOnlyModelForCausalLMConfig,
    ) -> None:
        # Get the actual memory allocation of each node by key
        alloc_memory_per_node_by_key: Dict[str, List[int]] = compiled_models["prefill"].get_alloc_per_node_by_key()
        alloc_memory_by_key: Dict[str, int] = {
            key: sum(memory_per_node) for key, memory_per_node in alloc_memory_per_node_by_key.items()
        }
        for key, memory_per_node in compiled_models["decoder"].get_alloc_per_node_by_key().items():
            alloc_memory_by_key[key] += sum(memory_per_node)
        alloc_memory_by_key.pop("PortRecur")  # kv-cache
        kernel_size = alloc_memory_by_key.pop("Kernel")  # model weight

        # Get the maximum number of blocks that can be allocated
        buffer = sum(alloc_memory_by_key.values())
        max_num_blocks = cls.get_maximum_num_blocks(
            config=model_config,
            tensor_parallel_size=rbln_config.tensor_parallel_size,
            kvcache_block_size=rbln_config.kvcache_block_size,
            kernel_size=kernel_size,
            buffer=buffer,
        )

        # Since our estimation logic is not always accurate,
        # users can set `kvcache_num_blocks` to `max_num_blocks`.
        # If the memory is not enough, the model will fail to compile.
        if rbln_config.kvcache_num_blocks < max_num_blocks:
            logger.warning(
                f"Current `kvcache_num_blocks` setting is {rbln_config.kvcache_num_blocks}. "
                "Our analysis indicates that additional memory is available for more blocks. "
                f"Consider increasing `kvcache_num_blocks` to {max_num_blocks} for potentially improved performance. "
                "Please be advised that our memory estimation algorithm has limitations, "
                "and increasing this value may not guarantee successful model compilation."
            )

    @classmethod
    def get_maximum_num_blocks(
        cls,
        config: PretrainedConfig,
        tensor_parallel_size: int,
        kvcache_block_size: int,
        nbits_per_param: Optional[int] = None,
        n_model_params: Optional[int] = None,
        kernel_size: Optional[int] = None,
        buffer: Optional[int] = None,
    ) -> int:
        """
        We are finding max_n_blocks(x) that satisfies the following equation:

        available_dram - kernel_size - buffer
            - num_layers * 2 * tensor_parallel_size
            * align_2MB(
                x
                * block_size
                * align_64(head_dim)
                * math.ceil(num_key_value_heads / tensor_parallel_size)
                * 2
            ) > 0

        This inequality can be rewritten as follows:

        a - c * align_2MB(b * x) > 0
        where
           a = available_dram - kernel_size - buffer
           b = block_size * align_64(head_dim) * math.ceil(num_key_value_heads / tensor_parallel_size) * 2
           c = num_layers * 2 * tensor_parallel_size

        We can rewrite the inequality as follows:
        k > align_2MB(b*x)
        where
           k = a / c

        After that, we can derive the following equation:
        x = floor(2**21 / b * floor((k - 1) / 2**21))
        """

        def align(x: int, nbytes: int) -> int:
            return int(math.ceil(x / nbytes) * nbytes)

        def align_2MB(x: int) -> int:
            return align(x, 2**21)

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

        if kernel_size is None:
            if n_model_params is None:
                raise ValueError("`n_model_params` should be specified to estimate the kernel memory.")
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
        elif n_model_params is not None:
            raise ValueError("Both `n_model_params` and `kernel_size` cannot be specified.")

        available_dram -= kernel_size

        if buffer is None:
            # TODO: Accurate buffer estimation
            buffer_per_core = 2**29  # 500MB per npu
            buffer = buffer_per_core * tensor_parallel_size
        available_dram -= buffer

        b = kvcache_block_size * align(head_dim, 64) * math.ceil(num_key_value_heads / tensor_parallel_size) * 2
        c = num_layers * 2 * tensor_parallel_size
        k = available_dram / c
        max_n_blocks = math.floor(2**21 / b * math.floor((k - 1) / 2**21))

        return max_n_blocks

    @classmethod
    def get_input_info(
        cls,
        batch_size: int,
        query_length: int,
        use_inputs_embeds: bool,
        use_attention_mask: bool,
        max_seq_len: int,
        kvcache_block_size: int,
        kvcache_num_blocks: int,
        num_key_value_heads: int,
        num_hidden_layers: int,
        hidden_size: int,
        head_dim: int,
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

        if use_attention_mask:
            input_info.extend(
                [
                    ("attention_mask", [batch_size, 1, query_length, max_seq_len], "float32"),
                ]
            )

        if query_length > 1:
            input_info.extend(
                [
                    ("query_position", [], "int16"),
                ]
            )

        max_block_cnt = max_seq_len // kvcache_block_size

        if query_length > 1:
            input_info.extend([("block_tables", [max_block_cnt], "int16")])
        else:
            input_info.extend([("block_tables", [batch_size, max_block_cnt], "int16")])

        input_info.extend(
            [
                (
                    f"past_key_values_{i}",
                    [
                        kvcache_num_blocks,
                        num_key_value_heads,
                        kvcache_block_size,
                        head_dim,
                    ],
                    "float32",
                )
                for i in range(num_hidden_layers * 2)
            ]
        )

        return input_info

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Optional[Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"]] = None,
        model: Optional["PreTrainedModel"] = None,
        model_config: Optional["PretrainedConfig"] = None,
        rbln_config: Optional[RBLNDecoderOnlyModelForCausalLMConfig] = None,
    ) -> RBLNDecoderOnlyModelForCausalLMConfig:
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

        if rbln_config.attn_impl == "flash_attn":
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
        )

        prefill_compile_config = RBLNCompileConfig(compiled_model_name="prefill", input_info=prefill_input_info)
        dec_compile_config = RBLNCompileConfig(compiled_model_name="decoder", input_info=dec_input_info)

        rbln_config.set_compile_cfgs([prefill_compile_config, dec_compile_config])

        return rbln_config

    @classmethod
    def _create_runtimes(
        cls,
        compiled_models: List[rebel.RBLNCompiledModel],
        rbln_config: RBLNDecoderOnlyModelForCausalLMConfig,
    ) -> List[rebel.Runtime]:
        if any(model_name not in rbln_config.device_map for model_name in ["prefill", "decoder"]):
            cls._raise_missing_compiled_file_error(["prefill", "decoder"])

        return [
            rebel.Runtime(
                compiled_models[0],
                tensor_type="pt",
                device=rbln_config.device_map["prefill"],
                activate_profiler=rbln_config.activate_profiler,
            ),
            rebel.Runtime(
                compiled_models[1],
                tensor_type="pt",
                device=rbln_config.device_map["decoder"],
                activate_profiler=rbln_config.activate_profiler,
            ),
        ]

    def get_decoder(self):
        return self.decoder

    def can_generate(self):
        return True

    def _reorder_cache(self, past_key_values, beam_idx):
        raise NotImplementedError

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        generate_idx: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        model_inputs = {}
        is_prefill_phase = generate_idx is None

        if is_prefill_phase:
            generate_idx = attention_mask.sum(dim=-1, keepdim=True).int()
            cache_position = None
        else:
            if inputs_embeds is not None:
                raise NotImplementedError("Specifying inputs_embeds in decoder phase is not supported.")

            input_ids = input_ids[:, -1:]
            cache_position = generate_idx
            generate_idx = generate_idx + 1
            model_inputs.update({"input_ids": input_ids})

        if inputs_embeds is not None:
            if self.rbln_config.use_inputs_embeds:
                model_inputs.update({"inputs_embeds": inputs_embeds})
            else:
                raise ValueError(
                    "The specifying inputs_embeds is only supported when using a compiled RBLN model with 'rbln_use_inputs_embeds' set to True."
                )
        else:
            model_inputs.update({"input_ids": input_ids})

        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "generate_idx": generate_idx,
            }
        )

        return model_inputs

    def _update_model_kwargs_for_generation(
        self,
        outputs: RBLNDecoderOnlyOutput,
        model_kwargs: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, Any]:
        # update generate_idx
        model_kwargs["generate_idx"] = outputs.generate_idx

        return model_kwargs

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        generate_idx: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor]:
        """
        Forward method for the RBLN-optimized model, designed for integration with the HuggingFace generate API.
        For continuous batching, the prefill stage processes one batch at a time and updates the KV cache using batch_idx.
        A for-loop ensures synchronization with the HuggingFace generate API.
        The decoder stage operates as usual, processing inputs in batch mode.
        """
        # Prefll
        if cache_position is None:
            logits = []
            inputs = inputs_embeds if inputs_embeds is not None else input_ids
            batch_size = inputs.shape[0]

            for b_idx in range(batch_size):
                cache_position = torch.arange(0, generate_idx[b_idx].item(), dtype=torch.int32).unsqueeze(0)
                logit = self.prefill_decoder(
                    input_ids=inputs[b_idx : b_idx + 1] if inputs_embeds is None else None,
                    inputs_embeds=inputs[b_idx : b_idx + 1] if inputs_embeds is not None else None,
                    attention_mask=attention_mask[b_idx] if attention_mask is not None else None,
                    cache_position=cache_position,
                    batch_idx=b_idx,
                )
                logits.append(logit)

            logits = torch.cat(logits, dim=0)
        # Decoder
        else:
            logits = self.decoder(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
            )

        return RBLNDecoderOnlyOutput(
            logits=logits,
            generate_idx=generate_idx,
        )
