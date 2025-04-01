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

from ....modeling import RBLNModel
from ....modeling_config import RBLNCompileConfig, RBLNConfig
from ....utils.logging import get_logger
from ....utils.runtime_utils import RBLNPytorchRuntime
from ...utils.rbln_quantization import QuantizationManager
from .decoderonly_architecture import (
    DecoderOnlyWrapper,
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
            )
        else:
            return self.prefill_forward(inputs, cache_position, attention_mask, batch_idx, block_tables)

    def decode_forward(
        self,
        inputs: torch.Tensor,
        cache_position: torch.Tensor = None,
        block_tables: torch.Tensor = None,
        is_external_block_tables: bool = None,
        attention_mask: Optional[torch.Tensor] = None,
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

            attention_mask = self.dec_attn_mask

        logits = super().forward(
            inputs,
            cache_position,
            attention_mask if self.use_attention_mask else None,
            block_tables,
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
    ) -> torch.FloatTensor:
        """
        Performs chunked prefill for efficient KV-cache updates and memory optimization.
        Instead of processing the entire sequence at once, the input is divided into chunks of size `prefill_chunk_size`,
        and each chunk is processed sequentially. This allows for better memory utilization and compatibility with continuous batching.
        """

        # Handle continuous batching in a compiled graph by extracting valid inputs
        # If an attention mask is provided, select only the valid (non-masked) inputs
        inputs = inputs[:, attention_mask.bool()] if attention_mask is not None else inputs

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

            # Extract the current chunk of inputs and cache positions
            input_chunk = inputs[:, step : step + self.prefill_chunk_size]
            cache_pos_chunk = cache_position[:, step : step + self.prefill_chunk_size]

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
        self.batch_size = self.rbln_config.model_cfg["batch_size"]
        self.max_seq_len = self.rbln_config.model_cfg["max_seq_len"]
        self.prefill_chunk_size = self.rbln_config.model_cfg["prefill_chunk_size"]
        self.kvcache_block_size = self.rbln_config.model_cfg["kvcache_block_size"]
        # FIXME get kvcache_num_blocks from compiled results.
        self.kvcache_num_blocks = self.rbln_config.model_cfg["kvcache_num_blocks"]
        self.use_attention_mask = self.rbln_config.model_cfg["use_attention_mask"]
        attn_impl = self.rbln_config.model_cfg["attn_impl"]
        main_input_name = self.main_input_name

        if self.rbln_config.model_cfg["use_inputs_embeds"]:
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
        dec_attn_mask = torch.zeros(self.batch_size, 1, 1, self.max_seq_len, dtype=torch.float32)
        block_tables = torch.zeros(
            self.batch_size, self.max_seq_len // self.kvcache_block_size, dtype=torch.int16
        ).fill_(-1)
        free_block_pool = deque(x for x in range(self.kvcache_num_blocks))

        self.prefill_decoder = RBLNRuntimeModel(
            runtime=self.model[0],
            main_input_name=main_input_name,
            embed_tokens=self.embed_tokens,
            phase="prefill",
            batch_size=self.batch_size,
            dec_attn_mask=dec_attn_mask,
            block_tables=block_tables,
            free_block_pool=free_block_pool,
            kvcache_block_size=self.kvcache_block_size,
            vocab_size=self.config.vocab_size,
            prefill_chunk_size=self.prefill_chunk_size,
            max_seq_len=self.max_seq_len,
            use_attention_mask=self.use_attention_mask,
            attn_impl=attn_impl,
        )
        self.decoder = RBLNRuntimeModel(
            runtime=self.model[1],
            main_input_name=main_input_name,
            embed_tokens=self.embed_tokens,
            phase="decode",
            batch_size=self.batch_size,
            dec_attn_mask=dec_attn_mask,
            block_tables=block_tables,
            free_block_pool=free_block_pool,
            kvcache_block_size=self.kvcache_block_size,
            use_attention_mask=self.use_attention_mask,
            attn_impl=attn_impl,
        )

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

    @classmethod
    def _create_runtimes(
        cls,
        compiled_models: List[rebel.RBLNCompiledModel],
        rbln_device_map: Dict[str, int],
        activate_profiler: Optional[bool] = None,
    ) -> List[rebel.Runtime]:
        if any(model_name not in rbln_device_map for model_name in ["prefill", "decoder"]):
            cls._raise_missing_compiled_file_error(["prefill", "decoder"])

        return [
            compiled_models[0].create_runtime(
                tensor_type="pt", device=rbln_device_map["prefill"], activate_profiler=activate_profiler
            ),
            compiled_models[1].create_runtime(
                tensor_type="pt", device=rbln_device_map["decoder"], activate_profiler=activate_profiler
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
            if self.rbln_config.model_cfg["use_inputs_embeds"]:
                model_inputs.update({"inputs_embeds": inputs_embeds})
            else:
                raise ValueError(
                    "The specifying inputs_embedst is only supported when using a compiled RBLN model with 'rbln_use_inputs_embeds' set to True."
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
