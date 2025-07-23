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
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Deque, Dict, List, Optional, Tuple, Union

import rebel
import torch
from rebel.compile_context import CompileContext
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import no_init_weights
from transformers.utils import ModelOutput

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
from ...utils.rbln_quantization import prepare_model_for_quantization
from .configuration_decoderonly import RBLNDecoderOnlyModelConfig, RBLNDecoderOnlyModelForCausalLMConfig
from .decoderonly_architecture import DecoderOnlyWrapper


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
        rbln_config: RBLNDecoderOnlyModelForCausalLMConfig,
        **kwargs: Any,
    ) -> None:
        super().__init__(runtime, **kwargs)
        self.phase = phase
        self.batch_size = batch_size
        self.rbln_config = rbln_config

        # shared tensor between prefill and decode phase
        self.dec_attn_mask = dec_attn_mask
        self.block_tables = block_tables
        self.free_block_pool = free_block_pool

        self.empty_block = -1
        if self.phase == "prefill":
            vocab_size = kwargs.pop("vocab_size")
            self.output_size = [1, 1, vocab_size]
            self.causal_mask = 1 - torch.triu(
                torch.ones(1, 1, self.rbln_config.prefill_chunk_size, self.rbln_config.prefill_chunk_size), diagonal=1
            )

    def get_block_tables(self, cache_position: torch.Tensor, batch_idx: int = None) -> torch.Tensor:
        """
        Manages and returns the KV cache block tables.
        Updates the block tables based on the given cache_position, allocating new blocks or reusing existing ones as needed.

        Args:
            cache_position (torch.Tensor): Tensor containing cache position information, indicating positions within the cache for each batch item.
            batch_idx (int, optional): Specific batch index, used when phase is 'prefill'.

        Returns:
            Updated block tables.
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

        def get_global_block_tables(batch_idx: int):
            if self.rbln_config.cache_impl == "sliding_window":
                return None

            if self.phase == "prefill":
                # Track previously used blocks and return them to the free_block_pool and
                # reset the current batch's block table to empty blocks
                prev_blocks = self.block_tables[batch_idx][self.block_tables[batch_idx] != self.empty_block].tolist()
                self.free_block_pool.extend(prev_blocks)
                self.block_tables[batch_idx].fill_(self.empty_block)

                # Get the start (s) and end (e) positions from cache_position and
                # iterate over the cache positions to allocate necessary blocks
                s, e = cache_position[0][0].item(), cache_position[0][-1].item()
                for position in range(s, e + 1, self.rbln_config.kvcache_block_size):
                    block_idx = position // self.rbln_config.kvcache_block_size
                    if batch_idx >= len(self.block_tables) or block_idx >= len(self.block_tables[batch_idx]):
                        raise IndexError(f"Invalid index: batch_idx={batch_idx}, block_idx={block_idx}")
                    update_block(batch_idx, block_idx)

                return replace_empty_block(self.block_tables[batch_idx])
            # Case for 'decoder' phase, iterate over the cache positions to allocate necessary blocks
            else:
                for b_idx in range(self.batch_size):
                    position = cache_position[b_idx][0].item()
                    block_idx = position // self.rbln_config.kvcache_block_size
                    update_block(b_idx, block_idx)

                return replace_empty_block(self.block_tables)

        def get_local_block_tables(batch_idx: int):
            if self.rbln_config.cache_impl == "static":
                return None
            else:
                return (
                    torch.tensor([batch_idx], dtype=torch.int16)
                    if self.phase == "prefill"
                    else torch.arange(self.batch_size, dtype=torch.int16).view(self.batch_size, -1)
                )

        return get_global_block_tables(batch_idx), get_local_block_tables(batch_idx)

    def is_external_block_tables(
        self, block_tables: Optional[torch.Tensor], local_block_tables: Optional[torch.Tensor]
    ):
        if self.rbln_config.cache_impl == "static" and block_tables is None:
            return False
        elif self.rbln_config.cache_impl == "sliding_window" and local_block_tables is None:
            return False
        elif self.rbln_config.cache_impl == "hybrid":
            if (block_tables is not None) != (local_block_tables is not None):
                raise ValueError(
                    "Both block_tables and local_block_tables must be provided or neither of them must be provided."
                )
            elif block_tables is None and local_block_tables is None:
                return False

        return True

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        cache_position: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        batch_idx: Optional[int] = None,
        block_tables: Optional[torch.Tensor] = None,
        position_embed: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        local_block_tables: Optional[torch.Tensor] = None,
    ):
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Either `input_ids` or `inputs_embeds` must be provided.")

        if inputs_embeds is None:
            inputs = input_ids
            if self.embed_tokens is not None:
                inputs = self.embed_tokens(inputs)
        else:
            inputs = inputs_embeds

        is_external_block_tables = self.is_external_block_tables(block_tables, local_block_tables)
        if not is_external_block_tables:
            block_tables, local_block_tables = self.get_block_tables(cache_position, batch_idx=batch_idx)

        if self.phase == "decode":
            return self.decode_forward(
                inputs,
                cache_position,
                block_tables,
                is_external_block_tables,
                attention_mask=attention_mask,
                position_embed=position_embed,
                position_ids=position_ids,
                local_block_tables=local_block_tables,
            )
        else:
            return self.prefill_forward(
                inputs,
                cache_position,
                attention_mask,
                batch_idx,
                block_tables,
                is_external_block_tables=is_external_block_tables,
                position_embed=position_embed,
                token_type_ids=token_type_ids,
                local_block_tables=local_block_tables,
            )

    def decode_forward(
        self,
        inputs: torch.Tensor,
        cache_position: torch.Tensor = None,
        block_tables: torch.Tensor = None,
        is_external_block_tables: bool = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_embed: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        local_block_tables: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        batch_size = inputs.shape[0]
        if batch_size != self.batch_size:
            raise RuntimeError(
                f"Batch size mismatch: got {batch_size}, expected {self.batch_size} (compiled batch size)."
            )

        if batch_size != cache_position.shape[0]:
            raise RuntimeError(f"Cache position size mismatch: got {cache_position.shape[0]}, expected {batch_size}.")

        if self.rbln_config.use_attention_mask and attention_mask is None:
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

        if self.rbln_config.use_global_attention and self.batch_size < block_tables.shape[0]:
            block_tables = block_tables[: self.batch_size]

        if attention_mask is not None and self.batch_size < attention_mask.shape[0]:
            attention_mask = attention_mask[: self.batch_size]

        logits = super().forward(
            inputs,
            cache_position,
            block_tables,
            local_block_tables,
            position_embed,
            attention_mask if self.rbln_config.use_attention_mask else None,
            position_ids if self.rbln_config.use_position_ids else None,
        )

        return RBLNDecoderOnlyForCausalLMOutput(logits=logits)

    def _prepare_prefill_inputs(
        self,
        inputs: torch.Tensor,
        cache_position: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embed: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ):
        """
        Prepare inputs for prefill phase.
        """
        # Handle continuous batching in a compiled graph by extracting valid inputs
        # If an attention mask is provided, select only the valid (non-masked) inputs
        inputs = inputs[:, attention_mask.bool()] if attention_mask is not None else inputs
        if position_embed is not None:
            position_embed = (
                position_embed[:, :, :, attention_mask.bool(), :] if attention_mask is not None else position_embed
            )
        if token_type_ids is not None:
            token_type_ids = token_type_ids[:, attention_mask.bool()] if attention_mask is not None else token_type_ids

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

        # Buffer for storing output logits
        out_buffers = [
            torch.empty(
                size=self.output_size,
                dtype=torch.float32,
                device="cpu",
            )
        ]

        # Pad input and cache_position if the last chunk is smaller than `prefill_chunk_size`
        padding_size = 0
        if query_length % self.rbln_config.prefill_chunk_size != 0:
            padding_size = (self.rbln_config.prefill_chunk_size - query_length) % self.rbln_config.prefill_chunk_size
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
                        query_length + padding_size,
                        dtype=torch.int32,
                    ).unsqueeze(0),
                ],
                dim=-1,
            )

            if position_embed is not None:
                position_embed = torch.nn.functional.pad(position_embed, (0, 0, 0, padding_size))

            if token_type_ids is not None:
                token_type_ids = torch.nn.functional.pad(token_type_ids, (0, padding_size), value=-1)

        # Overwrite position_ids and padded_cache_lengths
        position_ids = cache_position.clone()
        padded_cache_lengths = 0

        return (
            inputs,
            cache_position,
            chunked_attention_mask,
            out_buffers,
            position_ids,
            position_embed,
            padded_cache_lengths,
            query_length,
            token_type_ids,
        )

    def prefill_forward(
        self,
        inputs: torch.Tensor,
        cache_position: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        batch_idx: int = None,
        block_tables: torch.Tensor = None,
        is_external_block_tables: bool = False,
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
            out_buffers,
            position_ids,
            position_embed,
            padded_cache_lengths,
            query_length,
            token_type_ids,
        ) = self._prepare_prefill_inputs(
            inputs, cache_position, attention_mask, position_embed, token_type_ids=token_type_ids
        )

        # Process input in chunks of size `prefill_chunk_size`
        for step in range(0, query_length, self.rbln_config.prefill_chunk_size):
            # Extract the current chunk of inputs and cache positions
            input_chunk = inputs[:, step : step + self.rbln_config.prefill_chunk_size]
            cache_pos_chunk = cache_position[:, step : step + self.rbln_config.prefill_chunk_size]
            position_ids_chunk = (
                position_ids[:, step : step + self.rbln_config.prefill_chunk_size]
                if position_ids is not None
                else None
            )
            if position_embed is not None:
                position_embed_chunk = position_embed[:, :, :, step : step + self.rbln_config.prefill_chunk_size, :]

            if self.rbln_config.use_attention_mask and not self.rbln_config.use_position_ids:
                # Update attention mask to ensure proper causal behavior
                if step >= self.rbln_config.prefill_chunk_size:
                    chunked_attention_mask[:, :, :, step - self.rbln_config.prefill_chunk_size : step] = 1
                chunked_attention_mask[:, :, :, step : step + self.rbln_config.prefill_chunk_size] = self.causal_mask

            # Define query position
            if step + self.rbln_config.prefill_chunk_size >= query_length:
                query_position = torch.tensor(
                    (query_length - 1) % self.rbln_config.prefill_chunk_size, dtype=torch.int16
                )
            else:
                query_position = torch.tensor(self.rbln_config.prefill_chunk_size - 1, dtype=torch.int16)

            # Forward pass for the current chunk
            logits = super().forward(
                input_chunk,
                cache_pos_chunk,
                block_tables,
                local_block_tables,
                position_embed_chunk if position_embed is not None else None,
                query_position,
                chunked_attention_mask if self.rbln_config.use_attention_mask else None,
                position_ids_chunk if self.rbln_config.use_position_ids else None,
                out=out_buffers,
            )

        # Update decoder attention mask with processed KV-cache length from prefill phase
        if not is_external_block_tables and self.rbln_config.use_attention_mask:
            self.dec_attn_mask[batch_idx].fill_(0)
            self.dec_attn_mask[batch_idx, :, :, :query_length] = 1

        return RBLNDecoderOnlyForCausalLMOutput(logits=logits, padded_cache_lengths=padded_cache_lengths)


@dataclass
class RBLNDecoderOnlyForCausalLMOutput(ModelOutput):
    logits: torch.FloatTensor = None
    generate_idx: torch.Tensor = None
    padded_cache_lengths: int = None


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

        # TODO: add prefill runtime class.
        self.prefill_decoder = RBLNPytorchRuntime(runtime=self.model[0])

        # attributes for prefill
        if self.rbln_config.use_global_attention:
            self.block_tables = torch.arange(self.rbln_config.kvcache_num_blocks, dtype=torch.int16)
        if self.rbln_config.use_local_attention:
            self.local_block_tables = torch.tensor([0], dtype=torch.int16)
        if self.rbln_config.use_attention_mask:
            self.causal_mask = 1 - torch.triu(
                torch.ones(1, 1, self.rbln_config.prefill_chunk_size, self.rbln_config.prefill_chunk_size), diagonal=1
            )

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

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_attn_impl(self) -> str:
        return self.rbln_config.attn_impl

    def get_kvcache_num_blocks(self) -> int:
        return self.rbln_config.kvcache_num_blocks

    @classmethod
    def wrap_model_if_needed(cls, model: PreTrainedModel, rbln_config: "RBLNDecoderOnlyModelConfig"):
        wrapper_cfg = {
            "max_seq_len": rbln_config.max_seq_len,
            "attn_impl": rbln_config.attn_impl,
            "kvcache_partition_len": rbln_config.kvcache_partition_len,
            "kvcache_block_size": rbln_config.kvcache_block_size,
            "use_rotary_emb": cls._use_rotary_emb,
            "use_attention_mask": rbln_config.use_attention_mask,
            "use_position_ids": rbln_config.use_position_ids,
            "use_inputs_embeds": rbln_config.use_inputs_embeds,
            "cache_impl": rbln_config.cache_impl,
            "sliding_window": rbln_config.sliding_window,
            "sliding_window_layers": rbln_config.sliding_window_layers,
        }
        return cls._decoder_wrapper_cls(model, **wrapper_cfg).eval()

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
    def get_compiled_model(
        cls,
        model: PreTrainedModel,
        rbln_config: RBLNDecoderOnlyModelConfig,
    ):
        wrapped_model = cls.wrap_model_if_needed(model, rbln_config)
        compile_config = rbln_config.compile_cfgs[0]

        # Here we use meta tensor, for the memory efficiency.
        meta_tensor_names = [name for name, _, _ in compile_config.input_info if "past_key_values" in name]
        example_inputs = compile_config.get_dummy_inputs(fill=0, meta_tensor_names=meta_tensor_names)
        context, _ = cls._get_compile_context(compile_config, example_inputs)

        compiled_model = cls._compile_model(
            wrapped_model, compile_config, example_inputs, context, rbln_config, rbln_config.quantization, "prefill"
        )
        compiled_models = {"prefill": compiled_model}

        return compiled_models

    @classmethod
    def get_quantized_model(
        cls, *args, rbln_config: Optional[RBLNDecoderOnlyModelConfig] = None, **kwargs
    ) -> PreTrainedModel:
        raise NotImplementedError

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

        # 1. main input
        if rbln_config.use_inputs_embeds:
            main_input = ("inputs_embeds", [batch_size, query_length, hidden_size], "float32")
        else:
            main_input = ("input_ids", [batch_size, query_length], "int64")

        # 2. cache_position
        input_info = [
            main_input,
            (
                "cache_position",
                [batch_size, query_length],
                "int32",
            ),
        ]

        # 3. block_tables
        if rbln_config.use_global_attention:
            max_block_cnt = rbln_config.max_seq_len // rbln_config.kvcache_block_size
            input_info.extend(
                [("block_tables", [max_block_cnt] if is_prefill else [batch_size, max_block_cnt], "int16")]
            )
        if rbln_config.use_local_attention:
            input_info.extend([("local_block_tables", [1] if is_prefill else [batch_size, 1], "int16")])

        # 4. query_position for sliding window attention
        if cls.use_query_position(rbln_config.use_local_attention, is_prefill):
            input_info.extend([("query_position", [], "int16")])

        # 5. attention_mask & position_ids
        if rbln_config.use_attention_mask:
            input_info.extend(
                [
                    ("attention_mask", [batch_size, rbln_config.max_seq_len], "float32")
                    if rbln_config.use_position_ids
                    else ("attention_mask", [batch_size, 1, query_length, rbln_config.max_seq_len], "float32")
                ]
            )
        if rbln_config.use_position_ids:
            input_info.append(("position_ids", [batch_size, query_length], "int32"))

        # 6. past_key_values
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

        if rbln_config.kvcache_num_blocks is None:
            rbln_config.kvcache_num_blocks = (
                rbln_config.max_seq_len // rbln_config.kvcache_block_size
            ) * rbln_config.batch_size

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
        rbln_config.set_compile_cfgs([prefill_compile_config])

        return rbln_config

    @classmethod
    def _create_runtimes(
        cls,
        compiled_models: List[rebel.RBLNCompiledModel],
        rbln_config: RBLNDecoderOnlyModelForCausalLMConfig,
    ) -> List[rebel.Runtime]:
        expected_model_names = [
            "prefill",
        ]
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

        if inputs.dim() == 2 and self.rbln_config.use_inputs_embeds:
            inputs = self.get_input_embeddings()(inputs)

        if position_embed is not None:
            position_embed = (
                position_embed[:, :, :, attention_mask.bool(), :] if attention_mask is not None else position_embed
            )

        # padding for chunked prefill
        padding_size = (
            self.rbln_config.prefill_chunk_size - (query_length % self.rbln_config.prefill_chunk_size)
        ) % self.rbln_config.prefill_chunk_size
        padded_len = query_length + padding_size

        inputs = (
            torch.nn.functional.pad(inputs, (0, padding_size))
            if not self.rbln_config.use_inputs_embeds
            else torch.nn.functional.pad(inputs, (0, 0, 0, padding_size))
        )
        position_embed = (
            None if position_embed is None else torch.nn.functional.pad(position_embed, (0, 0, 0, padding_size))
        )
        cache_position = torch.arange(padded_len, dtype=torch.int32).unsqueeze(0)

        chunked_attention_mask = (
            torch.zeros(1, 1, self.rbln_config.prefill_chunk_size, self.rbln_config.max_seq_len, dtype=torch.float32)
            if self.rbln_config.use_attention_mask
            else None
        )

        return inputs, position_embed, cache_position, query_length, chunked_attention_mask

    def _chunked_prefill_forward(
        self,
        inputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embed: Optional[torch.Tensor] = None,
    ):
        padded_input, padded_position_embed, cache_position, query_length, chunked_attention_mask = (
            self._preprocess_chunked_prefill(inputs, attention_mask, position_embed)
        )

        # chunked prefill
        last_hidden_states = []
        for step in range(0, query_length, self.rbln_config.prefill_chunk_size):
            # Extract the current chunk of inputs and cache positions
            input_chunk = padded_input[:, step : step + self.rbln_config.prefill_chunk_size]
            cache_pos_chunk = cache_position[:, step : step + self.rbln_config.prefill_chunk_size]

            valid_length = (
                self.rbln_config.prefill_chunk_size
                if (step + self.rbln_config.prefill_chunk_size) <= query_length
                else query_length - step
            )
            if self.rbln_config.use_local_attention:
                query_position = torch.tensor(valid_length - 1, dtype=torch.int16)
            else:
                query_position = None

            if self.rbln_config.use_attention_mask:
                if step > 0:
                    chunked_attention_mask[:, :, :, :step] = 1
                chunked_attention_mask[:, :, :, step : step + self.rbln_config.prefill_chunk_size] = self.causal_mask

            # Forward pass for the current chunk
            last_hidden_states_chunk = self.prefill_decoder(
                input_ids=input_chunk if not self.rbln_config.use_inputs_embeds else None,
                inputs_embeds=input_chunk if self.rbln_config.use_inputs_embeds else None,
                cache_position=cache_pos_chunk,
                block_tables=self.block_tables if self.rbln_config.use_global_attention else None,
                local_block_tables=self.local_block_tables if self.rbln_config.use_local_attention else None,
                query_position=query_position,
                attention_mask=chunked_attention_mask,
                position_emb=padded_position_embed,
            )
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
    ) -> Tuple[torch.FloatTensor]:
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

        last_hidden_states = torch.concat(all_last_hidden_states, dim=0)
        return BaseModelOutputWithPast(last_hidden_state=last_hidden_states)


class RBLNDecoderOnlyModelForCausalLM(RBLNDecoderOnlyModel):
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
        main_input_name = self.main_input_name

        if self.rbln_config.use_inputs_embeds:
            main_input_name = "inputs_embeds"
            artifacts = torch.load(self.model_save_dir / self.subfolder / "torch_artifacts.pth", weights_only=False)
            self.embed_tokens = self._create_embedding_layer()
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
            rbln_config=self.rbln_config,
            vocab_size=self.config.vocab_size,
        )

        if self.can_generate():
            self.decoders = {}
            for i, batch_size in enumerate(self.rbln_config.decoder_batch_sizes):
                self.decoders[batch_size] = RBLNRuntimeModel(
                    runtime=self.model[i + 1],
                    main_input_name=main_input_name,
                    embed_tokens=self.embed_tokens,
                    phase="decode",
                    batch_size=batch_size,
                    dec_attn_mask=dec_attn_mask,
                    block_tables=block_tables,
                    free_block_pool=free_block_pool,
                    rbln_config=self.rbln_config,
                )

            # NOTE(eunji): Use a decoder whose batch size matches the model's main batch size for compatibility.
            self.decoder = self.decoders[self.rbln_config.batch_size]

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
            model = AutoModelForCausalLM.from_config(config)

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
    def wrap_model_if_needed(cls, model: PreTrainedModel, rbln_config: "RBLNDecoderOnlyModelForCausalLMConfig"):
        wrapper_cfg = {
            "max_seq_len": rbln_config.max_seq_len,
            "attn_impl": rbln_config.attn_impl,
            "kvcache_partition_len": rbln_config.kvcache_partition_len,
            "kvcache_block_size": rbln_config.kvcache_block_size,
            "use_rotary_emb": cls._use_rotary_emb,
            "use_attention_mask": rbln_config.use_attention_mask,
            "use_position_ids": rbln_config.use_position_ids,
            "use_inputs_embeds": rbln_config.use_inputs_embeds,
            "cache_impl": rbln_config.cache_impl,
            "sliding_window": rbln_config.sliding_window,
            "sliding_window_layers": rbln_config.sliding_window_layers,
        }
        return cls._decoder_wrapper_cls(model, **wrapper_cfg).eval()

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
    def use_query_position(cls, use_local_attention: bool, is_prefill: bool = True):
        return is_prefill

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
        rbln_config = super()._update_rbln_config(preprocessors, model, model_config, rbln_config)
        if rbln_config.can_generate:
            compile_configs = rbln_config.compile_cfgs
            for batch_size in rbln_config.decoder_batch_sizes:
                dec_input_info = cls.get_input_info(
                    batch_size=batch_size,
                    query_length=1,
                    rbln_config=rbln_config,
                    model_config=model_config,
                )
                compile_configs.append(
                    RBLNCompileConfig(compiled_model_name=f"decoder_batch_{batch_size}", input_info=dec_input_info)
                )
            rbln_config.set_compile_cfgs(compile_configs)

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

    def get_decoder(self):
        if not self.can_generate():
            raise ValueError("Decode stage is not supported in this model.")
        return self.decoder

    def can_generate(self):
        return self.rbln_config.can_generate

    def _reorder_cache(self, past_key_values, beam_idx):
        raise NotImplementedError

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        generate_idx: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        padded_cache_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        model_inputs = {}
        is_prefill_phase = generate_idx is None

        if is_prefill_phase:
            generate_idx = attention_mask.sum(dim=-1, keepdim=True).int()
            padded_cache_lengths = torch.zeros_like(generate_idx)
            cache_position = None
            position_ids = None
        else:
            if inputs_embeds is not None:
                # if `inputs_embeds` are passed, only use them in the 1st generation step for every prompt.
                inputs_embeds = None

            input_ids = input_ids[:, -1:]
            position_ids = generate_idx
            cache_position = generate_idx + padded_cache_lengths if padded_cache_lengths is not None else generate_idx
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
                "position_ids": position_ids,
                "padded_cache_lengths": padded_cache_lengths,
            }
        )

        return model_inputs

    def _update_model_kwargs_for_generation(
        self,
        outputs: RBLNDecoderOnlyForCausalLMOutput,
        model_kwargs: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, Any]:
        # update generate_idx
        model_kwargs["generate_idx"] = outputs.generate_idx
        model_kwargs["padded_cache_lengths"] = outputs.padded_cache_lengths

        return model_kwargs

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

        # Prefll
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
            return RBLNDecoderOnlyForCausalLMOutput(
                logits=logits, generate_idx=generate_idx, padded_cache_lengths=padded_cache_lengths
            )
