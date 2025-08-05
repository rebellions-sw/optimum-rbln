from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import ModelOutput

from .configuration_decoderonly import RBLNDecoderOnlyModelForCausalLMConfig


@dataclass
class RBLNDecoderOnlyForCausalLMOutput(ModelOutput):
    logits: torch.FloatTensor = None
    generate_idx: torch.Tensor = None
    padded_cache_lengths: int = None


class RBLNDecoderOnlyChunkedPrefillMixin(ABC):
    def _preprocess_chunked_prefill(
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

            if position_embed is not None:
                position_embed = torch.nn.functional.pad(position_embed, (0, 0, 0, padding_size))

            if token_type_ids is not None:
                token_type_ids = torch.nn.functional.pad(token_type_ids, (0, padding_size), value=-1)

        cache_position = torch.arange(query_length + padding_size, dtype=torch.int32).unsqueeze(0)

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

    def _chunked_prefill_forward(
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
                if step != 0:  # update previous chunk
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
            output_logit = self.prefill_runtime(
                input_chunk,
                cache_pos_chunk,
                block_tables,
                local_block_tables,
                position_embed_chunk,
                query_position,
                chunked_attention_mask if self.rbln_config.use_attention_mask else None,
                position_ids_chunk,
                out=out_buffers,
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

    @abstractmethod
    def _postprocess_chunked_prefill(self, *args, **kwargs):
        raise NotImplementedError


class RBLNPageTableManager:
    def __init__(self, rbln_config: RBLNDecoderOnlyModelForCausalLMConfig):
        self.rbln_config = rbln_config
        self.block_tables = torch.zeros(
            self.rbln_config.batch_size,
            self.rbln_config.max_seq_len // self.rbln_config.kvcache_block_size,
            dtype=torch.int16,
        ).fill_(-1)
        self.free_block_pool = deque(x for x in range(self.rbln_config.kvcache_num_blocks))
        self.empty_block = -1

    def get_block_tables(
        self, cache_position: torch.Tensor, batch_idx: int = None, batch_size: int = None, phase: str = "prefill"
    ) -> torch.Tensor:
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

        def get_global_block_tables(batch_idx: int, batch_size: int, phase: str):
            if not self.rbln_config.use_global_attention:
                return None

            if phase == "prefill":
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
                for b_idx in range(batch_size):
                    position = cache_position[b_idx][0].item()
                    block_idx = position // self.rbln_config.kvcache_block_size
                    update_block(b_idx, block_idx)

                return replace_empty_block(self.block_tables)

        def get_local_block_tables(batch_idx: int, batch_size: int, phase: str):
            if not self.rbln_config.use_local_attention:
                return None
            else:
                return (
                    torch.tensor([batch_idx], dtype=torch.int16)
                    if phase == "prefill"
                    else torch.arange(batch_size, dtype=torch.int16).view(batch_size, -1)
                )

        return get_global_block_tables(batch_idx, batch_size, phase), get_local_block_tables(
            batch_idx, batch_size, phase
        )

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


class RBLNDecoderOnlyGenerationMixin(RBLNDecoderOnlyChunkedPrefillMixin, GenerationMixin):
    _supports_cache_class = False  # Needed for GenerationMixin
    _is_stateful = False  # Needed for GenerationMixin

    def _setup_generation_components(self):
        self.dec_attn_mask = torch.zeros(
            self.rbln_config.batch_size, 1, 1, self.rbln_config.max_seq_len, dtype=torch.float32
        )
        self.output_size = [1, self.rbln_config.prefill_chunk_size, self.config.vocab_size] if self.rbln_config.logits_to_keep == 0 else [1, 1, self.rbln_config.logits_to_keep]
        self.causal_mask = 1 - torch.triu(
            torch.ones(1, 1, self.rbln_config.prefill_chunk_size, self.rbln_config.prefill_chunk_size), diagonal=1
        )
        self.page_table_manager = RBLNPageTableManager(self.rbln_config)

        # FIXME: this is a hack to keep backward compatibility with the old generation API
        self.prefill_decoder = self._prefill_forward
        self.decoder = self._decode_forward
        if self.can_generate():
            self.decoders = {}
            for batch_size in self.rbln_config.decoder_batch_sizes:
                self.decoders[batch_size] = self._decode_forward

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
        self, outputs: "RBLNDecoderOnlyForCausalLMOutput", model_kwargs: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        # update generate_idx
        model_kwargs["generate_idx"] = outputs.generate_idx
        model_kwargs["padded_cache_lengths"] = outputs.padded_cache_lengths
        return model_kwargs

    def _postprocess_chunked_prefill(
        self,
        logits: List[torch.Tensor],
        query_length: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        batch_idx: Optional[int] = None,
        is_external_block_tables: Optional[bool] = None,
        padded_cache_lengths: Optional[int] = None,
    ):
        # Update decoder attention mask with processed KV-cache length from prefill phase
        if not is_external_block_tables and self.rbln_config.use_attention_mask:
            self.dec_attn_mask[batch_idx].fill_(0)
            self.dec_attn_mask[batch_idx, :, :, :query_length] = 1

        return RBLNDecoderOnlyForCausalLMOutput(logits=logits, padded_cache_lengths=padded_cache_lengths)

    def _prefill_forward(self, *args, **kwargs):
        return self._forward(*args, phase="prefill", **kwargs)

    def _decode_forward(self, *args, **kwargs):
        return self._forward(*args, phase="decode", **kwargs)

    def _forward(
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
        phase: str = "prefill",
    ):
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Either `input_ids` or `inputs_embeds` must be provided.")

        if inputs_embeds is None:
            inputs = input_ids
            if self.embed_tokens is not None:
                inputs = self.embed_tokens(inputs)
        else:
            inputs = inputs_embeds

        is_external_block_tables = self.page_table_manager.is_external_block_tables(
            block_tables,
            local_block_tables,
        )
        if not is_external_block_tables:
            block_tables, local_block_tables = self.page_table_manager.get_block_tables(
                cache_position, batch_idx=batch_idx, batch_size=inputs.shape[0], phase=phase
            )

        if phase == "decode":
            return self._decode(
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
            return self._chunked_prefill_forward(
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

    def _decode(
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
        if batch_size not in self.rbln_config.decoder_batch_sizes:
            raise RuntimeError(
                f"Batch size mismatch: got {batch_size}, expected one of {self.rbln_config.decoder_batch_sizes} (compiled batch size)."
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

        if self.rbln_config.use_global_attention and batch_size < block_tables.shape[0]:
            block_tables = block_tables[:batch_size]

        if attention_mask is not None and batch_size < attention_mask.shape[0]:
            attention_mask = attention_mask[:batch_size]

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
