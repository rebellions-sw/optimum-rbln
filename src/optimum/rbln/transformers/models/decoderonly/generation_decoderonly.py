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


class RBLNPageTableManager:
    EMPTY_BLOCK = -1
    NO_BLOCKS_ERROR = (
        "No memory blocks are available for allocation. "
        "The generate() API cannot complete this inference task because Paged Attention is not fully supported by optimum-rbln. "
        "This is supported by vllm-rbln (see: https://docs.rbln.ai/software/model_serving/vllm_support/vllm-rbln.html). "
        "Using vllm-rbln should fix this issue and enhance inference performance."
    )

    def __init__(self, rbln_config: RBLNDecoderOnlyModelForCausalLMConfig):
        self.rbln_config = rbln_config
        self.block_tables = torch.zeros(
            self.rbln_config.batch_size,
            self.rbln_config.max_seq_len // self.rbln_config.kvcache_block_size,
            dtype=torch.int16,
        ).fill_(self.EMPTY_BLOCK)
        self.free_block_pool = deque(x for x in range(self.rbln_config.kvcache_num_blocks))

    def update_block(self, batch_idx: int, block_idx: int):
        """
        If the block is empty (empty_block), allocates a block from the free_block_pool.
        """
        if self.block_tables[batch_idx][block_idx] == self.EMPTY_BLOCK:
            if self.free_block_pool:
                block = self.free_block_pool.popleft()
                self.block_tables[batch_idx][block_idx] = block
            else:
                raise RuntimeError(self.NO_BLOCKS_ERROR)

    def replace_empty_block(self, block_tables: torch.Tensor):
        """
        Replaces all occurrences of `self.empty_block` in `block_tables` with a dummy block from `self.free_block_pool`.
        """
        if not torch.any(block_tables == self.EMPTY_BLOCK):
            return block_tables.clone()
        elif self.free_block_pool:
            _free_block = self.free_block_pool[0]
            return torch.where(block_tables == self.EMPTY_BLOCK, _free_block, block_tables)
        else:
            raise RuntimeError(self.NO_BLOCKS_ERROR)

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

        def get_global_block_tables():
            if not self.rbln_config.use_global_attention:
                return None

            if phase == "prefill":
                # Track previously used blocks and return them to the free_block_pool and
                # reset the current batch's block table to empty blocks
                prev_blocks = self.block_tables[batch_idx][self.block_tables[batch_idx] != self.EMPTY_BLOCK].tolist()
                self.free_block_pool.extend(prev_blocks)
                self.block_tables[batch_idx].fill_(self.EMPTY_BLOCK)

                # Get the start (s) and end (e) positions from cache_position and
                # iterate over the cache positions to allocate necessary blocks
                s, e = cache_position[0][0].item(), cache_position[0][-1].item()
                for position in range(s, e + 1, self.rbln_config.kvcache_block_size):
                    block_idx = position // self.rbln_config.kvcache_block_size
                    if batch_idx >= len(self.block_tables) or block_idx >= len(self.block_tables[batch_idx]):
                        raise IndexError(f"Invalid index: batch_idx={batch_idx}, block_idx={block_idx}")
                    self.update_block(batch_idx, block_idx)

                return self.replace_empty_block(self.block_tables[batch_idx])
            # Case for 'decoder' phase, iterate over the cache positions to allocate necessary blocks
            else:
                for b_idx in range(batch_size):
                    position = cache_position[b_idx][0].item()
                    block_idx = position // self.rbln_config.kvcache_block_size
                    self.update_block(b_idx, block_idx)

                return self.replace_empty_block(self.block_tables)

        def get_local_block_tables():
            if not self.rbln_config.use_local_attention:
                return None
            else:
                return (
                    torch.tensor([batch_idx], dtype=torch.int16)
                    if phase == "prefill"
                    else torch.arange(batch_size, dtype=torch.int16).view(batch_size, -1)
                )

        return get_global_block_tables(), get_local_block_tables()

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

    def get_block_tables_if_needed(
        self,
        batch_size,
        cache_position: torch.Tensor,
        batch_idx: int = None,
        phase: str = "prefill",
        block_tables: Optional[torch.Tensor] = None,
        local_block_tables: Optional[torch.Tensor] = None,
    ):
        is_external_block_tables = self.is_external_block_tables(block_tables, local_block_tables)
        if not is_external_block_tables:
            block_tables, local_block_tables = self.get_block_tables(
                cache_position, batch_idx=batch_idx, batch_size=batch_size, phase=phase
            )

        return block_tables, local_block_tables, is_external_block_tables


class RBLNDecoderOnlyGenerationMixin(GenerationMixin):
    _supports_cache_class = False  # Needed for GenerationMixin
    _is_stateful = False  # Needed for GenerationMixin

    def setup_generation_components(self):
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

        inputs = self.inputs_embeddings_if_needed(input_ids, inputs_embeds)
        block_tables, local_block_tables, is_external_block_tables = (
            self.page_table_manager.get_block_tables_if_needed(
                inputs.shape[0],
                cache_position,
                batch_idx=batch_idx,
                phase=phase,
                block_tables=block_tables,
                local_block_tables=local_block_tables,
            )
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

    def _validate_decoder_batch_size(self, inputs: torch.Tensor, **kwargs):
        batch_size = inputs.shape[0]
        if batch_size not in self.rbln_config.decoder_batch_sizes:
            raise RuntimeError(
                f"Batch size mismatch: got {batch_size}, expected one of {self.rbln_config.decoder_batch_sizes} (compiled batch size)."
            )

        for arg_name, arg_value in kwargs.items():
            if arg_value is not None and arg_value.shape[0] != batch_size:
                raise RuntimeError(f"{arg_name} batch size mismatch: got {arg_value.shape[0]}, expected {batch_size}.")

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
        self._validate_decoder_batch_size(
            inputs,
            cache_position=cache_position,
            block_tables=block_tables,
            attention_mask=attention_mask,
            position_embed=position_embed,
            position_ids=position_ids,
        )

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
