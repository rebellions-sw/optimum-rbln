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

from typing import TYPE_CHECKING, Any, Deque, Optional

import rebel
import torch

from ....utils.logging import get_logger
from ....utils.runtime_utils import RBLNPytorchRuntime


logger = get_logger()

if TYPE_CHECKING:
    pass


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
