from collections import deque
from typing import Optional

import torch

from .configuration_decoderonly import RBLNDecoderOnlyModelForCausalLMConfig


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
