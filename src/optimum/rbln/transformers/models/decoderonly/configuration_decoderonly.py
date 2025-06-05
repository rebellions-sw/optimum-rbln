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

from typing import Any, Dict, List, Optional, Union

import rebel

from ....configuration_utils import RBLNModelConfig
from ....utils.logging import get_logger
from ...utils.rbln_quantization import RBLNQuantizationConfig


logger = get_logger()


class RBLNDecoderOnlyModelForCausalLMConfig(RBLNModelConfig):
    def __init__(
        self,
        batch_size: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        use_inputs_embeds: Optional[bool] = None,
        use_attention_mask: Optional[bool] = None,
        use_position_ids: Optional[bool] = None,
        attn_impl: Optional[str] = None,
        kvcache_partition_len: Optional[int] = None,
        kvcache_block_size: Optional[int] = None,
        quantization: Optional[Union[Dict[str, Any], RBLNQuantizationConfig]] = None,
        prefill_chunk_size: Optional[int] = None,
        kvcache_num_blocks: Optional[int] = None,
        decoder_batch_sizes: Optional[List[int]] = None,
        **kwargs,
    ):
        """
        Args:
            batch_size (Optional[int]): The batch size for inference. Defaults to 1.
            max_seq_len (Optional[int]): The maximum sequence length supported by the model.
            use_inputs_embeds (Optional[bool]): Whether to use input embeddings directly. Defaults to False.
            use_attention_mask (Optional[bool]): Whether to use attention masks. This is automatically set to True
                for RBLN-CA02 devices.
            use_position_ids (Optional[bool]): Whether to use position IDs. Defaults to False.
            attn_impl (Optional[str]): The attention implementation to use.
            kvcache_partition_len (Optional[int]): The length of each KV cache partition.
            kvcache_block_size (Optional[int]): The block size for KV cache.
            quantization (Optional[Dict[str, Any]]): Configuration for model quantization.
            prefill_chunk_size (Optional[int]): The chunk size for prefilling the KV cache. Defaults to 128,
                and must be a positive integer divisible by 64.
            kvcache_num_blocks (Optional[int]): The number of blocks in the KV cache.
            decoder_batch_sizes (Optional[List[int]]): A list of batch sizes for which separate decoder models will be compiled.
                This allows the model to handle varying batch sizes efficiently during generation. If not specified,
                defaults to a list containing only the model's main batch size. When specifying multiple batch sizes:
                1) All values must be less than or equal to the main batch size.
                2) The list will be sorted in descending order (larger batch sizes first).
                3) If using multiple decoders, at least one batch size should match the main batch size.

            **kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If batch_size is not a positive integer or if prefill_chunk_size is not
                a positive integer divisible by 64.
        """
        super().__init__(**kwargs)
        self.batch_size = batch_size or 1
        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")

        self.max_seq_len = max_seq_len
        self.use_inputs_embeds = use_inputs_embeds or False
        self.use_position_ids = use_position_ids or False
        self.use_attention_mask = use_attention_mask

        npu = self.npu or rebel.get_npu_name()
        if npu == "RBLN-CA02":
            if self.use_attention_mask is False:
                logger.warning("Attention mask should be used with RBLN-CA02. Setting use_attention_mask to True.")
            self.use_attention_mask = True
        else:
            self.use_attention_mask = self.use_attention_mask or False

        if self.use_position_ids and not self.use_attention_mask:
            raise ValueError("Position IDs should be used with attention mask.")

        self.attn_impl = attn_impl
        self.kvcache_partition_len = kvcache_partition_len
        self.kvcache_block_size = kvcache_block_size
        self.quantization = quantization or {}
        if self.quantization and isinstance(self.quantization, dict):
            self.quantization = RBLNQuantizationConfig(**self.quantization)

        self.prefill_chunk_size = prefill_chunk_size or 128
        if self.prefill_chunk_size % 64 != 0 or self.prefill_chunk_size <= 0:
            raise ValueError("`prefill_chunk_size` must be a positive integer divisible by 64.")

        self.kvcache_num_blocks = kvcache_num_blocks
        self.decoder_batch_sizes = decoder_batch_sizes
        if self.decoder_batch_sizes is None:
            self.decoder_batch_sizes = [self.batch_size]

        if self.use_multiple_decoder:
            if max(self.decoder_batch_sizes) > self.batch_size:
                raise ValueError(
                    f"Decoder batch size ({max(self.decoder_batch_sizes)}) must be less than or equal to the runtime batch size ({self.batch_size})."
                )
            if max(self.decoder_batch_sizes) < self.batch_size:
                logger.warning(
                    f"Maximum decoder batch size ({max(self.decoder_batch_sizes)}) is less than the model's batch size ({self.batch_size}). "
                    "Appending the model's batch size to the decoder batch size."
                )
                self.decoder_batch_sizes.append(self.batch_size)

            # Larger batch size should be at the beginning of the list.
            self.decoder_batch_sizes.sort(reverse=True)

    @property
    def use_multiple_decoder(self):
        return isinstance(self.decoder_batch_sizes, list) and len(self.decoder_batch_sizes) > 1
