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

from typing import Optional

import rebel

from ....configuration_utils import RBLNModelConfig
from ....utils.logging import get_logger


logger = get_logger()


class RBLNDecoderOnlyModelForCausalLMConfig(RBLNModelConfig):
    def __init__(
        self,
        batch_size: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        use_inputs_embeds: Optional[bool] = None,
        use_attention_mask: Optional[bool] = None,
        attn_impl: Optional[str] = None,
        kvcache_partition_len: Optional[int] = None,
        kvcache_block_size: Optional[int] = None,
        quantization: Optional[str] = None,
        prefill_chunk_size: Optional[int] = None,
        kvcache_num_blocks: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.batch_size = batch_size or 1
        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")

        self.max_seq_len = max_seq_len
        self.use_inputs_embeds = use_inputs_embeds or False

        self.use_attention_mask = use_attention_mask
        npu = self.npu or rebel.get_npu_name()
        if npu == "RBLN-CA02":
            if self.use_attention_mask is False:
                logger.warning("Attention mask should be used with RBLN-CA02. Setting use_attention_mask to True.")
            self.use_attention_mask = True
        else:
            self.use_attention_mask = self.use_attention_mask or False

        self.attn_impl = attn_impl
        self.kvcache_partition_len = kvcache_partition_len
        self.kvcache_block_size = kvcache_block_size
        self.quantization = quantization

        self.prefill_chunk_size = prefill_chunk_size or 128
        if self.prefill_chunk_size % 64 != 0 or self.prefill_chunk_size <= 0:
            raise ValueError("`prefill_chunk_size` must be a positive integer divisible by 64.")

        self.kvcache_num_blocks = kvcache_num_blocks
