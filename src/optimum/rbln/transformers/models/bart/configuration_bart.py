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

from ...configuration_generic import RBLNTransformerEncoderForFeatureExtractionConfig
from ..seq2seq import RBLNModelForSeq2SeqLMConfig


class RBLNBartModelConfig(RBLNTransformerEncoderForFeatureExtractionConfig):
    """
    Configuration class for RBLNBartModel.

    This configuration class stores the configuration parameters specific to
    RBLN-optimized BART models for feature extraction tasks.
    """


class RBLNBartForConditionalGenerationConfig(RBLNModelForSeq2SeqLMConfig):
    """
    Configuration class for RBLNBartForConditionalGeneration.

    This configuration class stores the configuration parameters specific to
    RBLN-optimized BART models for conditional text generation tasks.
    """

    def __init__(
        self,
        *args,
        kvcache_num_blocks: Optional[int] = None,
        kvcache_block_size: Optional[int] = None,
        **kwargs,
    ):
        """
        Args:
            kvcache_num_blocks (Optional[int]): The total number of blocks to allocate for the
                PagedAttention KV cache for the SelfAttention. Defaults to batch_size.
            kvcache_block_size (Optional[int]): Sets the size (in number of tokens) of each block
                in the PagedAttention KV cache for the SelfAttention. Defaults to dec_max_seq_len.

        **kwargs: Any,
        """
        super().__init__(*args, **kwargs)
        self.kvcache_num_blocks = kvcache_num_blocks
        self.kvcache_block_size = kvcache_block_size
