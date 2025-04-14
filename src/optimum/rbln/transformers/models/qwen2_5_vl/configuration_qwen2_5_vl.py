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

from typing import List, Optional, Union

from ....configuration_utils import RBLNModelConfig
from ....utils.logging import get_logger
from ..decoderonly.configuration_decoderonly import RBLNDecoderOnlyModelForCausalLMConfig


logger = get_logger()


class RBLNQwen2_5_VLForConditionalGenerationConfig(RBLNDecoderOnlyModelForCausalLMConfig):

    def __init__(
        self,
        visual: Optional[RBLNModelConfig] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.visual = visual

class RBLNQwen2_5_VisionTransformerPretrainedModelConfig(RBLNModelConfig):
    def __init__(self, max_seq_lens: Optional[Union[int, List[int]]] = None, **kwargs):
        """
        Args:
            max_seq_lens (Optional[Union[int, List[int]]): The lengths of seq in ViT attention.
                Can be an integer a List of integer.
            **kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If batch_size is not a positive integer.
        """
        super().__init__(**kwargs)
        self.use_inputs_embeds = True
        if isinstance(max_seq_lens, int):
            max_seq_lens = [max_seq_lens]
        elif isinstance(max_seq_lens, list):
            max_seq_lens = max_seq_lens.sort(reverse=True)

        self.max_seq_lens = max_seq_lens

