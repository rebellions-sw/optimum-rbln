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

from ....configuration_utils import RBLNModelConfig
from ..decoderonly.configuration_decoderonly import RBLNDecoderOnlyModelForCausalLMConfig


class RBLNQwen2VLForConditionalGenerationConfig(RBLNDecoderOnlyModelForCausalLMConfig):
    submodules = ["visual"]

    def __init__(
        self,
        visual: Optional[RBLNModelConfig] = None,
        use_inputs_embeds: bool = True,
        **kwargs: Dict[str, Any],
    ):
        super().__init__(use_inputs_embeds=use_inputs_embeds, **kwargs)
        if not self.use_inputs_embeds:
            raise ValueError(
                "RBLNQwen2VLForConditionalGenerationConfig does not allow `use_inputs_embeds` to be set to False, "
                "as RBLNQwen2VLForConditionalGeneration accepts only `inputs_embeds` as input."
            )
        self.visual = visual


class RBLNQwen2VisionTransformerPretrainedModelConfig(RBLNModelConfig):
    def __init__(self, max_seq_lens: Union[int, List[int]] = None, **kwargs: Dict[str, Any]):
        """
        Args:
            max_seq_lens (Optional[Union[int, List[int]]]): Maximum sequence lengths for Vision
                Transformer attention. Can be an integer or list of integers, each indicating
                the number of patches in a sequence for an image or video. For example, an image
                of 224x224 pixels with patch size 14 results in (224/14) * (224/14) = 256 patches,
                so `max_seq_lens` must be at least 256. RBLN optimization runs inference per image
                or video frame, so set `max_seq_lens` to match the maximum expected resolution to
                optimize computation. If not provided, a `ValueError` is raised.
            **kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If batch_size is not a positive integer.

        Max Seq Lens:
            Since `Qwen2VLForConditionalGeneration` performs inference on a per-image or per-frame basis,
            `max_seq_lens` should be set based on the maximum expected resolution of the input images or video frames.

            The value must be greater than or equal to the number of patches generated from the input image.
            For example, a 224x224 image with a patch size of 14 results in (224 / 14) * (224 / 14) = 256 patches.
            Therefore, `max_seq_lens` must be at least 256.
        """
        super().__init__(**kwargs)

        if max_seq_lens is not None:
            if isinstance(max_seq_lens, int):
                max_seq_lens = [max_seq_lens]
            elif isinstance(max_seq_lens, list):
                max_seq_lens.sort(reverse=True)
        else:
            raise ValueError("'max_seq_lens' must be specified.")

        self.max_seq_lens = max_seq_lens
