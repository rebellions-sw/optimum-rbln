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
from ..decoderonly.configuration_decoderonly import RBLNDecoderOnlyModelForCausalLMConfig


class RBLNColQwen2_5ForConditionalGenerationConfig(RBLNDecoderOnlyModelForCausalLMConfig):
    # submodules = ["visual"]

    def __init__(
        self,
        visual: Optional[RBLNModelConfig] = None,
        use_inputs_embeds: bool = True,
        **kwargs,
    ):
        super().__init__(use_inputs_embeds=use_inputs_embeds, **kwargs)
        if not self.use_inputs_embeds:
            raise ValueError(
                "RBLNColQwen2_5ForConditionalGenerationConfig does not allow `use_inputs_embeds` to be set to False, "
                "as RBLNColQwen2_5ForConditionalGeneration accepts only `inputs_embeds` as input."
            )
        self.visual = visual
