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

from optimum.rbln.configuration_utils import RBLNModelConfig
# from optimum.rbln.transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
#     RBLNQwen2_5_VLForConditionalGenerationConfig,
# )

from ..decoderonly.configuration_decoderonly import RBLNDecoderOnlyModelConfig


class RBLNColQwen2ForRetrievalConfig(RBLNDecoderOnlyModelConfig):
    submodules = ["visual"]

    def __init__(
        self,
        visual: Optional[RBLNModelConfig] = None,
        batch_size: Optional[int] = None,
        use_inputs_embeds: bool = True,
        output_hidden_states: Optional[bool] = True,
        **kwargs,
    ):
        super().__init__(use_inputs_embeds=use_inputs_embeds, **kwargs)
        if not self.use_inputs_embeds:
            raise ValueError(
                "RBLNColQwen2ForRetrievalConfig does not allow `use_inputs_embeds` to be set to False, "
                "as RBLNColQwen2ForRetrieval accepts only `inputs_embeds` as input."
            )
        if batch_size is not None and batch_size != 1:
            raise ValueError(
                "batch_size is not supported for RBLNColQwen2ForRetrievalConfig"
            )

        self.visual = visual
        self.output_hidden_states = output_hidden_states
