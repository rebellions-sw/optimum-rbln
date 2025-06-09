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
from ..decoderonly.configuration_decoderonly import RBLNDecoderOnlyModelForCausalLMConfig
from ..siglip.configuration_siglip import RBLNSiglipVisionModelConfig


class RBLNGemma3ForCausalLMConfig(RBLNDecoderOnlyModelForCausalLMConfig):
    def __init__(
        self,
        prefill_chunk_size: Optional[int] = None,
        use_position_ids: Optional[bool] = None,
        use_attention_mask: Optional[bool] = None,
        **kwargs,
    ):
        # use_attention_mask and use_position_ids are always True for Gemma3
        use_attention_mask = use_attention_mask or True
        use_position_ids = use_position_ids or True
        prefill_chunk_size = prefill_chunk_size or 256

        super().__init__(
            prefill_chunk_size=prefill_chunk_size,
            use_attention_mask=use_attention_mask,
            use_position_ids=use_position_ids,
            **kwargs,
        )

        npu = self.npu or rebel.get_npu_name()
        if npu == "RBLN-CA02":
            raise NotImplementedError("Gemma3 is currently not supported on RBLN-CA02")


class RBLNGemma3ForConditionalGenerationConfig(RBLNModelConfig):
    submodules = ["vision_tower", "language_model"]

    def __init__(
        self,
        batch_size: Optional[int] = None,
        vision_tower: Optional[RBLNModelConfig] = None,
        language_model: Optional[RBLNModelConfig] = None,
        **kwargs,
    ):
        """
        Args:
            batch_size (Optional[int]): The batch size for inference. Defaults to 1.
            vision_tower (Optional[RBLNModelConfig]): Configuration for the vision encoder component.
            language_model (Optional[RBLNModelConfig]): Configuration for the language model component.
            **kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If batch_size is not a positive integer.
        """
        super().__init__(**kwargs)
        self.batch_size = batch_size or 1
        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")

        self.vision_tower = self.init_submodule_config(RBLNSiglipVisionModelConfig, vision_tower)
        self.language_model = self.init_submodule_config(RBLNGemma3ForCausalLMConfig, language_model)
