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

from typing import Any, Optional

from ....configuration_utils import RBLNModelConfig
from ....utils.logging import get_logger
from ...models.clip import RBLNCLIPVisionModelConfig


logger = get_logger(__name__)


class RBLNLlavaNextForConditionalGenerationConfig(RBLNModelConfig):
    """
    Configuration class for RBLNLlavaNextForConditionalGeneration.

    This configuration class stores the configuration parameters specific to
    RBLN-optimized LLaVA-Next models for multimodal conditional generation tasks
    that combine vision and language processing capabilities.
    """

    submodules = ["vision_tower", "language_model"]

    def __init__(
        self,
        batch_size: Optional[int] = None,
        vision_tower: Optional[RBLNModelConfig] = None,
        language_model: Optional[RBLNModelConfig] = None,
        **kwargs: Any,
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

        self.vision_tower = self.init_submodule_config(
            RBLNCLIPVisionModelConfig,
            vision_tower,
        )

        if self.vision_tower.output_hidden_states is False:
            raise ValueError(
                f"LlavaNext requires output_hidden_states to be True, but found output_hidden_states={self.vision_tower.output_hidden_states}. "
                f"Please compile again with the correct argument."
            )
        else:
            self.vision_tower.output_hidden_states = True

        self.language_model = language_model
