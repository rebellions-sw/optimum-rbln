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


class RBLNIdefics3VisionTransformerConfig(RBLNModelConfig):
    pass


class RBLNIdefics3ForConditionalGenerationConfig(RBLNModelConfig):
    """
    Configuration class for RBLNIdefics3ForConditionalGeneration models.

    This class extends `RBLNModelConfig` to include settings specific to the Idefics3 vision-language model optimized for RBLN devices.
    It allows configuration of the batch size and separate configurations for the vision and text submodules.

    Attributes:
        submodules (List[str]): List of submodules included in the model. Defaults to `["vision_model", "text_model"]`.
    """

    submodules = ["vision_model", "text_model"]

    def __init__(
        self,
        batch_size: Optional[int] = None,
        vision_model: Optional[RBLNModelConfig] = None,
        text_model: Optional[RBLNModelConfig] = None,
        **kwargs: Any,
    ):
        """
        Args:
            batch_size (Optional[int]): The batch size for inference. Defaults to 1.
            vision_model (Optional[RBLNModelConfig]): Configuration for the vision transformer component.
            text_model (Optional[RBLNModelConfig]): Configuration for the text model component.
            **kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If batch_size is not a positive integer.
        """

        super().__init__(**kwargs)
        self.batch_size = batch_size or 1
        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")

        self.vision_model = vision_model
        self.text_model = text_model
