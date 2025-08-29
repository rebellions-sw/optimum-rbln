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


class RBLNBlip2VisionModelConfig(RBLNModelConfig):
    """
    Configuration class for RBLNBlip2VisionModel.

    This configuration class stores the configuration parameters specific to
    RBLN-optimized BLIP-2 vision encoder models for multimodal tasks.
    """


class RBLNBlip2QFormerModelConfig(RBLNModelConfig):
    """
    Configuration class for RBLNBlip2QFormerModel.

    This configuration class stores the configuration parameters specific to
    RBLN-optimized BLIP-2 Q-Former models that bridge vision and language modalities.
    """

    def __init__(
        self,
        num_query_tokens: Optional[int] = None,
        image_text_hidden_size: Optional[int] = None,
        **kwargs,
    ):
        """
        Args:
            batch_size (Optional[int]): The batch size for inference. Defaults to 1.
            **kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If batch_size is not a positive integer.
        """
        super().__init__(**kwargs)
        self.num_query_tokens = num_query_tokens
        self.image_text_hidden_size = image_text_hidden_size


class RBLNBlip2ForConditionalGenerationConfig(RBLNModelConfig):
    submodules = ["vision_model", "qformer", "language_model"]

    def __init__(
        self,
        batch_size: Optional[int] = None,
        vision_model: Optional[RBLNModelConfig] = None,
        qformer: Optional[RBLNModelConfig] = None,
        language_model: Optional[RBLNModelConfig] = None,
        **kwargs: Any,
    ):
        """
        Args:
            batch_size (Optional[int]): The batch size for inference. Defaults to 1.
            vision_model (Optional[RBLNModelConfig]): Configuration for the vision encoder component.
            language_model (Optional[RBLNModelConfig]): Configuration for the language model component.
            **kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If batch_size is not a positive integer.
        """
        super().__init__(**kwargs)
        self.batch_size = batch_size or 1
        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")

        self.vision_model = self.init_submodule_config(RBLNBlip2VisionModelConfig, vision_model)
        self.language_model = language_model
        self.qformer = self.init_submodule_config(RBLNBlip2QFormerModelConfig, qformer)
