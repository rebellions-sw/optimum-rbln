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

from typing import Any, Dict, Optional, Type

from ....configuration_utils import RBLNModelConfig
from ....utils.logging import get_logger
from ...models.clip import RBLNCLIPVisionModelConfig
from ...models.decoderonly import RBLNDecoderOnlyModelForCausalLMConfig


logger = get_logger(__name__)


class RBLNLlavaNextForConditionalGenerationConfig(RBLNModelConfig):
    """
    Configuration class for RBLNLlavaNextForConditionalGeneration.

    This configuration class stores the configuration parameters specific to
    RBLN-optimized LLaVA-Next models for multimodal conditional generation tasks
    that combine vision and language processing capabilities.

    Examples:
        # Using default CLIP vision tower and DecoderOnly language model
        config = RBLNLlavaNextForConditionalGenerationConfig()

        # Using CLIP vision tower with explicit configuration
        config = RBLNLlavaNextForConditionalGenerationConfig(
            vision_tower={
                "image_size": 336,
                "output_hidden_states": True
            }
        )

        # Using different vision tower type
        config = RBLNLlavaNextForConditionalGenerationConfig(
            vision_tower={
                "cls_name": "RBLNSigLIPVisionModelConfig",
                "image_size": 224,
                "output_hidden_states": True
            }
        )
    """

    submodules = ["vision_tower", "language_model"]
    _vision_tower_config_mapping: Dict[str, Type[RBLNModelConfig]] = {
        "RBLNCLIPVisionModelConfig": RBLNCLIPVisionModelConfig,
        # "RBLNSigLIPVisionModelConfig": RBLNSigLIPVisionModelConfig,
    }

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

        if self.batch_size != 1:
            logger.warning("Ignore batch_size for LlavaNext vision tower. It will be set to 1.")

        self.vision_tower = self.initialize_submodule_config(
            submodule_name="vision_tower",
            submodule_config=vision_tower,
            default_config_cls=RBLNCLIPVisionModelConfig,
            config_type_mapping=self._vision_tower_config_mapping,
            batch_size=1,
        )

        if hasattr(self.vision_tower, "output_hidden_states"):
            if self.vision_tower.output_hidden_states is False:
                raise ValueError(
                    f"LlavaNext requires output_hidden_states to be True, but found output_hidden_states={self.vision_tower.output_hidden_states}. "
                    f"Please compile again with the correct argument."
                )
            else:
                self.vision_tower.output_hidden_states = True
        else:
            logger.warning(
                f"Vision tower config {type(self.vision_tower).__name__} does not have output_hidden_states attribute. "
                f"This might cause issues with LlavaNext."
            )

        self.language_model = self.initialize_submodule_config(
            submodule_name="language_model",
            submodule_config=language_model,
            default_config_cls=RBLNDecoderOnlyModelForCausalLMConfig,
            batch_size=self.batch_size,
        )
