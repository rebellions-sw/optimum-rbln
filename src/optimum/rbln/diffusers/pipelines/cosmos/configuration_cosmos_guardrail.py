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

from typing import Any, Dict, Optional, Tuple

from ....configuration_utils import RBLNAutoConfig, RBLNModelConfig
from ....transformers import RBLNSiglipVisionModelConfig


class RBLNVideoSafetyModelConfig(RBLNModelConfig):
    """
    Configuration class for RBLN Video Content Safety Filter.
    """

    def __init__(
        self,
        batch_size: Optional[int] = None,
        input_size: Optional[int] = None,
        image_size: Optional[Tuple[int, int]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.batch_size = batch_size or 1
        self.input_size = input_size or 1152


class RBLNRetinaFaceFilterConfig(RBLNModelConfig):
    """
    Configuration class for RBLN Retina Face Filter.
    """

    def __init__(
        self,
        batch_size: Optional[int] = None,
        image_size: Optional[Tuple[int, int]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.batch_size = batch_size or 1
        self.image_size = image_size or (704, 1280)


class RBLNCosmosSafetyCheckerConfig(RBLNModelConfig):
    """
    Configuration class for RBLN Cosmos Safety Checker.
    """

    submodules = ["aegis", "video_safety_model", "face_blur_filter", "siglip_encoder"]

    def __init__(
        self,
        aegis: Optional[RBLNModelConfig] = None,
        video_safety_model: Optional[RBLNModelConfig] = None,
        face_blur_filter: Optional[RBLNModelConfig] = None,
        siglip_encoder: Optional[RBLNSiglipVisionModelConfig] = None,
        *,
        batch_size: Optional[int] = None,
        image_size: Optional[Tuple[int, int]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        **kwargs: Dict[str, Any],
    ):
        super().__init__(**kwargs)
        if height is not None and width is not None:
            image_size = (height, width)

        self.aegis = self.init_submodule_config(RBLNModelConfig, aegis)
        self.siglip_encoder = self.init_submodule_config(
            RBLNSiglipVisionModelConfig,
            siglip_encoder,
            batch_size=batch_size,
            image_size=(384, 384),
        )

        self.video_safety_model = self.init_submodule_config(
            RBLNVideoSafetyModelConfig,
            video_safety_model,
            batch_size=batch_size,
            input_size=1152,
        )
        self.face_blur_filter = self.init_submodule_config(
            RBLNRetinaFaceFilterConfig,
            face_blur_filter,
            batch_size=batch_size,
            image_size=image_size,
        )


RBLNAutoConfig.register(RBLNVideoSafetyModelConfig)
RBLNAutoConfig.register(RBLNRetinaFaceFilterConfig)
RBLNAutoConfig.register(RBLNCosmosSafetyCheckerConfig)
