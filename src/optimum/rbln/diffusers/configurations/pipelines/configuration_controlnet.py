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

from ....configuration_utils import RBLNModelConfig
from ....transformers import RBLNCLIPTextModelConfig, RBLNCLIPTextModelWithProjectionConfig
from ....utils.logging import get_logger
from ..models import RBLNAutoencoderKLConfig, RBLNControlNetModelConfig, RBLNUNet2DConditionModelConfig


logger = get_logger(__name__)


class _RBLNStableDiffusionControlNetPipelineBaseConfig(RBLNModelConfig):
    submodules = ["text_encoder", "unet", "vae", "controlnet"]
    _vae_uses_encoder = False

    def __init__(
        self,
        batch_size: Optional[int] = None,
        text_encoder: Optional[RBLNModelConfig] = None,
        unet: Optional[RBLNModelConfig] = None,
        vae: Optional[RBLNModelConfig] = None,
        controlnet: Optional[RBLNModelConfig] = None,
        guidance_scale: Optional[float] = None,
        vae_uses_encoder: Optional[bool] = None,
        *,
        img_height: Optional[int] = None,
        img_width: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.batch_size = batch_size or 1
        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")

        self.text_encoder = self.init_submodule_config(RBLNCLIPTextModelConfig, text_encoder, batch_size=batch_size)
        self.unet = self.init_submodule_config(
            RBLNUNet2DConditionModelConfig, unet, batch_size=batch_size, img_height=img_height, img_width=img_width
        )
        self.vae = self.init_submodule_config(
            RBLNAutoencoderKLConfig,
            vae,
            batch_size=batch_size,
            uses_encoder=vae_uses_encoder or self.__class__._vae_uses_encoder,
            img_height=img_height,
            img_width=img_width,
        )
        self.controlnet = self.init_submodule_config(RBLNControlNetModelConfig, controlnet, batch_size=batch_size)

        if guidance_scale is not None:
            logger.warning("Specifying `guidance_scale` is deprecated. It will be removed in a future version.")
            do_classifier_free_guidance = guidance_scale > 1.0
            if do_classifier_free_guidance:
                self.unet.batch_size = self.batch_size * 2


class RBLNStableDiffusionControlNetPipelineConfig(_RBLNStableDiffusionControlNetPipelineBaseConfig):
    _vae_uses_encoder = False


class RBLNStableDiffusionImg2ImgControlNetPipelineConfig(_RBLNStableDiffusionControlNetPipelineBaseConfig):
    _vae_uses_encoder = True


class _RBLNStableDiffusionXLControlNetPipelineBaseConfig(RBLNModelConfig):
    submodules = ["text_encoder", "text_encoder_2", "unet", "vae", "controlnet"]
    _vae_uses_encoder = False

    def __init__(
        self,
        batch_size: Optional[int] = None,
        text_encoder: Optional[RBLNModelConfig] = None,
        text_encoder_2: Optional[RBLNModelConfig] = None,
        unet: Optional[RBLNModelConfig] = None,
        vae: Optional[RBLNModelConfig] = None,
        controlnet: Optional[RBLNModelConfig] = None,
        guidance_scale: Optional[float] = None,
        vae_uses_encoder: Optional[bool] = None,
        *,
        img_height: Optional[int] = None,
        img_width: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.batch_size = batch_size or 1
        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")

        self.text_encoder = self.init_submodule_config(RBLNCLIPTextModelConfig, text_encoder, batch_size=batch_size)
        self.text_encoder_2 = self.init_submodule_config(
            RBLNCLIPTextModelWithProjectionConfig, text_encoder_2, batch_size=batch_size
        )
        self.unet = self.init_submodule_config(
            RBLNUNet2DConditionModelConfig, unet, batch_size=batch_size, img_height=img_height, img_width=img_width
        )
        self.vae = self.init_submodule_config(
            RBLNAutoencoderKLConfig,
            vae,
            batch_size=batch_size,
            uses_encoder=vae_uses_encoder or self.__class__._vae_uses_encoder,
            img_height=img_height,
            img_width=img_width,
        )
        self.controlnet = self.init_submodule_config(RBLNControlNetModelConfig, controlnet, batch_size=batch_size)

        if guidance_scale is not None:
            logger.warning("Specifying `guidance_scale` is deprecated. It will be removed in a future version.")
            do_classifier_free_guidance = guidance_scale > 1.0
            if do_classifier_free_guidance:
                self.unet.batch_size = self.batch_size * 2


class RBLNStableDiffusionXLControlNetPipelineConfig(_RBLNStableDiffusionXLControlNetPipelineBaseConfig):
    _vae_uses_encoder = False


class RBLNStableDiffusionXLImg2ImgControlNetPipelineConfig(_RBLNStableDiffusionXLControlNetPipelineBaseConfig):
    _vae_uses_encoder = True
