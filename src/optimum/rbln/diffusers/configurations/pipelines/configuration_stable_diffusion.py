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

from typing import Optional, Tuple

from ....configuration_utils import RBLNModelConfig
from ....transformers import RBLNCLIPTextModelConfig
from ....utils.logging import get_logger
from ..models import RBLNAutoencoderKLConfig, RBLNUNet2DConditionModelConfig


logger = get_logger(__name__)


class _RBLNStableDiffusionPipelineBaseConfig(RBLNModelConfig):
    submodules = ["text_encoder", "unet", "vae"]
    _vae_uses_encoder = False

    def __init__(
        self,
        text_encoder: Optional[RBLNModelConfig] = None,
        unet: Optional[RBLNModelConfig] = None,
        vae: Optional[RBLNModelConfig] = None,
        *,
        batch_size: Optional[int] = None,
        img_height: Optional[int] = None,
        img_width: Optional[int] = None,
        sample_size: Optional[Tuple[int, int]] = None,
        image_size: Optional[Tuple[int, int]] = None,
        guidance_scale: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if image_size is not None and (img_height is not None or img_width is not None):
            raise ValueError("image_size and img_height/img_width cannot both be provided")

        if img_height is not None and img_width is not None:
            image_size = (img_height, img_width)

        self.text_encoder = self.init_submodule_config(RBLNCLIPTextModelConfig, text_encoder, batch_size=batch_size)
        self.unet = self.init_submodule_config(
            RBLNUNet2DConditionModelConfig,
            unet,
            batch_size=batch_size,
            sample_size=sample_size,
        )
        self.vae = self.init_submodule_config(
            RBLNAutoencoderKLConfig,
            vae,
            batch_size=batch_size,
            uses_encoder=self.__class__._vae_uses_encoder,
            sample_size=image_size,  # image size is equal to sample size in vae
        )

        if guidance_scale is not None:
            logger.warning("Specifying `guidance_scale` is deprecated. It will be removed in a future version.")
            do_classifier_free_guidance = guidance_scale > 1.0
            if do_classifier_free_guidance:
                self.unet.batch_size = batch_size * 2

    @property
    def batch_size(self):
        return self.vae.batch_size

    @property
    def sample_size(self):
        return self.unet.sample_size

    @property
    def image_size(self):
        return self.vae.sample_size


class RBLNStableDiffusionPipelineConfig(_RBLNStableDiffusionPipelineBaseConfig):
    _vae_uses_encoder = False


class RBLNStableDiffusionImg2ImgPipelineConfig(_RBLNStableDiffusionPipelineBaseConfig):
    _vae_uses_encoder = True


class RBLNStableDiffusionInpaintPipelineConfig(_RBLNStableDiffusionPipelineBaseConfig):
    _vae_uses_encoder = True
