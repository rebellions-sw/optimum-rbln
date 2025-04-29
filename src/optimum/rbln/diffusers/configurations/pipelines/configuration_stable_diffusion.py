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
from ..models import RBLNAutoencoderKLConfig, RBLNUNet2DConditionModelConfig


class _RBLNStableDiffusionPipelineBaseConfig(RBLNModelConfig):
    submodules = ["text_encoder", "unet", "vae"]
    _vae_uses_encoder = False

    def __init__(
        self,
        text_encoder: Optional[RBLNCLIPTextModelConfig] = None,
        unet: Optional[RBLNUNet2DConditionModelConfig] = None,
        vae: Optional[RBLNAutoencoderKLConfig] = None,
        *,
        batch_size: Optional[int] = None,
        img_height: Optional[int] = None,
        img_width: Optional[int] = None,
        sample_size: Optional[Tuple[int, int]] = None,
        image_size: Optional[Tuple[int, int]] = None,
        guidance_scale: Optional[float] = None,
        **kwargs,
    ):
        """
        Args:
            text_encoder (Optional[RBLNCLIPTextModelConfig]): Configuration for the text encoder component.
                Initialized as RBLNCLIPTextModelConfig if not provided.
            unet (Optional[RBLNUNet2DConditionModelConfig]): Configuration for the UNet model component.
                Initialized as RBLNUNet2DConditionModelConfig if not provided.
            vae (Optional[RBLNAutoencoderKLConfig]): Configuration for the VAE model component.
                Initialized as RBLNAutoencoderKLConfig if not provided.
            batch_size (Optional[int]): Batch size for inference, applied to all submodules.
            img_height (Optional[int]): Height of the generated images.
            img_width (Optional[int]): Width of the generated images.
            sample_size (Optional[Tuple[int, int]]): Spatial dimensions for the UNet model.
            image_size (Optional[Tuple[int, int]]): Alternative way to specify image dimensions.
                Cannot be used together with img_height/img_width.
            guidance_scale (Optional[float]): Scale for classifier-free guidance.
            **kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If both image_size and img_height/img_width are provided.

        Note:
            When guidance_scale > 1.0, the UNet batch size is automatically doubled to
            accommodate classifier-free guidance.
        """
        super().__init__(**kwargs)
        if image_size is not None and (img_height is not None or img_width is not None):
            raise ValueError("image_size and img_height/img_width cannot both be provided")

        if img_height is not None and img_width is not None:
            image_size = (img_height, img_width)

        self.text_encoder = self.init_submodule_config(RBLNCLIPTextModelConfig, text_encoder, batch_size=batch_size)
        self.unet = self.init_submodule_config(
            RBLNUNet2DConditionModelConfig,
            unet,
            sample_size=sample_size,
        )
        self.vae = self.init_submodule_config(
            RBLNAutoencoderKLConfig,
            vae,
            batch_size=batch_size,
            uses_encoder=self.__class__._vae_uses_encoder,
            sample_size=image_size,  # image size is equal to sample size in vae
        )

        # Get default guidance scale from original class to set UNet batch size
        if guidance_scale is None:
            guidance_scale = self.get_default_values_for_original_cls("__call__", ["guidance_scale"])["guidance_scale"]

        if not self.unet.batch_size_is_specified:
            do_classifier_free_guidance = guidance_scale > 1.0
            if do_classifier_free_guidance:
                self.unet.batch_size = self.text_encoder.batch_size * 2
            else:
                self.unet.batch_size = self.text_encoder.batch_size

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
