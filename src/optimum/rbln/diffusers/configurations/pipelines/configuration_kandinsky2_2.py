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
from ....transformers import RBLNCLIPTextModelWithProjectionConfig, RBLNCLIPVisionModelWithProjectionConfig
from ....utils.logging import get_logger
from ..models import RBLNUNet2DConditionModelConfig, RBLNVQModelConfig
from ..models.configuration_prior_transformer import RBLNPriorTransformerConfig


logger = get_logger(__name__)


class _RBLNKandinskyV22PipelineBaseConfig(RBLNModelConfig):
    submodules = ["unet", "movq"]

    def __init__(
        self,
        unet: Optional[RBLNUNet2DConditionModelConfig] = None,
        movq: Optional[RBLNVQModelConfig] = None,
        *,
        sample_size: Optional[Tuple[int, int]] = None,
        batch_size: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        image_size: Optional[Tuple[int, int]] = None,
        img_height: Optional[int] = None,
        img_width: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if image_size is not None and (img_height is not None or img_width is not None):
            raise ValueError("image_size and img_height/img_width cannot both be provided")

        if img_height is not None and img_width is not None:
            image_size = (img_height, img_width)

        self.unet = self.init_submodule_config(
            RBLNUNet2DConditionModelConfig, unet, batch_size=batch_size, sample_size=sample_size
        )
        self.movq = self.init_submodule_config(
            RBLNVQModelConfig,
            movq,
            batch_size=batch_size,
            sample_size=image_size,  # image size is equal to sample size in vae
        )

        if guidance_scale is not None:
            logger.warning("Specifying `guidance_scale` is deprecated. It will be removed in a future version.")
            do_classifier_free_guidance = guidance_scale > 1.0
            if do_classifier_free_guidance:
                self.unet.batch_size = self.movq.batch_size * 2

    @property
    def batch_size(self):
        return self.movq.batch_size

    @property
    def image_size(self):
        return self.movq.sample_size


class RBLNKandinskyV22PipelineConfig(_RBLNKandinskyV22PipelineBaseConfig):
    pass


class RBLNKandinskyV22Img2ImgPipelineConfig(_RBLNKandinskyV22PipelineBaseConfig):
    pass


class RBLNKandinskyV22InpaintPipelineConfig(_RBLNKandinskyV22PipelineBaseConfig):
    pass


class RBLNKandinskyV22PriorPipelineConfig(RBLNModelConfig):
    submodules = ["text_encoder", "image_encoder", "prior"]

    def __init__(
        self,
        text_encoder: Optional[RBLNCLIPTextModelWithProjectionConfig] = None,
        image_encoder: Optional[RBLNCLIPVisionModelWithProjectionConfig] = None,
        prior: Optional[RBLNPriorTransformerConfig] = None,
        *,
        batch_size: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.text_encoder = self.init_submodule_config(
            RBLNCLIPTextModelWithProjectionConfig, text_encoder, batch_size=batch_size
        )
        self.image_encoder = self.init_submodule_config(
            RBLNCLIPVisionModelWithProjectionConfig, image_encoder, batch_size=batch_size
        )

        self.prior = self.init_submodule_config(RBLNPriorTransformerConfig, prior, batch_size=batch_size)

        if guidance_scale is not None:
            logger.warning("Specifying `guidance_scale` is deprecated. It will be removed in a future version.")
            do_classifier_free_guidance = guidance_scale > 1.0
            if do_classifier_free_guidance:
                self.prior.batch_size = self.text_encoder.batch_size * 2

    @property
    def batch_size(self):
        return self.text_encoder.batch_size

    @property
    def image_size(self):
        return self.image_encoder.image_size


class _RBLNKandinskyV22CombinedPipelineBaseConfig(RBLNModelConfig):
    submodules = ["prior_pipe", "decoder_pipe"]

    def __init__(
        self,
        prior_pipe: Optional[RBLNKandinskyV22PriorPipelineConfig] = None,
        decoder_pipe: Optional[RBLNKandinskyV22PipelineConfig] = None,
        *,
        sample_size: Optional[Tuple[int, int]] = None,
        image_size: Optional[Tuple[int, int]] = None,
        batch_size: Optional[int] = None,
        img_height: Optional[int] = None,
        img_width: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        prior_prior: Optional[RBLNPriorTransformerConfig] = None,
        prior_image_encoder: Optional[RBLNCLIPVisionModelWithProjectionConfig] = None,
        prior_text_encoder: Optional[RBLNCLIPTextModelWithProjectionConfig] = None,
        unet: Optional[RBLNUNet2DConditionModelConfig] = None,
        movq: Optional[RBLNVQModelConfig] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.prior_pipe = self.init_submodule_config(
            RBLNKandinskyV22PriorPipelineConfig,
            prior_pipe,
            prior=prior_prior,
            image_encoder=prior_image_encoder,
            text_encoder=prior_text_encoder,
            batch_size=batch_size,
            guidance_scale=guidance_scale,
        )
        self.decoder_pipe = self.init_submodule_config(
            RBLNKandinskyV22PipelineConfig,
            decoder_pipe,
            unet=unet,
            movq=movq,
            batch_size=batch_size,
            sample_size=sample_size,
            image_size=image_size,
            img_height=img_height,
            img_width=img_width,
            guidance_scale=guidance_scale,
        )

    @property
    def batch_size(self):
        return self.prior_pipe.batch_size

    @property
    def image_size(self):
        return self.prior_pipe.image_size

    @property
    def prior_prior(self):
        return self.prior_pipe.prior

    @property
    def prior_image_encoder(self):
        return self.prior_pipe.image_encoder

    @property
    def prior_text_encoder(self):
        return self.prior_pipe.text_encoder

    @property
    def unet(self):
        return self.decoder_pipe.unet

    @property
    def movq(self):
        return self.decoder_pipe.movq


class RBLNKandinskyV22CombinedPipelineConfig(_RBLNKandinskyV22CombinedPipelineBaseConfig):
    pass


class RBLNKandinskyV22InpaintCombinedPipelineConfig(_RBLNKandinskyV22CombinedPipelineBaseConfig):
    pass


class RBLNKandinskyV22Img2ImgCombinedPipelineConfig(_RBLNKandinskyV22CombinedPipelineBaseConfig):
    pass
