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
from ..models import RBLNUNet2DConditionModelConfig, RBLNVQModelConfig
from ..models.configuration_prior_transformer import RBLNPriorTransformerConfig


class _RBLNKandinskyV22PipelineBaseConfig(RBLNModelConfig):
    submodules = ["unet", "movq"]
    _movq_uses_encoder = False

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
        """
        Args:
            unet (Optional[RBLNUNet2DConditionModelConfig]): Configuration for the UNet model component.
                Initialized as RBLNUNet2DConditionModelConfig if not provided.
            movq (Optional[RBLNVQModelConfig]): Configuration for the MoVQ (VQ-GAN) model component.
                Initialized as RBLNVQModelConfig if not provided.
            sample_size (Optional[Tuple[int, int]]): Spatial dimensions for the UNet model.
            batch_size (Optional[int]): Batch size for inference, applied to all submodules.
            guidance_scale (Optional[float]): Scale for classifier-free guidance.
            image_size (Optional[Tuple[int, int]]): Dimensions for the generated images.
                Cannot be used together with img_height/img_width.
            img_height (Optional[int]): Height of the generated images.
            img_width (Optional[int]): Width of the generated images.
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

        self.unet = self.init_submodule_config(RBLNUNet2DConditionModelConfig, unet, sample_size=sample_size)
        self.movq = self.init_submodule_config(
            RBLNVQModelConfig,
            movq,
            batch_size=batch_size,
            sample_size=image_size,  # image size is equal to sample size in vae
            uses_encoder=self._movq_uses_encoder,
        )

        # Get default guidance scale from original class to set UNet batch size
        if guidance_scale is None:
            guidance_scale = self.get_default_values_for_original_cls("__call__", ["guidance_scale"])["guidance_scale"]

        if not self.unet.batch_size_is_specified:
            do_classifier_free_guidance = guidance_scale > 1.0
            if do_classifier_free_guidance:
                self.unet.batch_size = self.movq.batch_size * 2
            else:
                self.unet.batch_size = self.movq.batch_size

    @property
    def batch_size(self):
        return self.movq.batch_size

    @property
    def image_size(self):
        return self.movq.sample_size


class RBLNKandinskyV22PipelineConfig(_RBLNKandinskyV22PipelineBaseConfig):
    _movq_uses_encoder = False


class RBLNKandinskyV22Img2ImgPipelineConfig(_RBLNKandinskyV22PipelineBaseConfig):
    _movq_uses_encoder = True


class RBLNKandinskyV22InpaintPipelineConfig(_RBLNKandinskyV22PipelineBaseConfig):
    _movq_uses_encoder = True


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
        """
        Initialize a configuration for Kandinsky 2.2 prior pipeline optimized for RBLN NPU.

        This configuration sets up the prior components of the Kandinsky 2.2 architecture, which includes
        text and image encoders along with a prior transformer that maps text/image embeddings to
        latent representations used to condition the diffusion process.

        Args:
            text_encoder (Optional[RBLNCLIPTextModelWithProjectionConfig]): Configuration for the text encoder component.
                Initialized as RBLNCLIPTextModelWithProjectionConfig if not provided.
            image_encoder (Optional[RBLNCLIPVisionModelWithProjectionConfig]): Configuration for the image encoder component.
                Initialized as RBLNCLIPVisionModelWithProjectionConfig if not provided.
            prior (Optional[RBLNPriorTransformerConfig]): Configuration for the prior transformer component.
                Initialized as RBLNPriorTransformerConfig if not provided.
            batch_size (Optional[int]): Batch size for inference, applied to all submodules.
            guidance_scale (Optional[float]): Scale for classifier-free guidance.
            **kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Note:
            When guidance_scale > 1.0, the prior batch size is automatically doubled to
            accommodate classifier-free guidance.
        """
        super().__init__(**kwargs)
        self.text_encoder = self.init_submodule_config(
            RBLNCLIPTextModelWithProjectionConfig, text_encoder, batch_size=batch_size
        )
        self.image_encoder = self.init_submodule_config(
            RBLNCLIPVisionModelWithProjectionConfig, image_encoder, batch_size=batch_size
        )

        self.prior = self.init_submodule_config(RBLNPriorTransformerConfig, prior)

        # Get default guidance scale from original class to set UNet batch size
        if guidance_scale is None:
            guidance_scale = self.get_default_values_for_original_cls("__call__", ["guidance_scale"])["guidance_scale"]

        if not self.prior.batch_size_is_specified:
            do_classifier_free_guidance = guidance_scale > 1.0
            if do_classifier_free_guidance:
                self.prior.batch_size = self.text_encoder.batch_size * 2
            else:
                self.prior.batch_size = self.text_encoder.batch_size

    @property
    def batch_size(self):
        return self.text_encoder.batch_size

    @property
    def image_size(self):
        return self.image_encoder.image_size


class _RBLNKandinskyV22CombinedPipelineBaseConfig(RBLNModelConfig):
    submodules = ["prior_pipe", "decoder_pipe"]
    _decoder_pipe_cls = RBLNKandinskyV22PipelineConfig

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
        """
        Initialize a configuration for combined Kandinsky 2.2 pipelines optimized for RBLN NPU.

        This configuration integrates both the prior and decoder components of Kandinsky 2.2 into
        a unified pipeline, allowing for end-to-end text-to-image generation in a single model.
        It combines the text/image encoding, prior mapping, and diffusion steps together.

        Args:
            prior_pipe (Optional[RBLNKandinskyV22PriorPipelineConfig]): Configuration for the prior pipeline.
                Initialized as RBLNKandinskyV22PriorPipelineConfig if not provided.
            decoder_pipe (Optional[RBLNKandinskyV22PipelineConfig]): Configuration for the decoder pipeline.
                Initialized as RBLNKandinskyV22PipelineConfig if not provided.
            sample_size (Optional[Tuple[int, int]]): Spatial dimensions for the UNet model.
            image_size (Optional[Tuple[int, int]]): Dimensions for the generated images.
                Cannot be used together with img_height/img_width.
            batch_size (Optional[int]): Batch size for inference, applied to all submodules.
            img_height (Optional[int]): Height of the generated images.
            img_width (Optional[int]): Width of the generated images.
            guidance_scale (Optional[float]): Scale for classifier-free guidance.
            prior_prior (Optional[RBLNPriorTransformerConfig]): Direct configuration for the prior transformer.
                Used if prior_pipe is not provided.
            prior_image_encoder (Optional[RBLNCLIPVisionModelWithProjectionConfig]): Direct configuration for the image encoder.
                Used if prior_pipe is not provided.
            prior_text_encoder (Optional[RBLNCLIPTextModelWithProjectionConfig]): Direct configuration for the text encoder.
                Used if prior_pipe is not provided.
            unet (Optional[RBLNUNet2DConditionModelConfig]): Direct configuration for the UNet.
                Used if decoder_pipe is not provided.
            movq (Optional[RBLNVQModelConfig]): Direct configuration for the MoVQ (VQ-GAN) model.
                Used if decoder_pipe is not provided.
            **kwargs: Additional arguments passed to the parent RBLNModelConfig.
        """
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
            self._decoder_pipe_cls,
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
    _decoder_pipe_cls = RBLNKandinskyV22PipelineConfig


class RBLNKandinskyV22InpaintCombinedPipelineConfig(_RBLNKandinskyV22CombinedPipelineBaseConfig):
    _decoder_pipe_cls = RBLNKandinskyV22InpaintPipelineConfig


class RBLNKandinskyV22Img2ImgCombinedPipelineConfig(_RBLNKandinskyV22CombinedPipelineBaseConfig):
    _decoder_pipe_cls = RBLNKandinskyV22Img2ImgPipelineConfig
