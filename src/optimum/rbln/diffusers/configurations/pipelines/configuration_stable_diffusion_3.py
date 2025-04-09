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
from ....transformers import RBLNCLIPTextModelWithProjectionConfig, RBLNT5EncoderModelConfig
from ....utils.logging import get_logger
from ..models import RBLNAutoencoderKLConfig, RBLNSD3Transformer2DModelConfig


logger = get_logger(__name__)


class _RBLNStableDiffusion3PipelineBaseConfig(RBLNModelConfig):
    submodules = ["transformer", "text_encoder", "text_encoder_2", "text_encoder_3", "vae"]
    _vae_uses_encoder = False

    def __init__(
        self,
        transformer: Optional[RBLNModelConfig] = None,
        text_encoder: Optional[RBLNModelConfig] = None,
        text_encoder_2: Optional[RBLNModelConfig] = None,
        text_encoder_3: Optional[RBLNModelConfig] = None,
        vae: Optional[RBLNModelConfig] = None,
        *,
        max_seq_len: Optional[int] = None,
        sample_size: Optional[Tuple[int, int]] = None,
        image_size: Optional[Tuple[int, int]] = None,
        batch_size: Optional[int] = None,
        img_height: Optional[int] = None,
        img_width: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if image_size is not None and (img_height is not None or img_width is not None):
            raise ValueError("image_size and img_height/img_width cannot both be provided")

        if img_height is not None and img_width is not None:
            image_size = (img_height, img_width)

        max_seq_len = max_seq_len or 256

        self.text_encoder = self.init_submodule_config(
            RBLNCLIPTextModelWithProjectionConfig, text_encoder, batch_size=batch_size
        )
        self.text_encoder_2 = self.init_submodule_config(
            RBLNCLIPTextModelWithProjectionConfig, text_encoder_2, batch_size=batch_size
        )
        self.text_encoder_3 = self.init_submodule_config(
            RBLNT5EncoderModelConfig,
            text_encoder_3,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
        )
        self.transformer = self.init_submodule_config(
            RBLNSD3Transformer2DModelConfig,
            transformer,
            batch_size=batch_size,
            sample_size=sample_size,
        )
        self.vae = self.init_submodule_config(
            RBLNAutoencoderKLConfig,
            vae,
            batch_size=batch_size,
            uses_encoder=self.__class__._vae_uses_encoder,
            sample_size=image_size,
        )

        if guidance_scale is not None:
            logger.warning("Specifying `guidance_scale` is deprecated. It will be removed in a future version.")
            do_classifier_free_guidance = guidance_scale > 1.0
            if do_classifier_free_guidance:
                self.transformer.batch_size = batch_size * 2

    @property
    def max_seq_len(self):
        return self.text_encoder_3.max_seq_len

    @property
    def batch_size(self):
        return self.vae.batch_size

    @property
    def sample_size(self):
        return self.transformer.sample_size

    @property
    def image_size(self):
        return self.vae.sample_size


class RBLNStableDiffusion3PipelineConfig(_RBLNStableDiffusion3PipelineBaseConfig):
    _vae_uses_encoder = False


class RBLNStableDiffusion3Img2ImgPipelineConfig(_RBLNStableDiffusion3PipelineBaseConfig):
    _vae_uses_encoder = True


class RBLNStableDiffusion3InpaintPipelineConfig(_RBLNStableDiffusion3PipelineBaseConfig):
    _vae_uses_encoder = True
