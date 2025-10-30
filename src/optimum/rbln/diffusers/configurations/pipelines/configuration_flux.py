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
from ....transformers import RBLNCLIPTextModelConfig, RBLNT5EncoderModelConfig
from ..models import RBLNAutoencoderKLConfig, RBLNFluxTransformer2DModelConfig


class _RBLNFluxPipelineBaseConfig(RBLNModelConfig):
    submodules = ["text_encoder", "text_encoder_2", "transformer", "vae"]
    _vae_uses_encoder = False

    def __init__(
        self,
        text_encoder: Optional[RBLNCLIPTextModelConfig] = None,
        text_encoder_2: Optional[RBLNT5EncoderModelConfig] = None,
        transformer: Optional[RBLNFluxTransformer2DModelConfig] = None,
        vae: Optional[RBLNAutoencoderKLConfig] = None,
        *,
        batch_size: Optional[int] = None,
        img_height: Optional[int] = None,
        img_width: Optional[int] = None,
        sample_size: Optional[Tuple[int, int]] = None,
        image_size: Optional[Tuple[int, int]] = None,
        guidance_scale: Optional[float] = None,
        max_sequence_length: Optional[int] = 512,
        **kwargs,
    ):
        """
        Args:
            text_encoder (Optional[RBLNCLIPTextModelConfig]): Configuration for the text encoder component.
                Initialized as RBLNCLIPTextModelConfig if not provided.
            text_encoder_2 (Optional[RBLNT5EncoderModelConfig]): Configuration for the text encoder 2 component.
                Initialized as RBLNT5EncoderModelConfig if not provided.
            transformer (Optional[RBLNFluxTransformer2DModelConfig]): Configuration for the flux transformer model component.
                Initialized as RBLNFluxTransformer2DModelConfig if not provided.
            vae (Optional[RBLNAutoencoderKLConfig]): Configuration for the VAE model component.
                Initialized as RBLNAutoencoderKLConfig if not provided.
            batch_size (Optional[int]): Batch size for inference, applied to all submodules.
            img_height (Optional[int]): Height of the generated images.
            img_width (Optional[int]): Width of the generated images.
            sample_size (Optional[Tuple[int, int]]): Spatial dimensions for the UNet model.
            image_size (Optional[Tuple[int, int]]): Alternative way to specify image dimensions.
                Cannot be used together with img_height/img_width.
            guidance_scale (Optional[float]): Scale for classifier-free guidance.
            max_sequence_length (Optional[int], default to 512): Maximum sequence length to use with the `prompt`
            **kwargs: Additional arguments passed to the parent RBLNModelConfig.
        Raises:
            ValueError: If both image_size and img_height/img_width are provided.
        """
        super().__init__(**kwargs)
        if image_size is not None and (img_height is not None or img_width is not None):
            raise ValueError("image_size and img_height/img_width cannot both be provided")

        if img_height is not None and img_width is not None:
            image_size = (img_height, img_width)

        self.text_encoder = self.initialize_submodule_config(
            text_encoder,
            cls_name="RBLNCLIPTextModelConfig",
            batch_size=batch_size,
        )
        self.text_encoder_2 = self.initialize_submodule_config(
            text_encoder_2,
            cls_name="RBLNT5EncoderModelConfig",
            batch_size=batch_size,
            max_seq_len=max_sequence_length,
        )
        self.transformer = self.initialize_submodule_config(
            transformer,
            cls_name="RBLNFluxTransformer2DModelConfig",
            sample_size=sample_size,
            max_sequence_length=max_sequence_length,
        )
        self.vae = self.initialize_submodule_config(
            vae,
            cls_name="RBLNAutoencoderKLConfig",
            batch_size=batch_size,
            uses_encoder=self.__class__._vae_uses_encoder,
            sample_size=image_size,
        )

        if guidance_scale is None:
            guidance_scale = self.get_default_values_for_original_cls("__call__", ["guidance_scale"])["guidance_scale"]

    @property
    def batch_size(self):
        return self.vae.batch_size

    @property
    def sample_size(self):
        return self.transformer.sample_size

    @property
    def image_size(self):
        return self.vae.sample_size

    @property
    def max_seq_len(self):
        return self.text_encoder_2.max_seq_len


class RBLNFluxPipelineConfig(_RBLNFluxPipelineBaseConfig):
    _vae_uses_encoder = False
