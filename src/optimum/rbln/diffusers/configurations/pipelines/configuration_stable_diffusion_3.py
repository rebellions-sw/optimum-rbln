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

from typing import Any, Optional, Tuple

from ....configuration_utils import RBLNModelConfig
from ....transformers import RBLNCLIPTextModelWithProjectionConfig, RBLNT5EncoderModelConfig
from ..models import RBLNAutoencoderKLConfig, RBLNSD3Transformer2DModelConfig


class RBLNStableDiffusion3PipelineBaseConfig(RBLNModelConfig):
    submodules = ["transformer", "text_encoder", "text_encoder_2", "text_encoder_3", "vae"]
    _vae_uses_encoder = False

    def __init__(
        self,
        transformer: Optional[RBLNSD3Transformer2DModelConfig] = None,
        text_encoder: Optional[RBLNCLIPTextModelWithProjectionConfig] = None,
        text_encoder_2: Optional[RBLNCLIPTextModelWithProjectionConfig] = None,
        text_encoder_3: Optional[RBLNT5EncoderModelConfig] = None,
        vae: Optional[RBLNAutoencoderKLConfig] = None,
        *,
        max_seq_len: Optional[int] = None,
        sample_size: Optional[Tuple[int, int]] = None,
        image_size: Optional[Tuple[int, int]] = None,
        batch_size: Optional[int] = None,
        img_height: Optional[int] = None,
        img_width: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        **kwargs: Any,
    ):
        """
        Args:
            transformer (Optional[RBLNSD3Transformer2DModelConfig]): Configuration for the transformer model component.
                Initialized as RBLNSD3Transformer2DModelConfig if not provided.
            text_encoder (Optional[RBLNCLIPTextModelWithProjectionConfig]): Configuration for the primary text encoder.
                Initialized as RBLNCLIPTextModelWithProjectionConfig if not provided.
            text_encoder_2 (Optional[RBLNCLIPTextModelWithProjectionConfig]): Configuration for the secondary text encoder.
                Initialized as RBLNCLIPTextModelWithProjectionConfig if not provided.
            text_encoder_3 (Optional[RBLNT5EncoderModelConfig]): Configuration for the tertiary text encoder.
                Initialized as RBLNT5EncoderModelConfig if not provided.
            vae (Optional[RBLNAutoencoderKLConfig]): Configuration for the VAE model component.
                Initialized as RBLNAutoencoderKLConfig if not provided.
            max_seq_len (Optional[int]): Maximum sequence length for text inputs. Defaults to 256.
            sample_size (Optional[Tuple[int, int]]): Spatial dimensions for the transformer model.
            image_size (Optional[Tuple[int, int]]): Dimensions for the generated images.
                Cannot be used together with img_height/img_width.
            batch_size (Optional[int]): Batch size for inference, applied to all submodules.
            img_height (Optional[int]): Height of the generated images.
            img_width (Optional[int]): Width of the generated images.
            height (Optional[int]): Height of the generated images.
            width (Optional[int]): Width of the generated images.
            guidance_scale (Optional[float]): Scale for classifier-free guidance.
            **kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If both image_size and img_height/img_width are provided.

        Note:
            When guidance_scale > 1.0, the transformer batch size is automatically doubled to
            accommodate classifier-free guidance.
        """
        super().__init__(**kwargs)

        # Initial check for image_size conflict remains as is
        if image_size is not None and (
            img_height is not None or img_width is not None or height is not None or width is not None
        ):
            raise ValueError("image_size cannot be provided alongside img_height/img_width or height/width")

        # Prioritize height/width (HF-aligned)
        if height is not None and width is not None:
            if img_height is not None or img_width is not None:
                # Raise error if both sets of arguments are provided
                raise ValueError(
                    "Cannot provide both 'height'/'width' and 'img_height'/'img_width' simultaneously. "
                    "Please use one set of arguments for image dimensions, preferring 'height'/'width'."
                )
            image_size = (height, width)
        elif (height is not None and width is None) or (height is None and width is not None):
            raise ValueError("Both height and width must be provided together if used")
        # Fallback to img_height/img_width for backward compatibility
        elif img_height is not None and img_width is not None:
            image_size = (img_height, img_width)
        elif (img_height is not None and img_width is None) or (img_height is None and img_width is not None):
            raise ValueError("Both img_height and img_width must be provided together if used")

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
            model_input_names=["input_ids"],
        )
        self.transformer = self.init_submodule_config(
            RBLNSD3Transformer2DModelConfig,
            transformer,
            sample_size=sample_size,
        )
        self.vae = self.init_submodule_config(
            RBLNAutoencoderKLConfig,
            vae,
            batch_size=batch_size,
            uses_encoder=self.__class__._vae_uses_encoder,
            sample_size=image_size,
        )

        # Get default guidance scale from original class to set Transformer batch size
        if guidance_scale is None:
            guidance_scale = self.get_default_values_for_original_cls("__call__", ["guidance_scale"])["guidance_scale"]

        if not self.transformer.batch_size_is_specified:
            do_classifier_free_guidance = guidance_scale > 1.0
            if do_classifier_free_guidance:
                self.transformer.batch_size = self.text_encoder.batch_size * 2
            else:
                self.transformer.batch_size = self.text_encoder.batch_size

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


class RBLNStableDiffusion3PipelineConfig(RBLNStableDiffusion3PipelineBaseConfig):
    """Config for SD3 Text2Img Pipeline"""

    _vae_uses_encoder = False


class RBLNStableDiffusion3Img2ImgPipelineConfig(RBLNStableDiffusion3PipelineBaseConfig):
    """Config for SD3 Img2Img Pipeline"""

    _vae_uses_encoder = True


class RBLNStableDiffusion3InpaintPipelineConfig(RBLNStableDiffusion3PipelineBaseConfig):
    """Config for SD3 Inpainting Pipeline"""

    _vae_uses_encoder = True
