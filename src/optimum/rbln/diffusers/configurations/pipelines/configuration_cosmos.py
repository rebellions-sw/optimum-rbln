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
from ....transformers import RBLNT5EncoderModelConfig
from ....utils.logging import get_logger
from ..models import RBLNAutoencoderKLCosmosConfig, RBLNCosmosTransformer3DModelConfig


logger = get_logger(__name__)


class RBLNCosmosPipelineConfig(RBLNModelConfig):
    submodules = ["text_encoder", "transformer", "vae"]
    _vae_uses_encoder = False

    def __init__(
        self,
        text_encoder: Optional[RBLNT5EncoderModelConfig] = None,
        transformer: Optional[RBLNCosmosTransformer3DModelConfig] = None,
        vae: Optional[RBLNAutoencoderKLCosmosConfig] = None,
        *,
        batch_size: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        **kwargs,
    ):
        """
        Args:
            text_encoder (Optional[RBLNT5EncoderModelConfig]): Configuration for the text encoder component.
                Initialized as RBLNT5EncoderModelConfig if not provided.
            transformer (Optional[RBLNCosmosTransformer3DModelConfig]): Configuration for the UNet model component.
                Initialized as RBLNCosmosTransformer3DModelConfig if not provided.
            vae (Optional[RBLNAutoencoderKLCosmosConfig]): Configuration for the VAE model component.
                Initialized as RBLNAutoencoderKLCosmosConfig if not provided.
            batch_size (Optional[int]): Batch size for inference, applied to all submodules.
            img_height (Optional[int]): Height of the generated images.
            img_width (Optional[int]): Width of the generated images.
            sample_size (Optional[Tuple[int, int]]): Spatial dimensions for the UNet model.
            image_size (Optional[Tuple[int, int]]): Alternative way to specify image dimensions.
                Cannot be used together with img_height/img_width.
            guidance_scale (Optional[float]): Scale for classifier-free guidance. Deprecated parameter.
            **kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If both image_size and img_height/img_width are provided.

        Note:
            When guidance_scale > 1.0, the UNet batch size is automatically doubled to
            accommodate classifier-free guidance.
        """
        super().__init__(**kwargs)

        self.text_encoder = self.init_submodule_config(
            RBLNT5EncoderModelConfig, text_encoder, batch_size=batch_size, max_seq_len=max_seq_len
        )
        self.transformer = self.init_submodule_config(
            RBLNCosmosTransformer3DModelConfig,
            transformer,
            batch_size=batch_size,
            max_sequence_length=max_seq_len,
        )
        self.vae = self.init_submodule_config(
            RBLNAutoencoderKLCosmosConfig,
            vae,
            batch_size=batch_size,
            uses_encoder=False,
        )


    @property
    def batch_size(self):
        return self.vae.batch_size

    @property
    def max_seq_len(self):
        return self.text_encoder.max_seq_len
