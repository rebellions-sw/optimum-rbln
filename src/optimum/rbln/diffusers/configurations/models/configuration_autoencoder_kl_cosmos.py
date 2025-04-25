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
from ....utils.logging import get_logger


logger = get_logger(__name__)


class RBLNAutoencoderKLCosmosConfig(RBLNModelConfig):
    def __init__(
        self,
        batch_size: Optional[int] = None,
        uses_encoder: Optional[bool] = None,
        num_frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_channel_latents: Optional[int] = None,
        num_latent_frames: Optional[int] = None,
        latent_height: Optional[int] = None,
        latent_width: Optional[int] = None,
        in_channels: Optional[int] = None,
        **kwargs,
    ):
        """
        Args:
            batch_size (Optional[int]): The batch size for inference. Defaults to 1.
            uses_encoder (Optional[bool]): Whether to include the encoder part of the VAE in the model.
                When False, only the decoder is used (for latent-to-image conversion).
            **kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If batch_size is not a positive integer.
        """
        super().__init__(**kwargs)
        # Since the Cosmos VAE Decoder already requires approximately 7.9 GiB of memory,
        # Optimum-rbln cannot execute this model on RBLN-CA12 when the batch size > 1.
        # However, the Cosmos VAE Decoder propose batch slicing when the batch size is greater than 1,
        # Optimum-rbln utilize this method by compiling with batch_size=1 to enable batch slicing.
        self.batch_size = batch_size or 1
        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")
        elif self.batch_size > 1:
            logger.warning("The batch size of Cosmos VAE Decoder will be explicitly 1 for memory efficiency.")
            self.batch_size = 1

        self.uses_encoder = uses_encoder
        self.num_frames = num_frames or 121
        self.height = height or 704
        self.width = width or 1280

        self.num_channel_latents = num_channel_latents
        self.num_latent_frames = num_latent_frames
        self.latent_height = self.latent_height
        self.latent_width = latent_width
        self.in_channels = in_channels

    @property
    def image_size(self):
        return [self.height, self.width]
