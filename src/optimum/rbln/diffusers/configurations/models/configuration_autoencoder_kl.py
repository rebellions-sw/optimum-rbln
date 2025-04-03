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


class RBLNAutoencoderKLConfig(RBLNModelConfig):
    def __init__(
        self,
        batch_size: Optional[int] = None,
        sample_size: Optional[Tuple[int, int]] = None,
        uses_encoder: Optional[bool] = None,
        vae_scale_factor: Optional[float] = None,
        in_channels: Optional[int] = None,
        latent_channels: Optional[int] = None,
        *,
        image_size: Optional[Tuple[int, int]] = None,
        img_height: Optional[int] = None,
        img_width: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.batch_size = batch_size or 1
        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")

        self.uses_encoder = uses_encoder
        self.vae_scale_factor = vae_scale_factor
        self.in_channels = in_channels
        self.latent_channels = latent_channels

        if image_size is not None and (img_height is not None or img_width is not None):
            raise ValueError("image_size and img_height/img_width cannot both be provided")

        if img_height is not None and img_width is not None:
            self.sample_size = (img_height, img_width)
        else:
            self.sample_size = sample_size

        if self.sample_size is not None and image_size is not None:
            raise ValueError("image_size and sample_size cannot both be provided")

        if self.sample_size is None and image_size is not None:
            self.sample_size = image_size

        if isinstance(self.sample_size, int):
            self.sample_size = (self.sample_size, self.sample_size)

    @property
    def image_size(self):
        return self.sample_size

    @property
    def latent_sample_size(self):
        return (self.image_size[0] // self.vae_scale_factor, self.image_size[1] // self.vae_scale_factor)
