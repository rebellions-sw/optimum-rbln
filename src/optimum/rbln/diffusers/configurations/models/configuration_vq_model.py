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


class RBLNVQModelConfig(RBLNModelConfig):
    def __init__(
        self,
        batch_size: Optional[int] = None,
        sample_size: Optional[Tuple[int, int]] = None,
        vqmodel_scale_factor: Optional[float] = None,  # TODO: rename to scaling_factor
        in_channels: Optional[int] = None,
        latent_channels: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.batch_size = batch_size or 1
        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")

        self.sample_size = sample_size
        if isinstance(self.sample_size, int):
            self.sample_size = (self.sample_size, self.sample_size)

        self.vqmodel_scale_factor = vqmodel_scale_factor
        self.in_channels = in_channels
        self.latent_channels = latent_channels

    @property
    def image_size(self):
        return self.sample_size

    @property
    def latent_sample_size(self):
        return (self.image_size[0] // self.vqmodel_scale_factor, self.image_size[1] // self.vqmodel_scale_factor)
