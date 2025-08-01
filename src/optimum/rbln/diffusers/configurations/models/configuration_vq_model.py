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


class RBLNVQModelConfig(RBLNModelConfig):
    """
    Configuration class for RBLN VQModel models, used in Kandinsky.

    This class inherits from RBLNModelConfig and provides specific configuration options
    for VQModel, which acts similarly to a VAE but uses vector quantization.
    """

    def __init__(
        self,
        batch_size: Optional[int] = None,
        sample_size: Optional[Tuple[int, int]] = None,
        uses_encoder: Optional[bool] = None,
        vqmodel_scale_factor: Optional[float] = None,  # TODO: rename to scaling_factor
        in_channels: Optional[int] = None,
        latent_channels: Optional[int] = None,
        **kwargs: Any,
    ):
        """
        Args:
            batch_size (Optional[int]): The batch size for inference. Defaults to 1.
            sample_size (Optional[Tuple[int, int]]): The spatial dimensions (height, width) of the input/output images.
                If an integer is provided, it's used for both height and width.
            uses_encoder (Optional[bool]): Whether to include the encoder part of the VAE in the model.
                When False, only the decoder is used (for latent-to-image conversion).
            vqmodel_scale_factor (Optional[float]): The scaling factor between pixel space and latent space.
                Determines the downsampling ratio between original images and latent representations.
            in_channels (Optional[int]): Number of input channels for the model.
            latent_channels (Optional[int]): Number of channels in the latent space.
            **kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If batch_size is not a positive integer.
        """
        super().__init__(**kwargs)
        self.batch_size = batch_size or 1
        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")

        self.uses_encoder = uses_encoder
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
