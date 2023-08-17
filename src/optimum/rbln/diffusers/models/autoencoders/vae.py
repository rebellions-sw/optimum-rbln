# Copyright 2024 Rebellions Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Portions of this software are licensed under the Apache License,
# Version 2.0. See the NOTICE file distributed with this work for
# additional information regarding copyright ownership.

# All other portions of this software, including proprietary code,
# are the intellectual property of Rebellions Inc. and may not be
# copied, modified, or distributed without prior written permission
# from Rebellions Inc.

import logging
from typing import TYPE_CHECKING

import torch  # noqa: I001
from diffusers import AutoencoderKL
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.models.modeling_outputs import AutoencoderKLOutput

from ....utils.runtime_utils import RBLNPytorchRuntime


if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


class RBLNRuntimeVAEEncoder(RBLNPytorchRuntime):
    def encode(self, x: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        moments = self.forward(x.contiguous())
        posterior = DiagonalGaussianDistribution(moments)
        return AutoencoderKLOutput(latent_dist=posterior)


class RBLNRuntimeVAEDecoder(RBLNPytorchRuntime):
    def decode(self, z: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        return (self.forward(z),)


class _VAEDecoder(torch.nn.Module):
    def __init__(self, vae: "AutoencoderKL"):
        super().__init__()
        self.vae = vae

    def forward(self, z):
        vae_out = self.vae.decode(z, return_dict=False)
        return vae_out


class _VAEEncoder(torch.nn.Module):
    def __init__(self, vae: "AutoencoderKL"):
        super().__init__()
        self.vae = vae

    def encode(self, x: torch.FloatTensor, return_dict: bool = True):
        if self.use_tiling and (x.shape[-1] > self.tile_sample_min_size or x.shape[-2] > self.tile_sample_min_size):
            return self.tiled_encode(x, return_dict=return_dict)

        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self.encoder(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = self.encoder(x)
            if self.quant_conv is not None:
                h = self.quant_conv(h)
        return h

    def forward(self, x):
        vae_out = _VAEEncoder.encode(self.vae, x, return_dict=False)
        return vae_out
