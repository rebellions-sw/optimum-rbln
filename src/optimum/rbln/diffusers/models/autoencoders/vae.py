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

from typing import TYPE_CHECKING, List, Union, Dict, Optional, Tuple

import torch  # noqa: I001
from diffusers import AutoencoderKL, AutoencoderKLCogVideoX, VQModel
from diffusers.models.autoencoders.vae import DecoderOutput, DiagonalGaussianDistribution
from diffusers.models.autoencoders.vq_model import VQEncoderOutput
from diffusers.models.modeling_outputs import AutoencoderKLOutput

from ....ops import register_rbln_custom_cache_update
from ....utils.logging import get_logger
from ....utils.runtime_utils import RBLNPytorchRuntime


if TYPE_CHECKING:
    import torch

logger = get_logger(__name__)


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


class RBLNRuntimeVQEncoder(RBLNPytorchRuntime):
    def encode(self, x: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        h = self.forward(x.contiguous())
        return VQEncoderOutput(latents=h)


class RBLNRuntimeVQDecoder(RBLNPytorchRuntime):
    def decode(self, h: torch.Tensor, force_not_quantize: bool = False, shape=None, **kwargs) -> List[torch.Tensor]:
        if not (force_not_quantize and not self.lookup_from_codebook):
            raise ValueError(
                "Currently, the `decode` method of the class `RBLNVQModel` is executed successfully only if `force_not_quantize` is True and `config.lookup_from_codebook` is False"
            )
        commit_loss = torch.zeros((h.shape[0])).to(h.device, dtype=h.dtype)
        dec = self.forward(h.contiguous())
        return dec, commit_loss


class _VQEncoder(torch.nn.Module):
    def __init__(self, vq_model: VQModel):
        super().__init__()
        self.vq_model = vq_model

    def encode(self, x: torch.Tensor, return_dict: bool = True):
        h = self.vq_model.encoder(x)
        h = self.vq_model.quant_conv(h)
        return h

    def forward(self, x: torch.Tensor):
        vq_out = self.encode(x)
        return vq_out


class _VQDecoder(torch.nn.Module):
    def __init__(self, vq_model: VQModel):
        super().__init__()
        self.vq_model = vq_model

    def decode(self, h: torch.Tensor, force_not_quantize: bool = False, return_dict: bool = True, shape=None):
        quant = h
        quant2 = self.vq_model.post_quant_conv(quant)
        quant = quant if self.vq_model.config.norm_type == "spatial" else None
        dec = self.vq_model.decoder(quant2, quant)
        return dec

    def forward(self, h: torch.Tensor):
        vq_out = self.decode(h)
        return vq_out


class RBLNRuntimeVAECogVideoXEncoder(RBLNPytorchRuntime):
    def encode(self, x: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        h = self.forward(x.contiguous())
        posterior = DiagonalGaussianDistribution(h)
        return AutoencoderKLOutput(latent_dist=posterior)


class RBLNRuntimeVAECogVideoXDecoder(RBLNPytorchRuntime):
    pass

class _VAECogVideoXEncoder(torch.nn.Module):
    def __init__(self, cog_video_x: AutoencoderKLCogVideoX):
        super().__init__()
        self.cog_video_x = cog_video_x

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = x.shape

        if self.use_tiling and (width > self.tile_sample_min_width or height > self.tile_sample_min_height):
            return self.tiled_encode(x)

        frame_batch_size = self.num_sample_frames_batch_size
        # Note: We expect the number of frames to be either `1` or `frame_batch_size * k` or `frame_batch_size * k + 1` for some k.
        # As the extra single frame is handled inside the loop, it is not required to round up here.
        num_batches = max(num_frames // frame_batch_size, 1)
        conv_cache = None
        enc = []

        for i in range(num_batches):
            remaining_frames = num_frames % frame_batch_size
            start_frame = frame_batch_size * i + (0 if i == 0 else remaining_frames)
            end_frame = frame_batch_size * (i + 1) + remaining_frames
            x_intermediate = x[:, :, start_frame:end_frame]
            x_intermediate, conv_cache = self.encoder(x_intermediate, conv_cache=conv_cache)
            if self.quant_conv is not None:
                x_intermediate = self.quant_conv(x_intermediate)
            enc.append(x_intermediate)

        enc = torch.cat(enc, dim=2)

        return enc

    def encode(self, x: torch.Tensor, return_dict: bool = True):
        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self._encode(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = self._encode(x)

        return h

    def forward(self, x: torch.Tensor):
        cov_video_enc_out = self.encode(x, return_dict=False)
        return cov_video_enc_out


from diffusers.models.autoencoders.autoencoder_kl_cogvideox import CogVideoXCausalConv3d
class RBLNCogVideoXCausalConv3dEnc(CogVideoXCausalConv3d):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

    def fake_context_parallel_forward(
        self, inputs: torch.Tensor, conv_cache: torch.Tensor = None
    ) -> torch.Tensor:
        kernel_size = self.time_kernel_size
        if kernel_size > 1:
            # inputs.shape = torch.Size([1, 3, F, 480, 720]) B C F H W
            cached_inputs = [inputs[:, :, :1]] * (kernel_size - 1)
            inputs = torch.cat(cached_inputs + [inputs], dim=2) # frame에 padding
        return inputs

    def forward(self, inputs: torch.Tensor, conv_cache: Optional[torch.Tensor] = None) -> torch.Tensor:
        inputs = self.fake_context_parallel_forward(inputs, conv_cache) # frame padded inputs # (1, 3, 3, 480, 720)
        # -1*(frame kernel size - 1) -> -2,-1
        conv_cache = inputs[:, :, -self.time_kernel_size + 1 :].clone() # TODO: convert to RBLN custom ops

        padding_2d = (self.width_pad, self.width_pad, self.height_pad, self.height_pad) 
        inputs = torch.nn.functional.pad(inputs, padding_2d, mode="constant", value=0) # (1, 3, 3, 482, 722)

        output = self.conv(inputs)
        return output, conv_cache

class RBLNCogVideoXCausalConv3dDec(CogVideoXCausalConv3d):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

    def fake_context_parallel_forward(
        self, inputs: torch.Tensor, conv_cache: torch.Tensor
    ) -> torch.Tensor:
        kernel_size = self.time_kernel_size
        if kernel_size > 1:
            cached_inputs = [conv_cache] # TODO: convert to RBLN custom ops
            inputs = torch.cat(cached_inputs + [inputs], dim=2)
        return inputs

    def forward(self, inputs: torch.Tensor, conv_cache: torch.Tensor) -> torch.Tensor:
        inputs = self.fake_context_parallel_forward(inputs, conv_cache) # frame padded inputs # (1, 3, 3, 480, 720)
        # -1*(frame kernel size - 1) -> -2,-1
        conv_cache = inputs[:, :, -self.time_kernel_size + 1 :].clone() # TODO: convert to RBLN custom ops

        padding_2d = (self.width_pad, self.width_pad, self.height_pad, self.height_pad) 
        inputs = torch.nn.functional.pad(inputs, padding_2d, mode="constant", value=0) # (1, 3, 3, 482, 722)

        output = self.conv(inputs)
        return output, conv_cache
    
class _VAECogVideoXDecoder(torch.nn.Module):
    def __init__(self, cog_video_x: AutoencoderKLCogVideoX):
        super().__init__()
        register_rbln_custom_cache_update()
        self.cog_video_x = cog_video_x
        # self.keys = None
        self.keys = self.get_conv_cache_key()

    def _to_tuple(self, conv_cache):
        conv_cache_list = []
        keys = []
        def isdict(obj, names):
            if isinstance(obj, dict):
                for _k, _v in obj.items():
                    isdict(_v, names+f"_{_k}")
            else :
                if "norm" not in names:
                    conv_cache_list.append(obj)
                    keys.append(names)
                # if names in set(self.keys):
                #     conv_cache_list.append(obj)
        
        for k, v in conv_cache.items():
            isdict(v, k)
            
        return tuple(conv_cache_list), keys

    def _to_nested_dict(self, conv_cache_list, keys):
        conv_cache_dict = {}
        for k, v in zip(keys, conv_cache_list):
            parts = k.split(".")
            current = conv_cache_dict
            
            # 마지막 부분을 제외한 모든 부분에 대해 중첩 딕셔너리 생성
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            # 마지막 부분에 값 할당
            current[parts[-1]] = v
        return conv_cache_dict

    def get_conv_cache_key(self):
        from .input_dict import conv_cache as CONV_CACHE
        keys = []
        for k in CONV_CACHE.keys():
            if "norm" not in k:
                keys.append(k)
        return keys
        
    def forward(self, z: torch.Tensor, *args: Optional[Tuple[torch.Tensor]]):
        enc = z.shape[2] > 2 # FIXME make generalized condition
        dummy_outs=[]
        if enc :
            cov_video_dec_out, conv_cache = self.cog_video_x.decoder(z)
            conv_cache_list, _ = self._to_tuple(conv_cache)
            
            for cache, conv_cache in zip(args, conv_cache_list):
                batch_dim = torch.tensor(0, dtype=torch.int16)
                batch_axis = torch.tensor(0, dtype=torch.int16)
                
                # import pdb; pdb.set_trace()
                dummy_out = torch.ops.rbln_custom_ops.rbln_cache_update(cache, conv_cache, batch_dim, batch_axis)
                dummy_outs.append(dummy_out)
            # import pdb; pdb.set_trace()
        else :
            conv_cache_dict = self._to_nested_dict(args, self.keys)
            cov_video_dec_out, conv_cache = self.cog_video_x.decoder(z, conv_cache=conv_cache_dict)
            conv_cache_list, _ = self._to_tuple(conv_cache)
            
            for cache, conv_cache in zip(args, conv_cache_list):
                batch_dim = torch.tensor(0, dtype=torch.int16)
                batch_axis = torch.tensor(0, dtype=torch.int16)
                
                dummy_out = torch.ops.rbln_custom_ops.rbln_cache_update(cache, conv_cache, batch_dim, batch_axis)
                dummy_outs.append(dummy_out)
        
        # import pdb; pdb.set_trace()
        # return cov_video_dec_out, conv_cache_list, tuple(dummy_outs)
        return cov_video_dec_out, torch.stack(tuple(dummy_outs))
