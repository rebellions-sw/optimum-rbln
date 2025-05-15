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

from diffusers import FluxPipeline

from ...configurations import RBLNFluxPipelineConfig
from ...modeling_diffusers import RBLNDiffusionMixin


class RBLNFluxPipeline(RBLNDiffusionMixin, FluxPipeline):
    original_class = FluxPipeline
    _rbln_config_class = RBLNFluxPipelineConfig
    _submodules = ["transformer"]
    # _submodules = ["text_encoder", "text_encoder_2"] -> verification done
    # _submodules = ["text_encoder_2"]
    # _submodules = ["vae"] -> verification done

    # text_encoder_2 -> t5 encoder model
    # text_encoder -> clip text model

    # for sin, cos into relay graph
    # def __init__(
    #     self,
    #     scheduler: FlowMatchEulerDiscreteScheduler,
    #     vae: AutoencoderKL,
    #     text_encoder: CLIPTextModel,
    #     tokenizer: CLIPTokenizer,
    #     text_encoder_2: T5EncoderModel,
    #     tokenizer_2: T5TokenizerFast,
    #     transformer: FluxTransformer2DModel,
    # ):
    #     super().__init__(
    #         scheduler=scheduler,
    #         vae=vae,
    #         text_encoder=text_encoder,
    #         tokenizer=tokenizer,
    #         text_encoder_2=text_encoder_2,
    #         tokenizer_2=tokenizer_2,
    #         transformer=transformer,
    #     )

    #     # if transformer is not None:
    #     #     axes_dims_rope = transformer.config.axes_dims_rope
    #     #     transformer.pos_embed = CustomFluxPosEmbed(theta=10000, axes_dim=axes_dims_rope)

    #     self.vae_scale_factor = (
    #         2 ** (len(self.vae.config.block_out_channels)) if hasattr(self, "vae") and self.vae is not None else 16
    #     )
    #     self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
    #     self.tokenizer_max_length = (
    #         self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 77
    #     )
    #     self.default_sample_size = 64


# for sin, cos into relay graph
# class CustomFluxPosEmbed(FluxPosEmbed):
#     def __init__(self, theta: int, axes_dim: Tuple[int, ...]):
#         super().__init__(theta, axes_dim)

#     @staticmethod
#     def get_1d_rotary_pos_embed(
#         dim: int,
#         pos: Union[np.ndarray, int],
#         theta: float = 10000.0,
#         use_real=False,
#         linear_factor=1.0,
#         ntk_factor=1.0,
#         repeat_interleave_real=True,
#         freqs_dtype=torch.float32,  #  torch.float32, torch.float64 (flux)
#     ):
#         """
#         Precompute the frequency tensor for complex exponentials (cis) with given dimensions.
#         This function calculates a frequency tensor with complex exponentials using the given dimension 'dim' and the end
#         index 'end'. The 'theta' parameter scales the frequencies. The returned tensor contains complex values in complex64
#         data type.
#         Args:
#             dim (`int`): Dimension of the frequency tensor.
#             pos (`np.ndarray` or `int`): Position indices for the frequency tensor. [S] or scalar
#             theta (`float`, *optional*, defaults to 10000.0):
#                 Scaling factor for frequency computation. Defaults to 10000.0.
#             use_real (`bool`, *optional*):
#                 If True, return real part and imaginary part separately. Otherwise, return complex numbers.
#             linear_factor (`float`, *optional*, defaults to 1.0):
#                 Scaling factor for the context extrapolation. Defaults to 1.0.
#             ntk_factor (`float`, *optional*, defaults to 1.0):
#                 Scaling factor for the NTK-Aware RoPE. Defaults to 1.0.
#             repeat_interleave_real (`bool`, *optional*, defaults to `True`):
#                 If `True` and `use_real`, real part and imaginary part are each interleaved with themselves to reach `dim`.
#                 Otherwise, they are concateanted with themselves.
#             freqs_dtype (`torch.float32` or `torch.float64`, *optional*, defaults to `torch.float32`):
#                 the dtype of the frequency tensor.
#         Returns:
#             `torch.Tensor`: Precomputed frequency tensor with complex exponentials. [S, D/2]
#         """
#         assert dim % 2 == 0

#         if isinstance(pos, int):
#             pos = torch.arange(pos)
#         if isinstance(pos, np.ndarray):
#             pos = torch.from_numpy(pos)  # type: ignore  # [S]

#         theta = theta * ntk_factor
#         freqs = (
#             1.0
#             / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device)[: (dim // 2)] / dim))
#             / linear_factor
#         )  # [D/2]

#         inv_freq_expand = freqs[None, :]
#         pos_expand = pos[:, None]
#         freqs = pos_expand.float() @ inv_freq_expand.float()
#         # fix torch.outer -> inner product
#         # freqs = torch.outer(pos, freqs)  # type: ignore   # [S, D/2]

#         if use_real and repeat_interleave_real:
#             # flux, hunyuan-dit, cogvideox
#             freqs_cos = freqs.cos().repeat_interleave(2, dim=1).float()  # [S, D]
#             freqs_sin = freqs.sin().repeat_interleave(2, dim=1).float()  # [S, D]
#             return freqs_cos, freqs_sin
#         elif use_real:
#             # stable audio
#             freqs_cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).float()  # [S, D]
#             freqs_sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).float()  # [S, D]
#             return freqs_cos, freqs_sin
#         else:
#             # lumina
#             freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64     # [S, D/2]
#             return freqs_cis


#     def forward(self, ids: torch.Tensor) -> torch.Tensor:
#         n_axes = ids.shape[-1]
#         cos_out = []
#         sin_out = []
#         pos = ids.float()
#         is_mps = ids.device.type == "mps"
#         freqs_dtype = torch.float32 if is_mps else torch.float64
#         for i in range(n_axes):
#             cos, sin = self.get_1d_rotary_pos_embed(
#                 self.axes_dim[i], pos[:, i], repeat_interleave_real=True, use_real=True, freqs_dtype=freqs_dtype
#             )
#             cos_out.append(cos)
#             sin_out.append(sin)
#         freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
#         freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)
#         return freqs_cos, freqs_sin
