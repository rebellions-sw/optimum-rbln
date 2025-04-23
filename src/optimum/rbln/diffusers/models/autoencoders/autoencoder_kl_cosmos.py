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

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import rebel
import torch
from diffusers.models.autoencoders.autoencoder_kl_cosmos import CosmosCausalConv3d
from diffusers.models.autoencoders.vae import DecoderOutput
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from torch.nn import functional as F
from transformers import PretrainedConfig

from ....modeling import RBLNModel
from ....modeling_config import DEFAULT_COMPILED_MODEL_NAME, RBLNCompileConfig, RBLNConfig
from ....utils.logging import get_logger
from ...modeling_diffusers import RBLNDiffusionMixin
from .vae import RBLNRuntimeVAEDecoder, RBLNRuntimeVAEEncoder, _VAECosmosDecoder, _VAEEncoder


if TYPE_CHECKING:
    import torch
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer, PretrainedConfig

logger = get_logger(__name__)


def replaced_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    if self.temporal_pad != 0:
        hidden_states_prev = hidden_states[:, :, :1, ...].repeat(1, 1, self.temporal_pad, 1, 1)
        hidden_states = torch.cat([hidden_states_prev, hidden_states], dim=2)
    hidden_states = F.pad(hidden_states, (*self.spatial_pad, 0, 0), mode=self.pad_mode, value=0.0)
    return super(CosmosCausalConv3d, self).forward(hidden_states)


class RBLNAutoencoderKLCosmos(RBLNModel):
    config_name = "config.json"
    hf_library_name = "diffusers"

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        if self.rbln_config.model_cfg.get("img2vid_pipeline"):
            self.encoder = RBLNRuntimeVAEEncoder(runtime=self.model[0], main_input_name="x")
            self.decoder = RBLNRuntimeVAEDecoder(runtime=self.model[1], main_input_name="z")
        else:
            self.decoder = RBLNRuntimeVAEDecoder(runtime=self.model[0], main_input_name="z")

        height = self.rbln_config.model_cfg.get("height")
        width = self.rbln_config.model_cfg.get("width")
        self.image_size = [height, width]

    @classmethod
    def wrap_model_if_needed(cls, model: torch.nn.Module, rbln_config: RBLNConfig) -> torch.nn.Module:
        def replace_forward_func(model):
            for name, module in model.named_children():
                if isinstance(module, CosmosCausalConv3d) and module.temporal_pad == 0:
                    module.forward = replaced_forward.__get__(module, module.__class__)
                else:
                    replace_forward_func(module)

        replace_forward_func(model)
        if rbln_config.model_cfg.get("img2vid_pipeline"):
            encoder_model = _VAEEncoder(model)
            decoder_model = _VAECosmosDecoder(model)
            encoder_model.eval()
            decoder_model.eval()
            return encoder_model, decoder_model
        else:
            decoder_model = _VAECosmosDecoder(model)
            decoder_model.eval()
            return decoder_model

    @classmethod
    def get_compiled_model(cls, model, rbln_config: RBLNConfig):
        def compile_encoder_decoder():
            encoder_model, decoder_model = cls.wrap_model_if_needed(model, rbln_config)
            enc_compiled_model = cls.compile(encoder_model, rbln_compile_config=rbln_config.compile_cfgs[0])
            dec_compiled_model = cls.compile(decoder_model, rbln_compile_config=rbln_config.compile_cfgs[1])
            return {"encoder": enc_compiled_model, "decoder": dec_compiled_model}

        def compile_decoder_only():
            decoder_model = cls.wrap_model_if_needed(model, rbln_config)
            dec_compiled_model = cls.compile(decoder_model, rbln_compile_config=rbln_config.compile_cfgs[0])
            return dec_compiled_model

        if rbln_config.model_cfg.get("img2vid_pipeline"):
            return compile_encoder_decoder()
        else:
            return compile_decoder_only()

    @classmethod
    def update_rbln_config_using_pipe(cls, pipe: RBLNDiffusionMixin, rbln_config: Dict[str, Any]) -> Dict[str, Any]:
        num_channel_latents = pipe.transformer.config.in_channels
        num_frames = rbln_config.get("num_frames", 121)
        num_latent_frames = (num_frames - 1) // pipe.vae_scale_factor_temporal + 1
        height = rbln_config.get("height", 704)
        latent_height = height // pipe.vae_scale_factor_spatial
        width = rbln_config.get("width", 1280)
        latent_width = width // pipe.vae_scale_factor_spatial

        rbln_config.update(
            {
                "num_channel_latents": num_channel_latents,
                "num_latent_frames": num_latent_frames,
                "latent_height": latent_height,
                "latent_width": latent_width,
            }
        )
        return rbln_config

    @classmethod
    def _get_rbln_config(
        cls,
        preprocessors: Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"],
        model_config: "PretrainedConfig",
        rbln_kwargs: Dict[str, Any] = {},
    ) -> RBLNConfig:
        rbln_batch_size = rbln_kwargs.get("batch_size")
        if rbln_batch_size is None:
            rbln_batch_size = 1

        rbln_num_channel_latents = rbln_kwargs.get("num_channel_latents")
        rbln_num_latent_frames = rbln_kwargs.get("num_latent_frames")
        rbln_latent_height = rbln_kwargs.get("latent_height")
        rbln_latent_width = rbln_kwargs.get("latent_width")

        if rbln_kwargs.get("img2vid_pipeline"):
            rbln_in_channels = model_config.in_channels
            rbln_num_frames = rbln_kwargs.get("num_frames", 121)
            rbln_height = rbln_kwargs.get("height", 704)
            rbln_width = rbln_kwargs.get("width", 1280)

            vae_enc_input_info = [
                (
                    "x",
                    [
                        rbln_batch_size,
                        rbln_in_channels,
                        rbln_num_frames,
                        rbln_height,
                        rbln_width,
                    ],
                    "float32",
                ),
            ]
            vae_dec_input_info = [
                (
                    "z",
                    [
                        rbln_batch_size,
                        rbln_num_channel_latents,
                        rbln_num_latent_frames,
                        rbln_latent_height,
                        rbln_latent_width,
                    ],
                    "float32",
                ),
            ]
            enc_rbln_compile_config = RBLNCompileConfig(compiled_model_name="encoder", input_info=vae_enc_input_info)
            dec_rbln_compile_config = RBLNCompileConfig(compiled_model_name="decoder", input_info=vae_dec_input_info)

            compile_cfgs = [enc_rbln_compile_config, dec_rbln_compile_config]
            rbln_config = RBLNConfig(
                rbln_cls=cls.__name__,
                compile_cfgs=compile_cfgs,
                rbln_kwargs=rbln_kwargs,
            )
            return rbln_config

        else:
            input_info = [
                (
                    "z",
                    [
                        rbln_batch_size,
                        rbln_num_channel_latents,
                        rbln_num_latent_frames,
                        rbln_latent_height,
                        rbln_latent_width,
                    ],
                    "float32",
                )
            ]
            vae_config = RBLNCompileConfig(input_info=input_info)
            rbln_config = RBLNConfig(
                rbln_cls=cls.__name__,
                compile_cfgs=[vae_config],
                rbln_kwargs=rbln_kwargs,
            )
            return rbln_config

    @classmethod
    def _create_runtimes(
        cls,
        compiled_models: List[rebel.RBLNCompiledModel],
        rbln_device_map: Dict[str, int],
        activate_profiler: Optional[bool] = None,
    ) -> List[rebel.Runtime]:
        if len(compiled_models) == 1:
            if DEFAULT_COMPILED_MODEL_NAME not in rbln_device_map:
                cls._raise_missing_compiled_file_error([DEFAULT_COMPILED_MODEL_NAME])

            device_val = rbln_device_map[DEFAULT_COMPILED_MODEL_NAME]
            return [
                compiled_models[0].create_runtime(
                    tensor_type="pt", device=device_val, activate_profiler=activate_profiler
                )
            ]

        if any(model_name not in rbln_device_map for model_name in ["encoder", "decoder"]):
            cls._raise_missing_compiled_file_error(["encoder", "decoder"])

        device_vals = [rbln_device_map["encoder"], rbln_device_map["decoder"]]
        return [
            compiled_model.create_runtime(tensor_type="pt", device=device_val, activate_profiler=activate_profiler)
            for compiled_model, device_val in zip(compiled_models, device_vals)
        ]

    def encode(self, x: torch.FloatTensor, return_dict: bool = True) -> torch.FloatTensor:
        posterior = self.encoder.encode(x)
        return AutoencoderKLOutput(latent_dist=posterior)

    def _decode(self, z: torch.FloatTensor, return_dict: bool = True) -> torch.FloatTensor:
        dec = self.decoder(z)

        if not return_dict:
            return (dec,)
        return DecoderOutput(sample=dec)

    def decode(self, z: torch.FloatTensor, return_dict: bool = True) -> torch.FloatTensor:
        if z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice).sample for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z).sample

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)
