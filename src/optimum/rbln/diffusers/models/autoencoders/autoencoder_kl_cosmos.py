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
from torch.nn import functional as F
from transformers import PretrainedConfig

from ....modeling import RBLNModel
from ....modeling_config import DEFAULT_COMPILED_MODEL_NAME, RBLNCompileConfig, RBLNConfig
from ....utils.logging import get_logger
from ...modeling_diffusers import RBLNDiffusionMixin
from .vae import RBLNRuntimeVAEDecoder, _VAEDecoder


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
        # Temporary support decoder only
        self.decoder = RBLNRuntimeVAEDecoder(runtime=self.model[0], main_input_name="z")

    @classmethod
    def wrap_model_if_needed(cls, model: torch.nn.Module, rbln_config: RBLNConfig) -> torch.nn.Module:
        def replace_forward_func(model):
            for name, module in model.named_children():
                if isinstance(module, CosmosCausalConv3d) and module.temporal_pad == 0:
                    module.forward = replaced_forward.__get__(module, module.__class__)
                else:
                    replace_forward_func(module)

        replace_forward_func(model)
        decoder_model = _VAEDecoder(model)
        decoder_model.eval()
        return decoder_model

    @classmethod
    def get_compiled_model(cls, model, rbln_config: RBLNConfig):
        decoder_model = cls.wrap_model_if_needed(model, rbln_config)
        dec_compiled_model = cls.compile(decoder_model, rbln_compile_config=rbln_config.compile_cfgs[0])
        return dec_compiled_model

    @classmethod
    def update_rbln_config_using_pipe(cls, pipe: RBLNDiffusionMixin, rbln_config: Dict[str, Any]) -> Dict[str, Any]:
        num_channel_latents = pipe.transformer.config.in_channels
        num_frames = rbln_config.get("num_frames", 121)
        num_latent_frames = (num_frames - 1) // pipe.vae_scale_factor_temporal + 1
        height = rbln_config.get("height", 704)
        latent_height = height // pipe.vae_scale_factor_temporal
        width = rbln_config.get("width", 1280)
        latent_width = width // pipe.vae_scale_factor_temporal

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
        if DEFAULT_COMPILED_MODEL_NAME not in rbln_device_map:
            cls._raise_missing_compiled_file_error([DEFAULT_COMPILED_MODEL_NAME])

        device_val = rbln_device_map[DEFAULT_COMPILED_MODEL_NAME]
        return [
            compiled_models[0].create_runtime(tensor_type="pt", device=device_val, activate_profiler=activate_profiler)
        ]

    def decode(self, z: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        return self.decoder.decode(z)
