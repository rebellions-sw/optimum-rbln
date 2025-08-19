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

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Tuple, Union

import rebel
from diffusers import AutoencoderKLWan
from diffusers.models.autoencoders.vae import DecoderOutput
from diffusers.models.modeling_outputs import AutoencoderKLOutput

from ....configuration_utils import RBLNCompileConfig
from ....modeling import RBLNModel
from ....utils.logging import get_logger
from ...configurations import RBLNAutoencoderKLWanConfig
from .vae import RBLNRuntimeWanVAEDecoder, RBLNRuntimeWanVAEEncoder, _VAEWanDecoder, _VAEWanEncoder


if TYPE_CHECKING:
    import torch
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer, PretrainedConfig, PreTrainedModel

    from ...modeling_diffusers import RBLNDiffusionMixin, RBLNDiffusionMixinConfig

logger = get_logger(__name__)


class RBLNAutoencoderKLWan(RBLNModel):
    """
    RBLN implementation of AutoencoderKLWan for diffusion models.

    This model is used to accelerate AutoencoderKLWan models from diffusers library on RBLN NPUs.
    It can be configured to include both encoder and decoder, or just the decoder part for latent-to-video
    conversion.

    This class inherits from [`RBLNModel`]. Check the superclass documentation for the generic methods
    the library implements for all its models.
    """

    auto_model_class = AutoencoderKLWan
    hf_library_name = "diffusers"
    _rbln_config_class = RBLNAutoencoderKLWanConfig

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)

        if self.rbln_config.uses_encoder:
            self.encoder = RBLNRuntimeWanVAEEncoder(
                runtime=self.model[0], main_input_name="x", use_slicing=self.rbln_config.use_slicing
            )

        self.decoder = RBLNRuntimeWanVAEDecoder(
            runtime=self.model[-1], main_input_name="z", use_slicing=self.rbln_config.use_slicing
        )
        self.image_size = self.rbln_config.image_size

    @classmethod
    def wrap_model_if_needed(cls, model: torch.nn.Module, rbln_config: RBLNAutoencoderKLWanConfig) -> torch.nn.Module:
        decoder_model = _VAEWanDecoder(model)
        decoder_model.eval()

        if rbln_config.uses_encoder:
            encoder_model = _VAEWanEncoder(model)
            encoder_model.eval()
            return encoder_model, decoder_model
        else:
            return decoder_model

    @classmethod
    def get_compiled_model(cls, model, rbln_config: RBLNAutoencoderKLWanConfig) -> Dict[str, rebel.RBLNCompiledModel]:
        compiled_models = {}
        if rbln_config.uses_encoder:
            encoder_model, decoder_model = cls.wrap_model_if_needed(model, rbln_config)
            enc_compiled_model = cls.compile(
                encoder_model,
                rbln_compile_config=rbln_config.compile_cfgs[0],
                create_runtimes=rbln_config.create_runtimes,
                device=rbln_config.device_map["encoder"],
            )
            compiled_models["encoder"] = enc_compiled_model
        else:
            decoder_model = cls.wrap_model_if_needed(model, rbln_config)
        dec_compiled_model = cls.compile(
            decoder_model,
            rbln_compile_config=rbln_config.compile_cfgs[-1],
            create_runtimes=rbln_config.create_runtimes,
            device=rbln_config.device_map["decoder"],
        )
        compiled_models["decoder"] = dec_compiled_model

        return compiled_models

    @classmethod
    def update_rbln_config_using_pipe(
        cls, pipe: RBLNDiffusionMixin, rbln_config: RBLNDiffusionMixinConfig, submodule_name: str
    ) -> RBLNDiffusionMixinConfig:
        rbln_config.vae.num_channels_latents = pipe.transformer.config.in_channels - int(rbln_config.vae.uses_encoder)
        rbln_config.vae.vae_scale_factor_temporal = pipe.vae_scale_factor_temporal
        rbln_config.vae.vae_scale_factor_spatial = pipe.vae_scale_factor_spatial
        return rbln_config

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Union[AutoFeatureExtractor, AutoProcessor, AutoTokenizer],
        model: PreTrainedModel,
        model_config: PretrainedConfig,
        rbln_config: RBLNAutoencoderKLWanConfig,
    ) -> RBLNAutoencoderKLWanConfig:
        batch_size = 1 if rbln_config.use_slicing else rbln_config.batch_size
        compile_cfgs = []
        if rbln_config.uses_encoder:
            vae_enc_input_info = [
                (
                    "x",
                    [
                        batch_size,
                        3,
                        rbln_config.num_frames,
                        rbln_config.height,
                        rbln_config.width,
                    ],
                    "float32",
                ),
            ]
            compile_cfgs.append(RBLNCompileConfig(compiled_model_name="encoder", input_info=vae_enc_input_info))

        num_latent_frames = (rbln_config.num_frames - 1) // rbln_config.vae_scale_factor_temporal + 1
        latent_height = rbln_config.height // rbln_config.vae_scale_factor_spatial
        latent_width = rbln_config.width // rbln_config.vae_scale_factor_spatial

        vae_dec_input_info = [
            (
                "z",
                [
                    batch_size,
                    rbln_config.num_channels_latents,
                    num_latent_frames,
                    latent_height,
                    latent_width,
                ],
                "float32",
            ),
        ]
        compile_cfgs.append(RBLNCompileConfig(compiled_model_name="decoder", input_info=vae_dec_input_info))

        rbln_config.set_compile_cfgs(compile_cfgs)
        return rbln_config

    @classmethod
    def _create_runtimes(
        cls,
        compiled_models: List[rebel.RBLNCompiledModel],
        rbln_config: RBLNAutoencoderKLWanConfig,
    ) -> List[rebel.Runtime]:
        if len(compiled_models) == 1:
            # decoder
            expected_models = ["decoder"]
        else:
            expected_models = ["encoder", "decoder"]

        if any(model_name not in rbln_config.device_map for model_name in expected_models):
            cls._raise_missing_compiled_file_error(expected_models)

        device_vals = [rbln_config.device_map[model_name] for model_name in expected_models]
        return [
            rebel.Runtime(
                compiled_model,
                tensor_type="pt",
                device=device_val,
                activate_profiler=rbln_config.activate_profiler,
                timeout=rbln_config.timeout,
            )
            for compiled_model, device_val in zip(compiled_models, device_vals)
        ]

    def encode(
        self, x: torch.Tensor, return_dict: bool = True
    ) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
        posterior = self.encoder.encode(x)
        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    def decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.Tensor]:
        decoded = self.decoder.decode(z)
        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)
