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

from typing import TYPE_CHECKING, List, Union

import rebel
import torch
from diffusers import VQModel
from diffusers.models.autoencoders.vae import DecoderOutput
from diffusers.models.autoencoders.vq_model import VQEncoderOutput

from ....configuration_utils import DEFAULT_COMPILED_MODEL_NAME, RBLNCompileConfig, RBLNModelConfig
from ....modeling import RBLNModel
from ....utils.logging import get_logger
from ...configurations.models.configuration_vq_model import RBLNVQModelConfig
from ...modeling_diffusers import RBLNDiffusionMixin, RBLNDiffusionMixinConfig
from .vae import RBLNRuntimeVQDecoder, RBLNRuntimeVQEncoder, _VQDecoder, _VQEncoder


if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer, PretrainedConfig, PreTrainedModel

logger = get_logger(__name__)


class RBLNVQModel(RBLNModel):
    auto_model_class = VQModel
    config_name = "config.json"
    hf_library_name = "diffusers"

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)

        self.encoder = RBLNRuntimeVQEncoder(runtime=self.model[0], main_input_name="x")
        self.decoder = RBLNRuntimeVQDecoder(runtime=self.model[1], main_input_name="z")
        self.decoder.lookup_from_codebook = self.config.lookup_from_codebook
        self.image_size = self.rbln_config.image_size

    @classmethod
    def get_compiled_model(cls, model, rbln_config: RBLNModelConfig):
        encoder_model = _VQEncoder(model)
        decoder_model = _VQDecoder(model)
        encoder_model.eval()
        decoder_model.eval()

        enc_compiled_model = cls.compile(encoder_model, rbln_compile_config=rbln_config.compile_cfgs[0])
        dec_compiled_model = cls.compile(decoder_model, rbln_compile_config=rbln_config.compile_cfgs[1])

        return {"encoder": enc_compiled_model, "decoder": dec_compiled_model}

    @classmethod
    def update_rbln_config_using_pipe(
        cls, pipe: RBLNDiffusionMixin, rbln_config: "RBLNDiffusionMixinConfig", submodule_name: str
    ) -> "RBLNDiffusionMixinConfig":
        return rbln_config

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"],
        model: "PreTrainedModel",
        model_config: "PretrainedConfig",
        rbln_config: RBLNVQModelConfig,
    ) -> RBLNVQModelConfig:
        if hasattr(model_config, "block_out_channels"):
            rbln_config.vqmodel_scale_factor = 2 ** (len(model_config.block_out_channels) - 1)
        else:
            # image processor default value 8 (int)
            rbln_config.vqmodel_scale_factor = 8

        enc_shape = rbln_config.image_size
        dec_shape = rbln_config.latent_sample_size

        enc_input_info = [
            (
                "x",
                [rbln_config.batch_size, model_config.in_channels, enc_shape[0], enc_shape[1]],
                "float32",
            )
        ]
        dec_input_info = [
            (
                "h",
                [rbln_config.batch_size, model_config.latent_channels, dec_shape[0], dec_shape[1]],
                "float32",
            )
        ]

        enc_rbln_compile_config = RBLNCompileConfig(compiled_model_name="encoder", input_info=enc_input_info)
        dec_rbln_compile_config = RBLNCompileConfig(compiled_model_name="decoder", input_info=dec_input_info)

        compile_cfgs = [enc_rbln_compile_config, dec_rbln_compile_config]
        rbln_config.set_compile_cfgs(compile_cfgs)
        return rbln_config

    @classmethod
    def _create_runtimes(
        cls,
        compiled_models: List[rebel.RBLNCompiledModel],
        rbln_config: RBLNVQModelConfig,
    ) -> List[rebel.Runtime]:
        if len(compiled_models) == 1:
            device_val = rbln_config.device_map[DEFAULT_COMPILED_MODEL_NAME]
            return [
                compiled_models[0].create_runtime(
                    tensor_type="pt", device=device_val, activate_profiler=rbln_config.activate_profiler
                )
            ]

        device_vals = [rbln_config.device_map["encoder"], rbln_config.device_map["decoder"]]
        return [
            rebel.Runtime(
                compiled_model,
                tensor_type="pt",
                device=device_val,
                activate_profiler=rbln_config.activate_profiler,
            )
            for compiled_model, device_val in zip(compiled_models, device_vals)
        ]

    def encode(self, x: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        posterior = self.encoder.encode(x)
        return VQEncoderOutput(latents=posterior)

    def decode(self, h: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        dec, commit_loss = self.decoder.decode(h, **kwargs)
        return DecoderOutput(sample=dec, commit_loss=commit_loss)
