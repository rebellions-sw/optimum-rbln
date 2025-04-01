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

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import rebel
import torch
from diffusers import VQModel
from diffusers.models.autoencoders.vae import DecoderOutput
from diffusers.models.autoencoders.vq_model import VQEncoderOutput
from transformers import PretrainedConfig

from ....modeling import RBLNModel
from ....modeling_config import DEFAULT_COMPILED_MODEL_NAME, RBLNCompileConfig, RBLNConfig
from ....utils.logging import get_logger
from ...modeling_diffusers import RBLNDiffusionMixin
from .vae import RBLNRuntimeVQDecoder, RBLNRuntimeVQEncoder, _VQDecoder, _VQEncoder


if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer

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
        height = self.rbln_config.model_cfg.get("img_height", 512)
        width = self.rbln_config.model_cfg.get("img_width", 512)
        self.image_size = [height, width]

    @classmethod
    def get_compiled_model(cls, model, rbln_config: RBLNConfig):
        encoder_model = _VQEncoder(model)
        decoder_model = _VQDecoder(model)
        encoder_model.eval()
        decoder_model.eval()

        enc_compiled_model = cls.compile(encoder_model, rbln_compile_config=rbln_config.compile_cfgs[0])
        dec_compiled_model = cls.compile(decoder_model, rbln_compile_config=rbln_config.compile_cfgs[1])

        return {"encoder": enc_compiled_model, "decoder": dec_compiled_model}

    @classmethod
    def update_rbln_config_using_pipe(cls, pipe: RBLNDiffusionMixin, rbln_config: Dict[str, Any]) -> Dict[str, Any]:
        batch_size = rbln_config.get("batch_size")
        if batch_size is None:
            batch_size = 1
        img_height = rbln_config.get("img_height")
        if img_height is None:
            img_height = 512
        img_width = rbln_config.get("img_width")
        if img_width is None:
            img_width = 512

        rbln_config.update(
            {
                "batch_size": batch_size,
                "img_height": img_height,
                "img_width": img_width,
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
        batch_size = rbln_kwargs.get("batch_size")
        if batch_size is None:
            batch_size = 1

        height = rbln_kwargs.get("img_height")
        if height is None:
            height = 512

        width = rbln_kwargs.get("img_width")
        if width is None:
            width = 512

        if hasattr(model_config, "block_out_channels"):
            scale_factor = 2 ** (len(model_config.block_out_channels) - 1)
        else:
            # image processor default value 8 (int)
            scale_factor = 8

        enc_shape = (height, width)
        dec_shape = (height // scale_factor, width // scale_factor)

        enc_input_info = [
            (
                "x",
                [batch_size, model_config.in_channels, enc_shape[0], enc_shape[1]],
                "float32",
            )
        ]
        dec_input_info = [
            (
                "h",
                [batch_size, model_config.latent_channels, dec_shape[0], dec_shape[1]],
                "float32",
            )
        ]

        enc_rbln_compile_config = RBLNCompileConfig(compiled_model_name="encoder", input_info=enc_input_info)
        dec_rbln_compile_config = RBLNCompileConfig(compiled_model_name="decoder", input_info=dec_input_info)

        compile_cfgs = [enc_rbln_compile_config, dec_rbln_compile_config]
        rbln_config = RBLNConfig(
            rbln_cls=cls.__name__,
            compile_cfgs=compile_cfgs,
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
            device_val = rbln_device_map[DEFAULT_COMPILED_MODEL_NAME]
            return [
                compiled_models[0].create_runtime(
                    tensor_type="pt", device=device_val, activate_profiler=activate_profiler
                )
            ]

        device_vals = [rbln_device_map["encoder"], rbln_device_map["decoder"]]
        return [
            compiled_model.create_runtime(tensor_type="pt", device=device_val, activate_profiler=activate_profiler)
            for compiled_model, device_val in zip(compiled_models, device_vals)
        ]

    def encode(self, x: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        posterior = self.encoder.encode(x)
        return VQEncoderOutput(latents=posterior)

    def decode(self, h: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        dec, commit_loss = self.decoder.decode(h, **kwargs)
        return DecoderOutput(sample=dec, commit_loss=commit_loss)
