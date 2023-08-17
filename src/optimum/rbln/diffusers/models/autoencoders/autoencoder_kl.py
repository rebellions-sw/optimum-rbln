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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import rebel
import torch  # noqa: I001
from diffusers import AutoencoderKL
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from transformers import PretrainedConfig

from ....modeling import RBLNModel
from ....modeling_config import DEFAULT_COMPILED_MODEL_NAME, RBLNCompileConfig, RBLNConfig
from ...modeling_diffusers import RBLNDiffusionMixin
from .vae import RBLNRuntimeVAEDecoder, RBLNRuntimeVAEEncoder, _VAEDecoder, _VAEEncoder


if TYPE_CHECKING:
    import torch
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer, PretrainedConfig

logger = logging.getLogger(__name__)


class RBLNAutoencoderKL(RBLNModel):
    auto_model_class = AutoencoderKL
    config_name = "config.json"
    hf_library_name = "diffusers"

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)

        if self.rbln_config.model_cfg.get("img2img_pipeline") or self.rbln_config.model_cfg.get("inpaint_pipeline"):
            self.encoder = RBLNRuntimeVAEEncoder(runtime=self.model[0], main_input_name="x")
            self.decoder = RBLNRuntimeVAEDecoder(runtime=self.model[1], main_input_name="z")
        else:
            self.decoder = RBLNRuntimeVAEDecoder(runtime=self.model[0], main_input_name="z")

        self.image_size = self.rbln_config.model_cfg["sample_size"]

    @classmethod
    def get_compiled_model(cls, model, rbln_config: RBLNConfig):
        def compile_img2img():
            encoder_model = _VAEEncoder(model)
            decoder_model = _VAEDecoder(model)
            encoder_model.eval()
            decoder_model.eval()

            enc_compiled_model = cls.compile(encoder_model, rbln_compile_config=rbln_config.compile_cfgs[0])
            dec_compiled_model = cls.compile(decoder_model, rbln_compile_config=rbln_config.compile_cfgs[1])

            return {"encoder": enc_compiled_model, "decoder": dec_compiled_model}

        def compile_text2img():
            decoder_model = _VAEDecoder(model)
            decoder_model.eval()

            dec_compiled_model = cls.compile(decoder_model, rbln_compile_config=rbln_config.compile_cfgs[0])

            return dec_compiled_model

        if rbln_config.model_cfg.get("img2img_pipeline") or rbln_config.model_cfg.get("inpaint_pipeline"):
            return compile_img2img()
        else:
            return compile_text2img()

    @classmethod
    def get_vae_sample_size(cls, pipe: RBLNDiffusionMixin, rbln_config: Dict[str, Any]) -> Union[int, Tuple[int, int]]:
        image_size = (rbln_config.get("img_height"), rbln_config.get("img_width"))
        noise_module = getattr(pipe, "unet", None) or getattr(pipe, "transformer", None)
        vae_scale_factor = (
            pipe.vae_scale_factor
            if hasattr(pipe, "vae_scale_factor")
            else 2 ** (len(pipe.vae.config.block_out_channels) - 1)
        )

        if noise_module is None:
            raise AttributeError(
                "Cannot find noise processing or predicting module attributes. ex. U-Net, Transformer, ..."
            )

        if (image_size[0] is None) != (image_size[1] is None):
            raise ValueError("Both image height and image width must be given or not given")

        elif image_size[0] is None and image_size[1] is None:
            if rbln_config["img2img_pipeline"]:
                sample_size = noise_module.config.sample_size
            elif rbln_config["inpaint_pipeline"]:
                sample_size = noise_module.config.sample_size * vae_scale_factor
            else:
                # In case of text2img, sample size of vae decoder is determined by unet.
                noise_module_sample_size = noise_module.config.sample_size
                if isinstance(noise_module_sample_size, int):
                    sample_size = noise_module_sample_size * vae_scale_factor
                else:
                    sample_size = (
                        noise_module_sample_size[0] * vae_scale_factor,
                        noise_module_sample_size[1] * vae_scale_factor,
                    )
        else:
            sample_size = (image_size[0], image_size[1])

        return sample_size

    @classmethod
    def update_rbln_config_using_pipe(cls, pipe: RBLNDiffusionMixin, rbln_config: Dict[str, Any]) -> Dict[str, Any]:
        rbln_config.update({"sample_size": cls.get_vae_sample_size(pipe, rbln_config)})
        return rbln_config

    @classmethod
    def _get_rbln_config(
        cls,
        preprocessors: Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"],
        model_config: "PretrainedConfig",
        rbln_kwargs: Dict[str, Any] = {},
    ) -> RBLNConfig:
        rbln_batch_size = rbln_kwargs.get("batch_size")
        sample_size = rbln_kwargs.get("sample_size")
        is_img2img = rbln_kwargs.get("img2img_pipeline")
        is_inpaint = rbln_kwargs.get("inpaint_pipeline")

        if rbln_batch_size is None:
            rbln_batch_size = 1

        if sample_size is None:
            sample_size = model_config.sample_size

        if isinstance(sample_size, int):
            sample_size = (sample_size, sample_size)

        rbln_kwargs["sample_size"] = sample_size

        if hasattr(model_config, "block_out_channels"):
            vae_scale_factor = 2 ** (len(model_config.block_out_channels) - 1)
        else:
            # vae image processor default value 8 (int)
            vae_scale_factor = 8

        dec_shape = (sample_size[0] // vae_scale_factor, sample_size[1] // vae_scale_factor)
        enc_shape = (sample_size[0], sample_size[1])

        if is_img2img or is_inpaint:
            vae_enc_input_info = [
                (
                    "x",
                    [rbln_batch_size, model_config.in_channels, enc_shape[0], enc_shape[1]],
                    "float32",
                )
            ]
            vae_dec_input_info = [
                (
                    "z",
                    [rbln_batch_size, model_config.latent_channels, dec_shape[0], dec_shape[1]],
                    "float32",
                )
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

        vae_config = RBLNCompileConfig(
            input_info=[
                (
                    "z",
                    [rbln_batch_size, model_config.latent_channels, dec_shape[0], dec_shape[1]],
                    "float32",
                )
            ]
        )
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

    def encode(self, x: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        posterior = self.encoder.encode(x)
        return AutoencoderKLOutput(latent_dist=posterior)

    def decode(self, z: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        return self.decoder.decode(z)
