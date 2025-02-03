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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Tuple, Union

import torch
from diffusers.models.unets.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
from transformers import PretrainedConfig

from ....modeling import RBLNModel
from ....modeling_config import RBLNCompileConfig, RBLNConfig
from ...modeling_diffusers import RBLNDiffusionMixin


if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor

logger = logging.getLogger(__name__)


class _UNet_STCD(torch.nn.Module):
    def __init__(self, unet: "UNetSpatioTemporalConditionModel"):
        super().__init__()
        self.unet = unet

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        added_time_ids: torch.Tensor,
    ) -> torch.Tensor:
        unet_out = self.unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            added_time_ids=added_time_ids,
            return_dict=False,
        )
        return unet_out


class RBLNUNetSpatioTemporalConditionModel(RBLNModel):
    hf_library_name = "diffusers"
    auto_model_class = UNetSpatioTemporalConditionModel

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        # self.in_features = self.config.projection_class_embeddings_input_dim
        self.in_features = self.rbln_config.model_cfg.get("in_features", None)
        if self.in_features is not None:

            @dataclass
            class LINEAR1:
                in_features: int

            @dataclass
            class ADDEMBEDDING:
                linear_1: LINEAR1

            self.add_embedding = ADDEMBEDDING(LINEAR1(self.in_features))

    @classmethod
    def wrap_model_if_needed(cls, model: torch.nn.Module, rbln_config: RBLNConfig) -> torch.nn.Module:
        return _UNet_STCD(model).eval()

    @classmethod
    def get_unet_sample_size(
        cls, pipe: RBLNDiffusionMixin, rbln_config: Dict[str, Any]
    ) -> Union[int, Tuple[int, int]]:
        image_size = (rbln_config.get("img_height"), rbln_config.get("img_width"))
        if (image_size[0] is None) != (image_size[1] is None):
            raise ValueError("Both image height and image width must be given or not given")
        elif image_size[0] is None and image_size[1] is None:
            if rbln_config["img2vid_pipeline"]:
                # In case of img2vid, sample size of unet is determined by vae encoder.
                vae_sample_size = pipe.vae.config.sample_size
                if isinstance(vae_sample_size, int):
                    sample_size = vae_sample_size // pipe.vae_scale_factor
                else:
                    sample_size = (
                        vae_sample_size[0] // pipe.vae_scale_factor,
                        vae_sample_size[1] // pipe.vae_scale_factor,
                    )
            else:
                sample_size = pipe.unet.config.sample_size
        else:
            sample_size = (image_size[0] // pipe.vae_scale_factor, image_size[1] // pipe.vae_scale_factor)
        return sample_size

    @classmethod
    def update_rbln_config_using_pipe(cls, pipe: RBLNDiffusionMixin, rbln_config: Dict[str, Any]) -> Dict[str, Any]:
        batch_size = rbln_config.get("batch_size")
        if not batch_size:
            if rbln_config.get("img2vid_pipeline"):
                do_classifier_free_guidance = True
            else:
                do_classifier_free_guidance = (
                    rbln_config.get("guidance_scale", 5.0) > 1.0 and pipe.unet.config.time_cond_proj_dim is None
                )
            batch_size = 2 if do_classifier_free_guidance else 1
        else:
            if rbln_config.get("guidance_scale"):
                logger.warning(
                    "guidance_scale is ignored because batch size is explicitly specified. "
                    "To ensure consistent behavior, consider removing the guidance scale or "
                    "adjusting the batch size configuration as needed."
                )

        rbln_config.update(
            {
                "sample_size": cls.get_unet_sample_size(pipe, rbln_config),
                "batch_size": batch_size,
            }
        )

        if rbln_config.get("img2vid_pipeline"):
            rbln_config["num_frames"] = pipe.unet.config.num_frames

        return rbln_config

    @classmethod
    def _get_rbln_config(
        cls,
        preprocessors: Union["AutoFeatureExtractor", "AutoProcessor"],
        model_config: "PretrainedConfig",
        rbln_kwargs: Dict[str, Any] = {},
    ) -> RBLNConfig:
        batch_size = rbln_kwargs.get("batch_size")
        sample_size = rbln_kwargs.get("sample_size")
        num_frames = rbln_kwargs.get("num_frames")
        rbln_in_features = None

        if batch_size is None:
            batch_size = 1

        if sample_size is None:
            sample_size = model_config.sample_size

        if isinstance(sample_size, int):
            sample_size = (sample_size, sample_size)

        input_info = [
            ("sample", [batch_size, num_frames, model_config.in_channels, sample_size[0], sample_size[1]], "float32"),
            ("timestep", [], "float32"),
            ("encoder_hidden_states", [batch_size, 1, model_config.cross_attention_dim], "float32"),
            ("added_time_ids", [batch_size, 3], "float32"),
        ]

        if hasattr(model_config, "addition_time_embed_dim"):
            rbln_in_features = model_config.projection_class_embeddings_input_dim

        rbln_compile_config = RBLNCompileConfig(input_info=input_info)
        rbln_config = RBLNConfig(
            rbln_cls=cls.__name__,
            compile_cfgs=[rbln_compile_config],
            rbln_kwargs=rbln_kwargs,
        )

        if rbln_in_features is not None:
            rbln_config.model_cfg["in_features"] = rbln_in_features

        return rbln_config

    @property
    def compiled_batch_size(self):
        return self.rbln_config.compile_cfgs[0].input_info[0][1][0]

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        added_time_ids: torch.Tensor,
        return_dict: bool = True,
        **kwargs,
    ):
        sample_batch_size = sample.size()[0]
        compiled_batch_size = self.compiled_batch_size
        if sample_batch_size != compiled_batch_size and (
            sample_batch_size * 2 == compiled_batch_size or sample_batch_size == compiled_batch_size * 2
        ):
            raise ValueError(
                f"Mismatch between UNet's runtime batch size ({sample_batch_size}) and compiled batch size ({compiled_batch_size}). "
                "This may be caused by the 'guidance scale' parameter, which doubles the runtime batch size in Stable Diffusion. "
                "Adjust the batch size during compilation or modify the 'guidance scale' to match the compiled batch size.\n\n"
                "For details, see: https://docs.rbln.ai/software/optimum/model_api.html#stable-diffusion"
            )

        return (
            super().forward(
                sample.contiguous(),
                timestep.float(),
                encoder_hidden_states,
                added_time_ids,
            ),
        )
