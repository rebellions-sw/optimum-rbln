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
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import torch
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from transformers import PretrainedConfig

from ....modeling import RBLNModel
from ....modeling_config import RBLNCompileConfig, RBLNConfig
from ...modeling_diffusers import RBLNDiffusionMixin


if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer

logger = logging.getLogger(__name__)


class _UNet_SD(torch.nn.Module):
    def __init__(self, unet: "UNet2DConditionModel"):
        super().__init__()
        self.unet = unet

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        *down_and_mid_block_additional_residuals: Optional[Tuple[torch.Tensor]],
        text_embeds: Optional[torch.Tensor] = None,
        time_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if text_embeds is not None and time_ids is not None:
            added_cond_kwargs = {"text_embeds": text_embeds, "time_ids": time_ids}
        else:
            added_cond_kwargs = {}

        if len(down_and_mid_block_additional_residuals) != 0:
            down_block_additional_residuals, mid_block_additional_residual = (
                down_and_mid_block_additional_residuals[:-1],
                down_and_mid_block_additional_residuals[-1],
            )
        else:
            down_block_additional_residuals, mid_block_additional_residual = None, None

        unet_out = self.unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )
        return unet_out


class _UNet_SDXL(torch.nn.Module):
    def __init__(self, unet: "UNet2DConditionModel"):
        super().__init__()
        self.unet = unet

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        *down_and_mid_block_additional_residuals: Optional[Tuple[torch.Tensor]],
    ) -> torch.Tensor:
        if len(down_and_mid_block_additional_residuals) == 2:
            added_cond_kwargs = {
                "text_embeds": down_and_mid_block_additional_residuals[0],
                "time_ids": down_and_mid_block_additional_residuals[1],
            }
            down_block_additional_residuals = None
            mid_block_additional_residual = None
        elif len(down_and_mid_block_additional_residuals) > 2:
            added_cond_kwargs = {
                "text_embeds": down_and_mid_block_additional_residuals[-2],
                "time_ids": down_and_mid_block_additional_residuals[-1],
            }
            down_block_additional_residuals, mid_block_additional_residual = (
                down_and_mid_block_additional_residuals[:-3],
                down_and_mid_block_additional_residuals[-3],
            )
        else:
            added_cond_kwargs = {}
            down_block_additional_residuals = None
            mid_block_additional_residual = None

        unet_out = self.unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )
        return unet_out


class RBLNUNet2DConditionModel(RBLNModel):
    hf_library_name = "diffusers"
    auto_model_class = UNet2DConditionModel

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
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
        if model.config.addition_embed_type == "text_time":
            return _UNet_SDXL(model).eval()
        else:
            return _UNet_SD(model).eval()

    @classmethod
    def get_unet_sample_size(
        cls, pipe: RBLNDiffusionMixin, rbln_config: Dict[str, Any]
    ) -> Union[int, Tuple[int, int]]:
        image_size = (rbln_config.get("img_height"), rbln_config.get("img_width"))
        if (image_size[0] is None) != (image_size[1] is None):
            raise ValueError("Both image height and image width must be given or not given")
        elif image_size[0] is None and image_size[1] is None:
            if rbln_config["img2img_pipeline"]:
                # In case of img2img, sample size of unet is determined by vae encoder.
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
        text_model_hidden_size = pipe.text_encoder_2.config.hidden_size if hasattr(pipe, "text_encoder_2") else None

        batch_size = rbln_config.get("batch_size")
        if not batch_size:
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
                "max_seq_len": pipe.text_encoder.config.max_position_embeddings,
                "text_model_hidden_size": text_model_hidden_size,
                "sample_size": cls.get_unet_sample_size(pipe, rbln_config),
                "batch_size": batch_size,
                "is_controlnet": "controlnet" in pipe.config.keys(),
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
        max_seq_len = rbln_kwargs.get("max_seq_len")
        sample_size = rbln_kwargs.get("sample_size")
        is_controlnet = rbln_kwargs.get("is_controlnet")
        rbln_in_features = None

        if batch_size is None:
            batch_size = 1

        if sample_size is None:
            sample_size = model_config.sample_size

        if isinstance(sample_size, int):
            sample_size = (sample_size, sample_size)

        if max_seq_len is None:
            raise ValueError("`rbln_max_seq_len` (ex. text_encoder's max_position_embeddings) must be specified.")

        input_info = [
            ("sample", [batch_size, model_config.in_channels, sample_size[0], sample_size[1]], "float32"),
            ("timestep", [], "float32"),
            ("encoder_hidden_states", [batch_size, max_seq_len, model_config.cross_attention_dim], "float32"),
        ]

        if is_controlnet:
            # down block addtional residuals
            first_shape = [batch_size, model_config.block_out_channels[0], sample_size[0], sample_size[1]]
            height, width = sample_size[0], sample_size[1]
            input_info.append(("down_block_additional_residuals_0", first_shape, "float32"))
            name_idx = 1
            for idx, _ in enumerate(model_config.down_block_types):
                shape = [batch_size, model_config.block_out_channels[idx], height, width]
                for _ in range(model_config.layers_per_block):
                    input_info.append((f"down_block_additional_residuals_{name_idx}", shape, "float32"))
                    name_idx += 1
                if idx != len(model_config.down_block_types) - 1:
                    height = height // 2
                    width = width // 2
                    shape = [batch_size, model_config.block_out_channels[idx], height, width]
                    input_info.append((f"down_block_additional_residuals_{name_idx}", shape, "float32"))
                    name_idx += 1

            # mid block addtional residual
            num_cross_attn_blocks = model_config.down_block_types.count("CrossAttnDownBlock2D")
            out_channels = model_config.block_out_channels[-1]
            shape = [
                batch_size,
                out_channels,
                sample_size[0] // 2**num_cross_attn_blocks,
                sample_size[1] // 2**num_cross_attn_blocks,
            ]
            input_info.append(("mid_block_additional_residual", shape, "float32"))

        if hasattr(model_config, "addition_embed_type") and model_config.addition_embed_type == "text_time":
            rbln_text_model_hidden_size = rbln_kwargs["text_model_hidden_size"]
            rbln_in_features = model_config.projection_class_embeddings_input_dim
            input_info.append(("text_embeds", [batch_size, rbln_text_model_hidden_size], "float32"))
            input_info.append(("time_ids", [batch_size, 6], "float32"))

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
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = {},
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
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

        added_cond_kwargs = {} if added_cond_kwargs is None else added_cond_kwargs

        if down_block_additional_residuals is not None:
            down_block_additional_residuals = [t.contiguous() for t in down_block_additional_residuals]
            return (
                super().forward(
                    sample.contiguous(),
                    timestep.float(),
                    encoder_hidden_states,
                    *down_block_additional_residuals,
                    mid_block_additional_residual,
                    **added_cond_kwargs,
                ),
            )

        return (
            super().forward(
                sample.contiguous(),
                timestep.float(),
                encoder_hidden_states,
                **added_cond_kwargs,
            ),
        )
