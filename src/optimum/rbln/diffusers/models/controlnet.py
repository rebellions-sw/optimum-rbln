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

import importlib
import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import torch
from diffusers import ControlNetModel
from transformers import PretrainedConfig

from ...modeling import RBLNModel
from ...modeling_config import RBLNCompileConfig, RBLNConfig
from ..modeling_diffusers import RBLNDiffusionMixin


if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer


logger = logging.getLogger(__name__)


class _ControlNetModel(torch.nn.Module):
    def __init__(self, controlnet: "ControlNetModel"):
        super().__init__()
        self.controlnet = controlnet

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        controlnet_cond: torch.Tensor,
        conditioning_scale,
        text_embeds: Optional[torch.Tensor] = None,
        time_ids: Optional[torch.Tensor] = None,
    ):
        if text_embeds is not None and time_ids is not None:
            added_cond_kwargs = {"text_embeds": text_embeds, "time_ids": time_ids}
        else:
            added_cond_kwargs = {}

        down_block_res_samples, mid_block_res_sample = self.controlnet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=None,
            controlnet_cond=controlnet_cond,
            conditioning_scale=conditioning_scale,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )
        return down_block_res_samples, mid_block_res_sample


class _ControlNetModel_Cross_Attention(torch.nn.Module):
    def __init__(self, controlnet: "ControlNetModel"):
        super().__init__()
        self.controlnet = controlnet

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: torch.Tensor,
        conditioning_scale,
        text_embeds: Optional[torch.Tensor] = None,
        time_ids: Optional[torch.Tensor] = None,
    ):
        if text_embeds is not None and time_ids is not None:
            added_cond_kwargs = {"text_embeds": text_embeds, "time_ids": time_ids}
        else:
            added_cond_kwargs = {}

        down_block_res_samples, mid_block_res_sample = self.controlnet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=controlnet_cond,
            conditioning_scale=conditioning_scale,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )
        return down_block_res_samples, mid_block_res_sample


class RBLNControlNetModel(RBLNModel):
    hf_library_name = "diffusers"
    auto_model_class = ControlNetModel

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        self.use_encoder_hidden_states = any(
            item[0] == "encoder_hidden_states" for item in self.rbln_config.compile_cfgs[0].input_info
        )

    @classmethod
    def wrap_model_if_needed(cls, model: torch.nn.Module, rbln_config: RBLNConfig) -> torch.nn.Module:
        use_encoder_hidden_states = False
        for down_block in model.down_blocks:
            if use_encoder_hidden_states := getattr(down_block, "has_cross_attention", False):
                break

        if use_encoder_hidden_states:
            return _ControlNetModel_Cross_Attention(model).eval()
        else:
            return _ControlNetModel(model).eval()

    @classmethod
    def update_rbln_config_using_pipe(cls, pipe: RBLNDiffusionMixin, rbln_config: Dict[str, Any]) -> Dict[str, Any]:
        rbln_vae_cls = getattr(importlib.import_module("optimum.rbln"), f"RBLN{pipe.vae.__class__.__name__}")
        rbln_unet_cls = getattr(importlib.import_module("optimum.rbln"), f"RBLN{pipe.unet.__class__.__name__}")
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
                "vae_sample_size": rbln_vae_cls.get_vae_sample_size(pipe, rbln_config),
                "unet_sample_size": rbln_unet_cls.get_unet_sample_size(pipe, rbln_config),
                "batch_size": batch_size,
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
        unet_sample_size = rbln_kwargs.get("unet_sample_size")
        vae_sample_size = rbln_kwargs.get("vae_sample_size")

        if batch_size is None:
            batch_size = 1

        if unet_sample_size is None:
            raise ValueError(
                "`rbln_unet_sample_size` (latent height, widht) must be specified (ex. unet's sample_size)"
            )

        if vae_sample_size is None:
            raise ValueError(
                "`rbln_vae_sample_size` (input image height, width) must be specified (ex. vae's sample_size)"
            )

        if max_seq_len is None:
            raise ValueError("`rbln_max_seq_len` (ex. text_encoder's max_position_embeddings )must be specified")

        input_info = [
            (
                "sample",
                [
                    batch_size,
                    model_config.in_channels,
                    unet_sample_size[0],
                    unet_sample_size[1],
                ],
                "float32",
            ),
            ("timestep", [], "float32"),
        ]

        use_encoder_hidden_states = any(element != "DownBlock2D" for element in model_config.down_block_types)
        if use_encoder_hidden_states:
            input_info.append(
                (
                    "encoder_hidden_states",
                    [batch_size, max_seq_len, model_config.cross_attention_dim],
                    "float32",
                )
            )

        input_info.append(
            (
                "controlnet_cond",
                [batch_size, 3, vae_sample_size[0], vae_sample_size[1]],
                "float32",
            )
        )
        input_info.append(("conditioning_scale", [], "float32"))

        if hasattr(model_config, "addition_embed_type") and model_config.addition_embed_type == "text_time":
            rbln_text_model_hidden_size = rbln_kwargs["text_model_hidden_size"]
            input_info.append(("text_embeds", [batch_size, rbln_text_model_hidden_size], "float32"))
            input_info.append(("time_ids", [batch_size, 6], "float32"))

        rbln_compile_config = RBLNCompileConfig(input_info=input_info)

        rbln_config = RBLNConfig(
            rbln_cls=cls.__name__,
            compile_cfgs=[rbln_compile_config],
            rbln_kwargs=rbln_kwargs,
        )

        return rbln_config

    @property
    def compiled_batch_size(self):
        return self.rbln_config.compile_cfgs[0].input_info[0][1][0]

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: torch.FloatTensor,
        conditioning_scale: torch.Tensor = 1.0,
        added_cond_kwargs: Dict[str, torch.Tensor] = {},
        **kwargs,
    ):
        sample_batch_size = sample.size()[0]
        compiled_batch_size = self.compiled_batch_size
        if sample_batch_size != compiled_batch_size and (
            sample_batch_size * 2 == compiled_batch_size or sample_batch_size == compiled_batch_size * 2
        ):
            raise ValueError(
                f"Mismatch between ControlNet's runtime batch size ({sample_batch_size}) and compiled batch size ({compiled_batch_size}). "
                "This may be caused by the 'guidance scale' parameter, which doubles the runtime batch size in Stable Diffusion. "
                "Adjust the batch size during compilation or modify the 'guidance scale' to match the compiled batch size.\n\n"
                "For details, see: https://docs.rbln.ai/software/optimum/model_api.html#stable-diffusion"
            )

        added_cond_kwargs = {} if added_cond_kwargs is None else added_cond_kwargs
        if self.use_encoder_hidden_states:
            output = super().forward(
                sample.contiguous(),
                timestep.float(),
                encoder_hidden_states,
                controlnet_cond,
                torch.tensor(conditioning_scale),
                **added_cond_kwargs,
            )
        else:
            output = super().forward(
                sample.contiguous(),
                timestep.float(),
                controlnet_cond,
                torch.tensor(conditioning_scale),
                **added_cond_kwargs,
            )
        down_block_res_samples = output[:-1]
        mid_block_res_sample = output[-1]

        return down_block_res_samples, mid_block_res_sample
