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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
from transformers import PretrainedConfig

from ....modeling import RBLNModel
from ....modeling_config import RBLNCompileConfig, RBLNConfig
from ...modeling_diffusers import RBLNDiffusionMixin


if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer

logger = logging.getLogger(__name__)


class SD3Transformer2DModelWrapper(torch.nn.Module):
    def __init__(self, model: "SD3Transformer2DModel") -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        pooled_projections: torch.FloatTensor = None,
        timestep: torch.LongTensor = None,
        # need controlnet support?
        block_controlnet_hidden_states: List = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ):
        return self.model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timestep,
            return_dict=False,
        )


class RBLNSD3Transformer2DModel(RBLNModel):
    hf_library_name = "diffusers"

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)

    @classmethod
    def wrap_model_if_needed(cls, model: torch.nn.Module, rbln_config: RBLNConfig) -> torch.nn.Module:
        return SD3Transformer2DModelWrapper(model).eval()

    @classmethod
    def update_rbln_config_using_pipe(cls, pipe: RBLNDiffusionMixin, rbln_config: Dict[str, Any]) -> Dict[str, Any]:
        sample_size = rbln_config.get("sample_size", pipe.default_sample_size)
        img_width = rbln_config.get("img_width")
        img_height = rbln_config.get("img_height")

        if (img_width is None) ^ (img_height is None):
            raise RuntimeError

        elif img_width and img_height:
            sample_size = img_height // pipe.vae_scale_factor, img_width // pipe.vae_scale_factor

        prompt_max_length = rbln_config.get("max_sequence_length", 256)
        prompt_embed_length = pipe.tokenizer_max_length + prompt_max_length

        batch_size = rbln_config.get("batch_size")
        if not batch_size:
            do_classifier_free_guidance = rbln_config.get("guidance_scale", 5.0) > 1.0
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
                "batch_size": batch_size,
                "prompt_embed_length": prompt_embed_length,
                "sample_size": sample_size,
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
        rbln_batch_size = rbln_kwargs.get("batch_size", None)

        sample_size = rbln_kwargs.get("sample_size", model_config.sample_size)
        if isinstance(sample_size, int):
            sample_size = (sample_size, sample_size)

        rbln_prompt_embed_length = rbln_kwargs.get("prompt_embed_length")
        if rbln_prompt_embed_length is None:
            raise ValueError("rbln_prompt_embed_length should be specified.")

        input_info = [
            (
                "hidden_states",
                [
                    rbln_batch_size,
                    model_config.in_channels,
                    sample_size[0],
                    sample_size[1],
                ],
                "float32",
            ),
            (
                "encoder_hidden_states",
                [
                    rbln_batch_size,
                    rbln_prompt_embed_length,
                    model_config.joint_attention_dim,
                ],
                "float32",
            ),
            (
                "pooled_projections",
                [
                    rbln_batch_size,
                    model_config.pooled_projection_dim,
                ],
                "float32",
            ),
            ("timestep", [rbln_batch_size], "float32"),
        ]

        rbln_compile_config = RBLNCompileConfig(input_info=input_info)

        rbln_config = RBLNConfig(
            rbln_cls=cls.__name__,
            compile_cfgs=[rbln_compile_config],
            rbln_kwargs=rbln_kwargs,
        )

        rbln_config.model_cfg.update({"batch_size": rbln_batch_size})

        return rbln_config

    @property
    def compiled_batch_size(self):
        return self.rbln_config.compile_cfgs[0].input_info[0][1][0]

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        pooled_projections: torch.FloatTensor = None,
        timestep: torch.LongTensor = None,
        block_controlnet_hidden_states: List = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        **kwargs,
    ):
        sample_batch_size = hidden_states.size()[0]
        compiled_batch_size = self.compiled_batch_size
        if sample_batch_size != compiled_batch_size and (
            sample_batch_size * 2 == compiled_batch_size or sample_batch_size == compiled_batch_size * 2
        ):
            raise ValueError(
                f"Mismatch between Transformers' runtime batch size ({sample_batch_size}) and compiled batch size ({compiled_batch_size}). "
                "This may be caused by the 'guidance scale' parameter, which doubles the runtime batch size in Stable Diffusion. "
                "Adjust the batch size during compilation or modify the 'guidance scale' to match the compiled batch size.\n\n"
                "For details, see: https://docs.rbln.ai/software/optimum/model_api.html#stable-diffusion"
            )

        sample = super().forward(hidden_states, encoder_hidden_states, pooled_projections, timestep)
        return Transformer2DModelOutput(sample=sample)
