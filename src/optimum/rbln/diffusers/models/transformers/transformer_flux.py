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
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import torch
import numpy as np
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from transformers import PretrainedConfig

from ....modeling import RBLNModel
from ....modeling_config import RBLNCompileConfig, RBLNConfig
from ...modeling_diffusers import RBLNDiffusionMixin


if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer

logger = logging.getLogger(__name__)


class FluxTransformer2DModelWrapper(torch.nn.Module):
    def __init__(self, model: "FluxTransformer2DModel", rbln_img_height, rbln_img_width, rbln_max_seq_len, vae_scale_factor) -> None:
        super().__init__()
        # self.model = model
        txt_ids = torch.zeros(rbln_max_seq_len, 3)
        height = 2 * (int(rbln_img_height) // vae_scale_factor)
        width = 2 * (int(rbln_img_width) // vae_scale_factor)
        latent_image_ids = torch.zeros(height // 2, width // 2, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]
        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape
        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        ids = torch.cat((txt_ids, latent_image_ids), dim=0)
        self.image_rotary_emb = model.pos_embed(ids)
        self.x_embedder = model.x_embedder
        self.context_embedder = model.context_embedder
        self.transformer_blocks = model.transformer_blocks
        self.single_transformer_blocks = model.single_transformer_blocks
        self.norm_out = model.norm_out
        self.proj_out = model.proj_out
        self.time_text_embed = model.time_text_embed

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        pooled_projections: torch.FloatTensor = None,
        timestep: torch.LongTensor = None,
        # img_ids: torch.Tensor = None,
        # txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = False,
        controlnet_blocks_repeat: bool = False,
    ):
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None
            
        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        for index_block, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=self.image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

            # controlnet residual
            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                # For Xlabs ControlNet.
                if controlnet_blocks_repeat:
                    hidden_states = (
                        hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                    )
                else:
                    hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        for index_block, block in enumerate(self.single_transformer_blocks):
            hidden_states = block(
                hidden_states=hidden_states,
                temb=temb,
                image_rotary_emb=self.image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

            # controlnet residual
            if controlnet_single_block_samples is not None:
                interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                    hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                    + controlnet_single_block_samples[index_block // interval_control]
                )

        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
        
        
        # return self.model(
        #     hidden_states=hidden_states,
        #     encoder_hidden_states=encoder_hidden_states,
        #     pooled_projections=pooled_projections,
        #     timestep=timestep,
        #     # img_ids=img_ids,
        #     # txt_ids=txt_ids,
        #     return_dict=False,
        # )


class RBLNFluxTransformer2DModel(RBLNModel):
    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)

    @classmethod
    def wrap_model_if_needed(cls, model: torch.nn.Module, rbln_config: RBLNConfig) -> torch.nn.Module:
        height = rbln_config.model_cfg.get("sample_size") * rbln_config.model_cfg.get("vae_scale_factor")
        width = rbln_config.model_cfg.get("sample_size") * rbln_config.model_cfg.get("vae_scale_factor")
        max_seq_len = rbln_config.model_cfg.get("max_sequence_length")
        vae_scale_factor = rbln_config.model_cfg.get("vae_scale_factor")
        return FluxTransformer2DModelWrapper(model, height, width, max_seq_len, vae_scale_factor).eval()

    # hidden_states -> latent shape -> 1, 4096, 64
        # timestep -> 1
        # guidance -> config.json의 guidance_embeds에 따라 true/false -> FLUX.1-schnell에서는 false = None
        # pooled_projections -> 1, 768 (batch_size, config.pooled_projection_dim)
        # encoder_hidden_states -> 1, 512, 4096 (batch_size, seq_len, joint_attention_dim)
        # txt_ids -> 512,3 (seq_len, 3)
        # img_ids -> 4096,3 (latent.shape[1], 3)
        # joint_attention_kwargs -> self.joint_attention_kwargs

    @classmethod
    def update_rbln_config_using_pipe(cls, pipe: RBLNDiffusionMixin, rbln_config: Dict[str, Any]) -> Dict[str, Any]:
        sample_size = rbln_config.get("sample_size", pipe.default_sample_size)
        img_width = rbln_config.get("img_width")
        img_height = rbln_config.get("img_height")
        
        if (img_height is None) ^ (img_width is None):
            raise RuntimeError

        elif img_height and img_width:
            sample_size = img_height // pipe.vae_scale_factor, img_width // pipe.vae_scale_factor

        max_sequence_length = pipe.tokenizer_2.model_max_length
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
        tensor_parallel_size = rbln_config.get("tensor_parallel_size")
        
        return {
            "batch_size": batch_size,
            "max_sequence_length": max_sequence_length,
            "sample_size": sample_size,
            "vae_scale_factor": pipe.vae_scale_factor,
            "tensor_parallel_size": tensor_parallel_size,
        }

    @classmethod
    def _get_rbln_config(
        cls,
        preprocessors: Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"],
        model_config: "PretrainedConfig",
        rbln_kwargs: Dict[str, Any] = {},
    ) -> RBLNConfig:
        rbln_batch_size = rbln_kwargs.get("batch_size", None)

        sample_size = rbln_kwargs.get("sample_size", None)
        vae_scale_factor = rbln_kwargs.get("vae_scale_factor", None)

        if isinstance(sample_size, int):
            sample_size = (sample_size, sample_size)

        rbln_max_seqeunce_length = rbln_kwargs.get("max_sequence_length")
        if rbln_max_seqeunce_length is None:
            raise ValueError("rbln_max_seqeunce_length should be specified.")

        #FIXME (if no width height for users -> default)
        base_height = int(sample_size[0]) * vae_scale_factor
        base_width = int(sample_size[1]) * vae_scale_factor
        
        # prepare_latents function
        height = 2 * (int(base_height) // vae_scale_factor)
        width = 2 * (int(base_width) // vae_scale_factor)
        latent_shape = (height // 2) * (width // 2)
        num_channels_latents = model_config.in_channels // 4

        input_info = [
            (
                "hidden_states",
                [
                    rbln_batch_size,
                    latent_shape,
                    num_channels_latents * 4,
                ],
                "float32",
            ),
            (
                "encoder_hidden_states",
                [
                    rbln_batch_size,
                    rbln_max_seqeunce_length,
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
            # ("img_ids", [latent_shape, 3], "float32"),
            # ("txt_ids", [rbln_max_seqeunce_length, 3], "float32"),
        ]
        
        if model_config.guidance_embeds:
            input_info.extend(
                    [
                        ("guidance", [rbln_batch_size], "float32"),
                    ]
                )

        rbln_compile_config = RBLNCompileConfig(input_info=input_info)

        rbln_config = RBLNConfig(
            rbln_cls=cls.__name__,
            compile_cfgs=[rbln_compile_config],
            rbln_kwargs=rbln_kwargs,
        )

        rbln_config.model_cfg.update({"batch_size": rbln_batch_size})

        return rbln_config

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        # img_ids: torch.Tensor = None,
        # txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
    ):
        output = super().forward(hidden_states, encoder_hidden_states, pooled_projections, timestep, guidance)
        return Transformer2DModelOutput(sample=output)