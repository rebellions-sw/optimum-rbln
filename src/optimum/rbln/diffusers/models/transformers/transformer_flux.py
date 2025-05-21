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

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import numpy as np
import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from transformers import PretrainedConfig

from ....configuration_utils import RBLNCompileConfig, RBLNModelConfig
from ....modeling import RBLNModel
from ...configurations import RBLNFluxTransformer2DModelConfig
from ...modeling_diffusers import RBLNDiffusionMixin


if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer, PreTrainedModel

    from ...modeling_diffusers import RBLNDiffusionMixin, RBLNDiffusionMixinConfig

logger = logging.getLogger(__name__)


class FluxTransformer2DModelWrapper(torch.nn.Module):
    def __init__(self, model: "FluxTransformer2DModel", ids: torch.tensor) -> None:
        super().__init__()
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
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = False,
        controlnet_blocks_repeat: bool = False,
    ):
        # TODO(kblee): need to support lora (?)
        # if joint_attention_kwargs is not None:
        #     joint_attention_kwargs = joint_attention_kwargs.copy()
        #     lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        # else:
        #     lora_scale = 1.0

        # hidden_states = self.x_embedder(hidden_states)

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
        # encoder_hidden_states = self.context_embedder(encoder_hidden_states)

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


class RBLNFluxTransformer2DModel(RBLNModel):
    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        self.x_embedder = torch.load(self.model_save_dir / self.subfolder / "x_embedder.pth", weights_only=False)
        self.context_embedder = torch.load(
            self.model_save_dir / self.subfolder / "context_embedder.pth", weights_only=False
        )

    @classmethod
    def save_torch_artifacts(
        cls,
        model: "PreTrainedModel",
        save_dir_path: Path,
        subfolder: str,
        rbln_config: RBLNModelConfig,
    ):
        save_dict = {}
        save_dict["context_embedder"] = model.context_embedder.state_dict()
        save_dict["x_embedder"] = model.x_embedder.state_dict()

        torch.save(save_dict, save_dir_path / subfolder / "torch_artifacts.pth")

    @classmethod
    def wrap_model_if_needed(cls, model: torch.nn.Module, rbln_config: RBLNModelConfig) -> torch.nn.Module:
        txt_ids = torch.zeros(rbln_config.max_sequence_length, 3)
        height = 2 * (int(rbln_config.sample_size[0] * rbln_config.vae_scale_factor) // rbln_config.vae_scale_factor)
        width = 2 * (int(rbln_config.sample_size[1] * rbln_config.vae_scale_factor) // rbln_config.vae_scale_factor)
        latent_image_ids = torch.zeros(height // 2, width // 2, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]
        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape
        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )
        ids = torch.cat((txt_ids, latent_image_ids), dim=0)
        return FluxTransformer2DModelWrapper(model, ids).eval()

    @classmethod
    def update_rbln_config_using_pipe(
        cls, pipe: "RBLNDiffusionMixin", rbln_config: "RBLNDiffusionMixinConfig", submodule_name: str
    ) -> "RBLNDiffusionMixinConfig":
        if rbln_config.sample_size is None:
            if rbln_config.image_size is not None:
                rbln_config.transformer.sample_size = (
                    rbln_config.image_size[0] // pipe.vae_scale_factor,
                    rbln_config.image_size[1] // pipe.vae_scale_factor,
                )
            else:
                rbln_config.transformer.sample_size = pipe.default_sample_size

        rbln_config.transformer.vae_scale_factor = pipe.vae_scale_factor

        return rbln_config

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"],
        model: "PreTrainedModel",
        model_config: "PretrainedConfig",
        rbln_config: RBLNFluxTransformer2DModelConfig,
    ) -> RBLNFluxTransformer2DModelConfig:
        if rbln_config.sample_size is None:
            rbln_config.sample_size = model_config.sample_size

        if isinstance(rbln_config.sample_size, int):
            rbln_config.sample_size = (rbln_config.sample_size, rbln_config.sample_size)

        latent_shape = ((2 * int(rbln_config.sample_size[0])) // 2) * (2 * int(rbln_config.sample_size[1]) // 2)
        # num_channels_latents = model_config.in_channels // 4

        input_info = [
            (
                "hidden_states",
                [
                    rbln_config.batch_size,
                    latent_shape,
                    # num_channels_latents * 4,
                    model_config.num_attention_heads * model_config.attention_head_dim,
                ],
                "float32",
            ),
            (
                "encoder_hidden_states",
                [
                    rbln_config.batch_size,
                    rbln_config.max_sequence_length,
                    # model_config.joint_attention_dim,
                    model_config.num_attention_heads * model_config.attention_head_dim,
                ],
                "float32",
            ),
            (
                "pooled_projections",
                [
                    rbln_config.batch_size,
                    model_config.pooled_projection_dim,
                ],
                "float32",
            ),
            ("timestep", [rbln_config.batch_size], "float32"),
        ]

        if model_config.guidance_embeds:
            input_info.extend(
                [
                    ("guidance", [rbln_config.batch_size], "float32"),
                ]
            )

        compile_config = RBLNCompileConfig(input_info=input_info)
        rbln_config.set_compile_cfgs([compile_config])
        return rbln_config

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
    ):
        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        output = self.model[0].forward(hidden_states, encoder_hidden_states, pooled_projections, timestep, guidance)
        return Transformer2DModelOutput(sample=output)
