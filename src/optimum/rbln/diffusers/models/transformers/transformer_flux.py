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

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import numpy as np
import torch
from diffusers.models.embeddings import (
    CombinedTimestepGuidanceTextProjEmbeddings,
    CombinedTimestepTextProjEmbeddings,
    FluxPosEmbed,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from transformers import PretrainedConfig

from ....configuration_utils import RBLNCompileConfig, RBLNModelConfig
from ....modeling import RBLNModel
from ....utils.logging import get_logger
from ...configurations import RBLNFluxTransformer2DModelConfig


if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer, PreTrainedModel

    from ...modeling_diffusers import RBLNDiffusionMixin, RBLNDiffusionMixinConfig

logger = get_logger(__name__)


class FluxTransformer2DModelWrapper(torch.nn.Module):
    def __init__(self, model: "FluxTransformer2DModel") -> None:
        super().__init__()
        self.context_embedder = model.context_embedder
        self.x_embedder = model.x_embedder
        self.transformer_blocks = model.transformer_blocks
        self.single_transformer_blocks = model.single_transformer_blocks
        self.norm_out = model.norm_out
        self.proj_out = model.proj_out

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        image_rotary_emb_0: torch.Tensor = None,
        image_rotary_emb_1: torch.Tensor = None,
        temb: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
    ):
        hidden_states = self.x_embedder(hidden_states)
        timestep = timestep.to(hidden_states.dtype) * 1000

        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        for index_block, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=[image_rotary_emb_0, image_rotary_emb_1],
                joint_attention_kwargs=joint_attention_kwargs,
            )
            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                # For Xlabs ControlNet.
                if controlnet_blocks_repeat:
                    hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]
                else:
                    hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

        for index_block, block in enumerate(self.single_transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=[image_rotary_emb_0, image_rotary_emb_1],
                joint_attention_kwargs=joint_attention_kwargs,
            )
            if controlnet_single_block_samples is not None:
                interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states = hidden_states + controlnet_single_block_samples[index_block // interval_control]

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        return (output,)


class RBLNFluxTransformer2DModel(RBLNModel):
    """
    RBLN implementation of FluxTransformer2DModel for diffusion models like Flux.

    The FluxTransformer2DModel takes text and/or image embeddings from encoders (like CLIP) and
    maps them to a shared latent space that guides the diffusion process to generate the desired image.

    This class inherits from [`RBLNModel`]. Check the superclass documentation for the generic methods
    the library implements for all its models.
    """

    hf_library_name = "diffusers"
    auto_model_class = FluxTransformer2DModel
    _output_class = Transformer2DModelOutput

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        artifacts = torch.load(self.model_save_dir / self.subfolder / "torch_artifacts.pth", weights_only=False)
        inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        text_time_guidance_cls = (
            CombinedTimestepGuidanceTextProjEmbeddings
            if self.config.guidance_embeds
            else CombinedTimestepTextProjEmbeddings
        )
        self.time_text_embed = text_time_guidance_cls(
            embedding_dim=inner_dim, pooled_projection_dim=self.config.pooled_projection_dim
        )
        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=self.config.axes_dims_rope)
        self.time_text_embed.load_state_dict(artifacts["time_text_embed"])
        self.pos_embed.load_state_dict(artifacts["pos_embed"])

    @classmethod
    def save_torch_artifacts(
        cls,
        model: "PreTrainedModel",
        save_dir_path: Path,
        subfolder: str,
        rbln_config: RBLNModelConfig,
    ):
        save_dict = {}
        save_dict["time_text_embed"] = model.time_text_embed.state_dict()
        save_dict["pos_embed"] = model.pos_embed.state_dict()

        torch.save(save_dict, save_dir_path / subfolder / "torch_artifacts.pth")

    @classmethod
    def wrap_model_if_needed(cls, model: torch.nn.Module, rbln_config: RBLNModelConfig) -> torch.nn.Module:
        return FluxTransformer2DModelWrapper(model).eval()

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

        prompt_embed_length = pipe.tokenizer_max_length + rbln_config.transformer.max_sequence_length
        rbln_config.transformer.prompt_embed_length = prompt_embed_length
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

        input_info = [
            (
                "hidden_states",
                [
                    rbln_config.batch_size,
                    (rbln_config.sample_size[0] // 2) * (rbln_config.sample_size[1] // 2),
                    model_config.in_channels,
                ],
                "float32",
            ),
            (
                "encoder_hidden_states",
                [
                    rbln_config.batch_size,
                    rbln_config.max_sequence_length,
                    model_config.joint_attention_dim,
                ],
                "float32",
            ),
            ("timestep", [rbln_config.batch_size], "float32"),
            (
                "image_rotary_emb_0",
                [
                    rbln_config.max_sequence_length
                    + (rbln_config.sample_size[0] // 2) * (rbln_config.sample_size[1] // 2),
                    model_config.attention_head_dim,
                ],
                "float32",
            ),
            (
                "image_rotary_emb_1",
                [
                    rbln_config.max_sequence_length
                    + (rbln_config.sample_size[0] // 2) * (rbln_config.sample_size[1] // 2),
                    model_config.attention_head_dim,
                ],
                "float32",
            ),
            (
                "temb",
                [rbln_config.batch_size, model_config.attention_head_dim * model_config.num_attention_heads],
                "float32",
            ),
        ]

        compile_config = RBLNCompileConfig(input_info=input_info)
        rbln_config.set_compile_cfgs([compile_config])
        return rbln_config

    @property
    def compiled_batch_size(self):
        return self.rbln_config.compile_cfgs[0].input_info[0][1][0]

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
    ):
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )

        if txt_ids.ndim == 3:
            logger.warning(
                "Passing `txt_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            logger.warning(
                "Passing `img_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        output = self.model[0].forward(
            hidden_states,
            encoder_hidden_states,
            timestep,
            image_rotary_emb[0],
            image_rotary_emb[1],
            temb,
            joint_attention_kwargs,
            controlnet_block_samples,
            controlnet_single_block_samples,
            return_dict,
            controlnet_blocks_repeat,
        )

        return Transformer2DModelOutput(sample=output)
