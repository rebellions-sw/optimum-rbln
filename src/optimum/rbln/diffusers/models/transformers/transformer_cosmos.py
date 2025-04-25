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

import math
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple, Union

import numpy as np
import torch
from diffusers import CosmosTransformer3DModel
from diffusers.models.embeddings import Timesteps
from diffusers.pipelines.cosmos.pipeline_cosmos import retrieve_timesteps

from ....configuration_utils import RBLNCompileConfig, RBLNModelConfig
from ....modeling import RBLNModel
from ....utils.logging import get_logger
from ...configurations import RBLNCosmosTransformer3DModelConfig


if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer, PretrainedConfig, PreTrainedModel

    from ...modeling_diffusers import RBLNCosmosTransformer3DModelConfig, RBLNDiffusionMixin


logger = get_logger(__name__)


class CosmosTransformer3DModelWrapper(torch.nn.Module):
    def __init__(
        self,
        model: CosmosTransformer3DModel,
        num_latent_frames: int = 16,
        latent_height: int = 88,
        latent_width: int = 160,
        hidden_size: int = 4096,
        fps: int = 30,
    ) -> None:
        super().__init__()
        self.model = model
        self.set_rope(num_latent_frames, latent_height, latent_width, fps)
        self.set_learnable_pos_emb(num_latent_frames, latent_height, latent_width, hidden_size)
        self.set_time_embed()

    def set_rope(self, num_latent_frames, latent_height, latent_width, fps):
        self.model.rope = RBLNCosmosRotaryPosEmbed(
            self.model.rope, num_latent_frames, latent_height, latent_width, fps
        )

    def set_learnable_pos_emb(self, num_latent_frames, latent_height, latent_width, hidden_size):
        native_lpoe = self.model.learnable_pos_embed
        patch_size = native_lpoe.patch_size
        eps = native_lpoe.eps
        batch_size = 1
        pe_size = [num_latent_frames // patch_size[0], latent_height // patch_size[1], latent_width // patch_size[2]]
        emb_t = native_lpoe.pos_emb_t[: pe_size[0]][None, :, None, None, :].repeat(
            batch_size, 1, pe_size[1], pe_size[2], 1
        )
        emb_h = native_lpoe.pos_emb_h[: pe_size[1]][None, None, :, None, :].repeat(
            batch_size, pe_size[0], 1, pe_size[2], 1
        )
        emb_w = native_lpoe.pos_emb_w[: pe_size[2]][None, None, None, :, :].repeat(
            batch_size, pe_size[0], pe_size[1], 1, 1
        )

        self.model.learnable_pos_embed = RBLNCosmosLearnablePositionalEmbed(
            emb_t.data, emb_h.data, emb_w.data, hidden_size, eps
        ).train(self.model.training)

    def set_time_embed(self):
        self.model.time_embed = RBLNCosmosEmbedding(self.model.time_embed)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        fps: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        condition_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        return self.model(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            fps=fps,
            padding_mask=padding_mask,
            return_dict=False,
        )


class RBLNCosmosRotaryPosEmbed(torch.nn.Module):
    def __init__(
        self,
        rope,
        num_latent_frames,
        latent_height,
        latent_width,
        fps=None,
    ) -> None:
        super().__init__()

        self.max_size = rope.max_size
        self.patch_size = rope.patch_size
        self.base_fps = rope.base_fps

        self.dim_h = rope.dim_h
        self.dim_w = rope.dim_w
        self.dim_t = rope.dim_t

        self.h_ntk_factor = rope.h_ntk_factor
        self.w_ntk_factor = rope.w_ntk_factor
        self.t_ntk_factor = rope.t_ntk_factor

        seq = torch.arange(max(self.max_size), dtype=torch.float32)
        pe_size = [
            num_latent_frames // self.patch_size[0],
            latent_height // self.patch_size[1],
            latent_width // self.patch_size[2],
        ]

        h_theta = 10000.0 * self.h_ntk_factor
        w_theta = 10000.0 * self.w_ntk_factor
        t_theta = 10000.0 * self.t_ntk_factor

        dim_h_range = torch.arange(0, self.dim_h, 2, dtype=torch.float32)[: (self.dim_h // 2)] / self.dim_h
        dim_w_range = torch.arange(0, self.dim_w, 2, dtype=torch.float32)[: (self.dim_w // 2)] / self.dim_w
        dim_t_range = torch.arange(0, self.dim_t, 2, dtype=torch.float32)[: (self.dim_t // 2)] / self.dim_t

        h_spatial_freqs = 1.0 / (h_theta**dim_h_range)
        w_spatial_freqs = 1.0 / (w_theta**dim_w_range)
        temporal_freqs = 1.0 / (t_theta**dim_t_range)

        emb_h = torch.outer(seq[: pe_size[1]], h_spatial_freqs)[None, :, None, :].repeat(pe_size[0], 1, pe_size[2], 1)
        emb_w = torch.outer(seq[: pe_size[2]], w_spatial_freqs)[None, None, :, :].repeat(pe_size[0], pe_size[1], 1, 1)

        if fps is None:
            emb_t = torch.outer(seq[: pe_size[0]], temporal_freqs)
        else:
            emb_t = torch.outer(seq[: pe_size[0]] / fps * self.base_fps, temporal_freqs)

        emb_t = emb_t[:, None, None, :].repeat(1, pe_size[1], pe_size[2], 1)
        freqs = torch.cat([emb_t, emb_h, emb_w] * 2, dim=-1).flatten(0, 2).float()
        self.cos = torch.cos(freqs)
        self.sin = torch.sin(freqs)

    def forward(self, hidden_states: torch.Tensor, fps: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cos.to(dtype=hidden_states.dtype), self.sin.to(dtype=hidden_states.dtype)


class RBLNCosmosLearnablePositionalEmbed(torch.nn.Module):
    def __init__(
        self,
        emb_t,
        emb_h,
        emb_w,
        hidden_size,
        eps,
    ) -> None:
        super().__init__()

        self.emb_t = emb_t
        self.emb_h = emb_h
        self.emb_w = emb_w
        self.eps = eps
        self.alpha = np.sqrt(1 / hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        emb = self.emb_t + self.emb_h + self.emb_w
        emb = emb.flatten(1, 3)
        norm = torch.linalg.vector_norm(emb, dim=-1, keepdim=True, dtype=torch.float32)
        norm = torch.add(self.eps, norm, alpha=self.alpha)
        return (emb / norm).type_as(hidden_states)


class RBLNCosmosEmbedding(torch.nn.Module):
    def __init__(self, time_embed):
        super().__init__()
        self.time_proj = time_embed.time_proj
        self.t_embedder = time_embed.t_embedder
        self.norm = time_embed.norm

    def forward(self, hidden_states, timesteps):
        timesteps_proj = timesteps.type_as(hidden_states)
        temb = self.t_embedder(timesteps_proj)
        embedded_timestep = self.norm(timesteps_proj)
        return temb, embedded_timestep


class RBLNCosmosTransformer3DModel(RBLNModel):
    hf_library_name = "diffusers"
    auto_model_class = CosmosTransformer3DModel

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        artifacts = torch.load(self.model_save_dir / self.subfolder / "torch_artifacts.pth", weights_only=False)
        num_channels = artifacts["num_channels"]
        flip_sin_to_cos = artifacts["flip_sin_to_cos"]
        downscale_freq_shift = artifacts["downscale_freq_shift"]
        scale = artifacts["scale"]
        self.time_proj = Timesteps(
            num_channels=num_channels,
            flip_sin_to_cos=flip_sin_to_cos,
            downscale_freq_shift=downscale_freq_shift,
            scale=scale,
        )

    def get_time_embed_table(self, scheduler, num_inference_steps):
        timesteps = retrieve_timesteps(scheduler, num_inference_steps)[0]

        embedding_dim = self.time_proj.num_channels
        flip_sin_to_cos = self.time_proj.flip_sin_to_cos
        downscale_freq_shift = self.time_proj.downscale_freq_shift
        scale = self.time_proj.scale

        half_dim = embedding_dim // 2
        max_period = 10000
        exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32)
        exponent = exponent / (half_dim - downscale_freq_shift)

        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]

        emb = scale * emb
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        if flip_sin_to_cos:
            emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

        if embedding_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))

        emb_dict = {}
        for timestep, e in zip(timesteps, emb):
            emb_dict[timestep.item()] = e.reshape(1, -1)

        self._emb_cached = emb_dict

    def time_embed_table(self, timestep):
        emb = self._emb_cached[timestep[0].item()]
        emb = emb.repeat(timestep.shape[0], 1)
        return emb

    @classmethod
    def wrap_model_if_needed(cls, model: torch.nn.Module, rbln_config: RBLNModelConfig) -> torch.nn.Module:
        num_latent_frames = rbln_config.num_latent_frames
        latent_height = rbln_config.latent_height
        latent_width = rbln_config.latent_width
        hidden_size = rbln_config.hidden_size
        fps = rbln_config.fps
        return CosmosTransformer3DModelWrapper(
            model=model,
            num_latent_frames=num_latent_frames,
            latent_height=latent_height,
            latent_width=latent_width,
            hidden_size=hidden_size,
            fps=fps,
        ).eval()

    @classmethod
    def update_rbln_config_using_pipe(
        cls, pipe: "RBLNDiffusionMixin", rbln_config: "RBLNCosmosTransformer3DModelConfig", submodule_name: str
    ) -> RBLNCosmosTransformer3DModelConfig:
        rbln_config.num_channel_latents = pipe.transformer.config.in_channels
        rbln_config.num_latent_frames = (rbln_config.num_frames - 1) // pipe.vae_scale_factor_temporal + 1
        rbln_config.latent_height = rbln_config.height // pipe.vae_scale_factor_temporal
        rbln_config.latent_width = rbln_config.width // pipe.vae_scale_factor_temporal
        rbln_config.hidden_size = (
            pipe.transformer.config.num_attention_heads * pipe.transformer.config.attention_head_dim
        )
        rbln_config.embedding_dim = pipe.text_encoder.encoder.embed_tokens.embedding_dim
        rbln_config.time_proj_num_channels = pipe.transformer.time_embed.time_proj.num_channels

        return rbln_config

    @classmethod
    def save_torch_artifacts(
        cls,
        model: "PreTrainedModel",
        save_dir_path: Path,
        subfolder: str,
        rbln_config: RBLNModelConfig,
    ):
        time_proj = model.time_embed.time_proj
        save_dict = {}
        save_dict["num_channels"] = time_proj.num_channels
        save_dict["flip_sin_to_cos"] = time_proj.flip_sin_to_cos
        save_dict["downscale_freq_shift"] = time_proj.downscale_freq_shift
        save_dict["scale"] = time_proj.scale
        torch.save(save_dict, save_dir_path / subfolder / "torch_artifacts.pth")

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"],
        model: "PreTrainedModel",
        model_config: "PretrainedConfig",
        rbln_config: "RBLNCosmosTransformer3DModelConfig",
    ) -> RBLNCosmosTransformer3DModelConfig:
        input_info = [
            (
                "hidden_states",
                [
                    rbln_config.batch_size,
                    rbln_config.num_channel_latents,
                    rbln_config.num_latent_frames,
                    rbln_config.latent_height,
                    rbln_config.latent_width,
                ],
                "float32",
            ),
            ("timestep", [rbln_config.batch_size, rbln_config.time_proj_num_channels], "float32"),
            (
                "encoder_hidden_states",
                [
                    rbln_config.batch_size,
                    rbln_config.max_sequence_length,
                    rbln_config.embedding_dim,
                ],
                "float32",
            ),
            (
                "padding_mask",
                [
                    1,
                    1,
                    rbln_config.height,
                    rbln_config.width,
                ],
                "float32",
            ),
        ]

        compile_config = RBLNCompileConfig(input_info=input_info)
        rbln_config.set_compile_cfgs([compile_config])
        return rbln_config

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        fps: Optional[int] = None,
        condition_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        if not hasattr(self, "_emb_cached"):
            raise RuntimeError(
                "To run 'RBLNCosmosTransformer3DModel', the method 'get_time_embed_table' should be executed with the scheduler and 'num_inference_steps'."
            )

        timestep = self.time_embed_table(timestep)
        output = self.model[0].forward(
            hidden_states,
            timestep,
            encoder_hidden_states,
            padding_mask,
        )
        return output
