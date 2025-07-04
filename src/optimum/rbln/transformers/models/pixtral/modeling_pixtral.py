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
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union

import rebel
import torch
import torch.nn as nn
from transformers import PixtralVisionConfig, PixtralVisionModel
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import no_init_weights
from transformers.models.pixtral.modeling_pixtral import (
    PixtralRMSNorm,
    PixtralRotaryEmbedding,
    generate_block_attention_mask,
    position_ids_in_meshgrid,
)

from ....configuration_utils import RBLNCompileConfig, RBLNModelConfig
from ....modeling import RBLNModel
from ....utils.logging import get_logger
from ....utils.runtime_utils import RBLNPytorchRuntime
from .configuration_pixtral import RBLNPixtralVisionModelConfig
from .pixtral_architecture import PixtralAttention


logger = get_logger(__name__)

if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer, PreTrainedModel

    from ....diffusers.modeling_diffusers import RBLNDiffusionMixin, RBLNDiffusionMixinConfig


class RBLNRuntimePixtralVisionModel(RBLNPytorchRuntime):
    mandatory_members = ["main_input_name"]

    def __init__(
        self,
        runtime: rebel.Runtime,
        config: PixtralVisionConfig,
        rbln_config: RBLNPixtralVisionModelConfig,
        **kwargs: Any,
    ) -> None:
        super().__init__(runtime, **kwargs)
        self.patch_positional_embedding = PixtralRotaryEmbedding(config)
        self.patch_size = config.patch_size
        self.image_size = config.image_size
        self.hidden_size = config.hidden_size
        self.max_image_size = rbln_config.max_image_size

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_sizes: torch.Tensor,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        max_height, max_width = self.max_image_size
        max_patches = pixel_values.shape[0] * (max_height // self.patch_size) * (max_width // self.patch_size)
        patch_embeds_conv = self.patch_conv(pixel_values)
        patch_embeds_list = [
            embed[..., : (size[0] // self.patch_size), : (size[1] // self.patch_size)]
            for embed, size in zip(patch_embeds_conv, image_sizes)
        ]

        real_patch_embeds = torch.cat([p.flatten(1).T for p in patch_embeds_list], dim=0).unsqueeze(0)
        num_real_patches = real_patch_embeds.shape[1]

        if num_real_patches > max_patches:
            raise ValueError(
                f"The number of real patches ({num_real_patches}) exceeds the "
                f"configured max_total_patches ({max_patches}). "
                f"Please increase max_total_patches in the model config."
            )

        num_padding_patches = max_patches - num_real_patches
        padding_embeds = torch.zeros(
            (1, num_padding_patches, self.hidden_size),
            dtype=real_patch_embeds.dtype,
        )
        patch_embeds = torch.cat([real_patch_embeds, padding_embeds], dim=1)
        patch_embeds = self.ln_pre(patch_embeds)

        real_position_ids = position_ids_in_meshgrid(patch_embeds_list, max_width=self.image_size // self.patch_size)

        padding_position_ids = torch.zeros((num_padding_patches,), dtype=real_position_ids.dtype)
        position_ids = torch.cat([real_position_ids, padding_position_ids], dim=0)
        position_embeddings = self.patch_positional_embedding(patch_embeds, position_ids)

        d_min = torch.finfo(patch_embeds.dtype).min
        real_patch_counts = [p.shape[-2] * p.shape[-1] for p in patch_embeds_list]

        dummy_tensor_for_mask = torch.empty(patch_embeds.shape[0], num_real_patches, 1, dtype=patch_embeds.dtype)
        real_attention_mask = generate_block_attention_mask(real_patch_counts, dummy_tensor_for_mask)

        attention_mask = torch.full(
            (patch_embeds.shape[0], 1, max_patches, max_patches),
            fill_value=d_min,
            dtype=patch_embeds.dtype,
        )
        attention_mask[..., :num_real_patches, :num_real_patches] = real_attention_mask

        transformer_output = super().forward(
            patch_embeds, attention_mask, position_embeddings[0], position_embeddings[1]
        )

        transformer_output = [x[:, :num_real_patches, :] for x in transformer_output]

        return transformer_output


class _PixtralVisionModel(torch.nn.Module):
    def __init__(self, model: PixtralVisionModel):
        super().__init__()
        self.transformer = self.convert_to_rbln_pixtral_vision_model(model)

    def convert_to_rbln_pixtral_vision_model(self, model: nn.Module):
        for layer in model.transformer.layers:
            layer.attention = PixtralAttention(layer.attention)
        return model.transformer

    def forward(self, patch_embeds, attention_mask, position_embeddings_1, position_embeddings_2):
        output = self.transformer(
            inputs_embeds=patch_embeds,
            attention_mask=attention_mask,
            position_embeddings=(position_embeddings_1, position_embeddings_2),
            # it need for llavaforconditionalgeneration -> compile fail with tp
            output_hidden_states=True,
            return_dict=False,
        )
        return output


class RBLNPixtralVisionModel(RBLNModel):
    """
    RBLN optimized Pixtral vision encoder model.

    This class provides hardware-accelerated inference for Pixtral vision encoders
    on RBLN devices, supporting image encoding for multimodal tasks.
    """

    def __post_init__(self, **kwargs):
        artifacts = torch.load(self.model_save_dir / self.subfolder / "torch_artifacts.pth", weights_only=False)
        with no_init_weights():
            self.patch_conv = nn.Conv2d(
                in_channels=self.config.num_channels,
                out_channels=self.config.hidden_size,
                kernel_size=self.config.patch_size,
                stride=self.config.patch_size,
                bias=False,
            )
            self.ln_pre = PixtralRMSNorm(self.config.hidden_size, eps=1e-5)
        self.patch_conv.load_state_dict(artifacts["patch_conv"])
        self.ln_pre.load_state_dict(artifacts["ln_pre"])
        self.model = RBLNRuntimePixtralVisionModel(
            self.model[0],
            main_input_name="pixel_values",
            config=self.config,
            rbln_config=self.rbln_config,
            patch_conv=self.patch_conv,
            ln_pre=self.ln_pre,
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
        save_dict["patch_conv"] = model.get_input_embeddings().state_dict()
        save_dict["ln_pre"] = model.ln_pre.state_dict()
        torch.save(save_dict, save_dir_path / subfolder / "torch_artifacts.pth")

    @classmethod
    def wrap_model_if_needed(
        cls, model: torch.nn.Module, rbln_config: RBLNPixtralVisionModelConfig
    ) -> torch.nn.Module:
        return _PixtralVisionModel(model).eval()

    @classmethod
    def update_rbln_config_using_pipe(
        cls, pipe: "RBLNDiffusionMixin", rbln_config: "RBLNDiffusionMixinConfig", submodule_name: str
    ) -> "RBLNDiffusionMixinConfig":
        return rbln_config

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"],
        model: Optional["PreTrainedModel"] = None,
        model_config: "PixtralVisionConfig" = None,
        rbln_config: Optional[RBLNPixtralVisionModelConfig] = None,
    ) -> RBLNPixtralVisionModelConfig:
        if rbln_config.max_image_size is None:
            raise ValueError("`rbln_image_size` should be specified!")

        batch_size = rbln_config.batch_size

        num_total_patches = (
            batch_size
            * (rbln_config.max_image_size[0] // model_config.patch_size)
            * (rbln_config.max_image_size[1] // model_config.patch_size)
        )

        rbln_compile_config = RBLNCompileConfig(
            input_info=[
                (
                    "patch_embeds",
                    [1, num_total_patches, model_config.hidden_size],
                    "float32",
                ),
                ("attention_mask", [1, 1, num_total_patches, num_total_patches], "float32"),
                (
                    "position_embeddings_1",
                    [
                        num_total_patches,
                        model_config.head_dim,
                    ],
                    "float32",
                ),
                (
                    "position_embeddings_2",
                    [
                        num_total_patches,
                        model_config.head_dim,
                    ],
                    "float32",
                ),
            ]
        )

        rbln_config.set_compile_cfgs([rbln_compile_config])
        return rbln_config

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: bool = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutput]:
        output = self.model(
            pixel_values, image_sizes, output_hidden_states=output_hidden_states, return_dict=return_dict
        )

        if not return_dict:
            return (output,) if not isinstance(output, (tuple, list)) else output
        else:
            return BaseModelOutput(
                last_hidden_state=output[0],
                hidden_states=output[1:],
            )
