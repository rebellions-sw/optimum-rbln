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

from typing import TYPE_CHECKING, Optional, Union

import torch
from transformers.models.grounding_dino.modeling_grounding_dino import (
    GroundingDinoDecoderOutput,
    GroundingDinoEncoderOutput,
    get_sine_pos_embed,
)

from ....configuration_utils import RBLNCompileConfig
from ....modeling import RBLNModel
from ....utils.logging import get_logger
from .configuration_grounding_dino import RBLNGroundingDinoEncoderConfig, RBLNGroundingDinoForObjectDetectionConfig


# from transformers.models.grounding_dino.modeling_grounding_dino import generate_masks_with_special_tokens_and_transfer_map

logger = get_logger(__name__)

if TYPE_CHECKING:
    from transformers import (
        AutoFeatureExtractor,
        AutoProcessor,
        AutoTokenizer,
        GroundingDinoModel,
        PreTrainedModel,
    )

    from ....diffusers.modeling_diffusers import RBLNDiffusionMixin, RBLNDiffusionMixinConfig

class RBLNGroundingDinoForObjectDetection(RBLNModel):
    _rbln_submodules = [
        {"name": "encoder"},
        # {"name": "text_backbone"},
        # {"name": "backbone"},
        {"name": "decoder"},
    ]
    """
    RBLN optimized CLIP text encoder model.

    This class provides hardware-accelerated inference for CLIP text encoders
    on RBLN devices, supporting text encoding for multimodal tasks.
    """

    @classmethod
    def get_pytorch_model(cls, *args, **kwargs):
        model = super().get_pytorch_model(*args, **kwargs)
        model.encoder = model.model.encoder
        model.encoder.config = model.config
        return model

    @classmethod
    def wrap_model_if_needed(
        cls, model: torch.nn.Module, rbln_config: RBLNGroundingDinoForObjectDetectionConfig
    ) -> torch.nn.Module:
        return _GroundingDinoModel(model.model).eval()

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
        model_config: "GroundingDinoConfig" = None,
        rbln_config: Optional[RBLNGroundingDinoForObjectDetectionConfig] = None,
    ) -> RBLNGroundingDinoForObjectDetectionConfig:
        if rbln_config.image_size is None:
            longest_edge = None
            for processor in preprocessors:
                if hasattr(processor, "image_processor"):
                    longest_edge = processor.image_processor.size["longest_edge"]
                    break
            padded_longest_edge = longest_edge + ((64 - longest_edge) % 64)
            rbln_config.image_size = padded_longest_edge

        input_info = [
            (
                "pixel_values",
                [rbln_config.batch_size, 3, 1333, 1333],  # 800 < h,w<1333
                "float32",
            ),
            (
                "input_ids",
                [
                    rbln_config.batch_size,
                    model_config.max_text_len,
                ],
                "int64",
            ),
            (
                "token_type_ids",
                [
                    rbln_config.batch_size,
                    model_config.max_text_len,
                ],
                "int64",
            ),
            (
                "pixel_mask",
                [rbln_config.batch_size, 1333, 1333],
                "int16",
            ),
            (
                "text_self_attention_masks",
                [
                    rbln_config.batch_size,
                    model_config.max_text_len,
                    model_config.max_text_len,
                ],
                "int16",
            ),
            (
                "position_ids",
                [
                    rbln_config.batch_size,
                    model_config.max_text_len,
                ],
                "int64",
            ),
        ]

        rbln_config.set_compile_cfgs([RBLNCompileConfig(input_info=input_info)])
        return rbln_config

    def forward(self, *args, return_dict: bool = None, **kwargs) -> torch.FloatTensor:
        # To ignore using attention_mask, we override forward method.
        output = super().forward(*args, **kwargs, return_dict=return_dict)
        import pdb

        pdb.set_trace()
        return output

    # def _prepare_output(self, output, return_dict):
    #     # Prepare model output based on return_dict flag.
    #     # This method can be overridden by subclasses to provide task-specific output handling.

    #     if not return_dict:
    #         return (output,) if not isinstance(output, (tuple, list)) else output
    #     else:
    #         return CLIPTextModelOutput(
    #             text_embeds=output[0],
    #             last_hidden_state=output[1],
    #             hidden_states=output[2:],
    #         )


class RBLNGroundingDinoEncoder(RBLNModel):
    """
    RBLN optimized CLIP text encoder model.

    This class provides hardware-accelerated inference for CLIP text encoders
    on RBLN devices, supporting text encoding for multimodal tasks.
    """

    @classmethod
    def wrap_model_if_needed(
        cls, model: torch.nn.Module, rbln_config: RBLNGroundingDinoForObjectDetectionConfig
    ) -> torch.nn.Module:
        return GroundingDinoEncoder(model).eval()

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
        model_config: "GroundingDinoEncoderConfig" = None,
        rbln_config: Optional[RBLNGroundingDinoEncoderConfig] = None,
    ) -> RBLNGroundingDinoEncoderConfig:
        input_info = [
            (
                "vision_features",
                [rbln_config.batch_size, 37150, model_config.d_model],
                "float32",
            ),
            (
                "vision_attention_mask",
                [
                    rbln_config.batch_size,
                    37150,
                ],
                "float32",
            ),
            (
                "vision_position_embedding",
                [rbln_config.batch_size, 37150, model_config.d_model],
                "float32",
            ),
            (
                "text_features",
                [rbln_config.batch_size, model_config.max_text_len, model_config.d_model],
                "float32",
            ),
            (
                "text_attention_mask",
                [
                    rbln_config.batch_size,
                    model_config.max_text_len,
                ],
                "float32",
            ),
            (
                "text_self_attention_masks",
                [
                    rbln_config.batch_size,
                    model_config.max_text_len,
                    model_config.max_text_len,
                ],
                "float32",
            ),
            (
                "text_position_ids",
                [
                    rbln_config.batch_size,
                    model_config.max_text_len,
                ],
                "int32",
            ),
            (
                "reference_points",
                [rbln_config.batch_size, 37150, 4, 2],
                "float32",
            ),
        ]

        rbln_config.set_compile_cfgs([RBLNCompileConfig(input_info=input_info)])
        return rbln_config

    def forward(self, *args, return_dict: bool = None, **kwargs) -> torch.FloatTensor:
        # To ignore using attention_mask, we override forward method.
        output = super().forward(*args, **kwargs, return_dict=return_dict)
        return output


class RBLNGroundingDinoDecoder(RBLNModel):
    """
    RBLN optimized CLIP text encoder model.

    This class provides hardware-accelerated inference for CLIP text encoders
    on RBLN devices, supporting text encoding for multimodal tasks.
    """

    @classmethod
    def wrap_model_if_needed(
        cls, model: torch.nn.Module, rbln_config: RBLNGroundingDinoForObjectDetectionConfig
    ) -> torch.nn.Module:
        return GroundingDinoEncoder(model).eval()

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
        model_config: "GroundingDinoDecoderConfig" = None,
        rbln_config: Optional[RBLNGroundingDinoEncoderConfig] = None,
    ) -> RBLNGroundingDinoEncoderConfig:
        input_info = [
            (
                "inputs_embeds",
                [rbln_config.batch_size, 900, model_config.d_model],
                "float32",
            ),
            (
                "vision_encoder_hidden_states",
                [
                    rbln_config.batch_size,
                    37150,
                ],
                "float32",
            ),
            (
                "vision_encoder_attention_mask",
                [rbln_config.batch_size, 37150],
                "float32",
            ),
            (
                "text_encoder_hidden_states",
                [rbln_config.batch_size, model_config.max_text_len, model_config.d_model],
                "float32",
            ),
            (
                "text_encoder_attention_mask",
                [
                    rbln_config.batch_size,
                    model_config.max_text_len,
                ],
                "float32",
            ),
            (
                "reference_points",
                [
                    rbln_config.batch_size,
                    900,
                    4,
                ],
                "float32",
            ),
            (
                "valid_ratios",
                [
                    rbln_config.batch_size,
                    4,
                    2,
                ],
                "float32",
            ),
        ]

        rbln_config.set_compile_cfgs([RBLNCompileConfig(input_info=input_info)])
        return rbln_config

    def forward(self, *args, return_dict: bool = None, **kwargs) -> torch.FloatTensor:
        # To ignore using attention_mask, we override forward method.
        output = super().forward(*args, **kwargs, return_dict=return_dict)
        return output
