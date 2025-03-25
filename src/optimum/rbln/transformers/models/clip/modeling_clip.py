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

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import torch
from transformers import (
    CLIPTextConfig,
    CLIPTextModel,
    CLIPVisionConfig,
    CLIPVisionModel,
)
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip.modeling_clip import CLIPTextModelOutput, CLIPVisionModelOutput

from ....configuration_utils import RBLNCompileConfig, RBLNModelConfig
from ....diffusers.modeling_diffusers import RBLNDiffusionMixin
from ....modeling import RBLNModel
from ....utils.logging import get_logger


logger = get_logger(__name__)

if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer, CLIPTextModel, PreTrainedModel


class _TextEncoder(torch.nn.Module):
    def __init__(self, enc: "CLIPTextModel"):
        super().__init__()
        self.enc = enc

    def forward(self, inp):
        enc_out = self.enc(inp, output_hidden_states=True, return_dict=False)
        return enc_out


class RBLNCLIPTextModel(RBLNModel):
    @classmethod
    def wrap_model_if_needed(cls, model: torch.nn.Module, rbln_config: RBLNModelConfig) -> torch.nn.Module:
        return _TextEncoder(model).eval()

    @classmethod
    def update_rbln_config_using_pipe(cls, pipe: RBLNDiffusionMixin, rbln_config: Dict[str, Any]) -> Dict[str, Any]:
        return rbln_config

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"],
        model: Optional["PreTrainedModel"] = None,
        model_config: "CLIPTextConfig" = None,
        rbln_config: Optional[RBLNModelConfig] = None,
    ) -> RBLNModelConfig:
        input_info = [
            (
                "input_ids",
                [
                    rbln_config.batch_size,
                    model_config.max_position_embeddings,
                ],
                "int64",
            ),
        ]

        rbln_config.set_compile_cfgs([RBLNCompileConfig(input_info=input_info)])
        return rbln_config

    def forward(self, input_ids: "torch.Tensor", **kwargs):
        text_output = super().forward(input_ids)
        return CLIPTextModelOutput(
            text_embeds=text_output[0],
            last_hidden_state=text_output[1],
            hidden_states=text_output[2:],
        )


class RBLNCLIPTextModelWithProjection(RBLNCLIPTextModel):
    pass


class _VisionEncoder(torch.nn.Module):
    def __init__(self, enc: CLIPVisionModel):
        super().__init__()
        self.enc = enc

    def forward(self, inp):
        enc_out = self.enc(inp, output_hidden_states=True, return_dict=False)
        return enc_out


class RBLNCLIPVisionModel(RBLNModel):
    @classmethod
    def wrap_model_if_needed(cls, model: torch.nn.Module, rbln_config: RBLNModelConfig) -> torch.nn.Module:
        return _VisionEncoder(model).eval()

    @classmethod
    def update_rbln_config_using_pipe(cls, pipe: RBLNDiffusionMixin, rbln_config: Dict[str, Any]) -> Dict[str, Any]:
        return rbln_config

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"],
        model: Optional["PreTrainedModel"] = None,
        model_config: "CLIPVisionConfig" = None,
        rbln_config: Optional[RBLNModelConfig] = None,
    ) -> RBLNModelConfig:
        if rbln_config.image_size is None:
            rbln_config.image_size = getattr(model_config, "image_size", None)

        if isinstance(rbln_config.image_size, int):
            rbln_config.image_size = (rbln_config.image_size, rbln_config.image_size)

        if rbln_config.image_size is None:
            raise ValueError("`rbln_image_size` should be specified!")

        rbln_compile_config = RBLNCompileConfig(
            input_info=[
                (
                    "pixel_values",
                    [
                        rbln_config.batch_size,
                        3,
                        rbln_config.image_height,
                        rbln_config.image_width,
                    ],
                    "float32",
                )
            ]
        )

        rbln_config.set_compile_cfgs([rbln_compile_config])
        return rbln_config

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        if len(kwargs) > 0 and any(kwargs.values()):
            logger.warning(f"Currently, optimum-rbln does not support kwargs {kwargs.keys()} for {self.__class__}.")

        output = super().forward(pixel_values)
        return BaseModelOutputWithPooling(
            last_hidden_state=output[0],
            pooler_output=output[1],
            hidden_states=output[2:],
        )


class RBLNCLIPVisionModelWithProjection(RBLNCLIPVisionModel):
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CLIPVisionModelOutput]:
        if len(kwargs) > 0 and any(kwargs.values()):
            logger.warning(f"Currently, optimum-rbln does not support kwargs {kwargs.keys()} for {self.__class__}.")

        output = super().forward(pixel_values)
        image_embeds = output[0]
        last_hidden_state = output[1]
        hidden_states = output[2:]

        return CLIPVisionModelOutput(
            image_embeds=image_embeds,
            last_hidden_state=last_hidden_state,
            hidden_states=hidden_states,
        )
