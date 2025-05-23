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

import inspect
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple, Union

import torch
from transformers import (
    AutoModelForVisualQuestionAnswering,
    Blip2ForConditionalGeneration,
    Blip2QFormerModel,
    Blip2VisionModel,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import BaseModelOutputWithPooling, BaseModelOutputWithPoolingAndCrossAttentions
from transformers.utils import logging

from ....configuration_utils import RBLNCompileConfig, RBLNModelConfig
from ....modeling import RBLNModel


logger = logging.get_logger(__name__)

if TYPE_CHECKING:
    from transformers import (
        AutoFeatureExtractor,
        AutoProcessor,
        AutoTokenizer,
    )


class RBLNBlip2VisionModel(RBLNModel):
    def get_input_embeddings(self):
        return self.embeddings

    @classmethod
    def wrap_model_if_needed(cls, model: torch.nn.Module, rbln_config: RBLNModelConfig) -> torch.nn.Module:
        class Blip2VisionModelWrapper(torch.nn.Module):
            def __init__(self, model: "Blip2VisionModel") -> None:
                super().__init__()
                self.model = model

            def forward(self, *args, **kwargs):
                kwargs.pop("return_dict", None)
                return self.model(*args, **kwargs, return_dict=False)

        return Blip2VisionModelWrapper(model).eval()

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Optional[Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"]],
        model: Optional["PreTrainedModel"] = None,
        model_config: Optional["PretrainedConfig"] = None,
        rbln_config: Optional[RBLNModelConfig] = None,
    ) -> RBLNModelConfig:
        input_info = [
            (
                "pixel_values",
                [
                    rbln_config.batch_size,
                    model_config.num_channels,
                    model_config.image_size,
                    model_config.image_size,
                ],
                "float32",
            ),
        ]

        rbln_compile_config = RBLNCompileConfig(input_info=input_info)
        rbln_config.set_compile_cfgs([rbln_compile_config])
        return rbln_config

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        output = super().forward(pixel_values, return_dict=return_dict)
        return output

    def _prepare_output(self, output, return_dict):
        """
        Prepare model output based on return_dict flag.
        This method can be overridden by subclasses to provide task-specific output handling.
        """
        if not return_dict:
            return (output,) if not isinstance(output, (tuple, list)) else output
        else:
            return BaseModelOutputWithPooling(
                last_hidden_state=output[0],
                pooler_output=output[1],
            )


class RBLNBlip2QFormerModel(RBLNModel):
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    @classmethod
    def wrap_model_if_needed(cls, model: torch.nn.Module, rbln_config: RBLNModelConfig) -> torch.nn.Module:
        class Blip2QFormerModelWrapper(torch.nn.Module):
            def __init__(self, model: "Blip2QFormerModel"):
                super().__init__()
                self.model = model

            def forward(
                self,
                query_embeds: torch.FloatTensor,
                encoder_hidden_states: Optional[torch.FloatTensor] = None,
                encoder_attention_mask: Optional[torch.FloatTensor] = None,
            ) -> torch.Tensor:
                qformer_out = self.model(
                    query_embeds=query_embeds,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )
                return qformer_out

        return Blip2QFormerModelWrapper(model).eval()

    @classmethod
    def _update_submodule_config(cls, model: "PreTrainedModel", rbln_config: "RBLNModelConfig") -> "RBLNModelConfig":
        if rbln_config.num_query_tokens is None:
            rbln_config.num_query_tokens = model.config.num_query_tokens

        if rbln_config.image_text_hidden_size is None:
            rbln_config.image_text_hidden_size = model.config.image_text_hidden_size

        return rbln_config

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Optional[Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"]],
        model: Optional["PreTrainedModel"] = None,
        model_config: Optional["PretrainedConfig"] = None,
        rbln_config: Optional[RBLNModelConfig] = None,
    ) -> RBLNModelConfig:
        input_info = [
            (
                "query_embeds",
                [
                    rbln_config.batch_size,
                    rbln_config.num_query_tokens,
                    model_config.hidden_size,
                ],
                "float32",
            ),
            (
                "encoder_hidden_states",
                [
                    rbln_config.batch_size,
                    # image_text_hidden_size + cls token
                    rbln_config.image_text_hidden_size + 1,
                    model_config.encoder_hidden_size,
                ],
                "float32",
            ),
            (
                "encoder_attention_mask",
                # image_text_hidden_size + cls token
                [rbln_config.batch_size, rbln_config.image_text_hidden_size + 1],
                "int64",
            ),
        ]

        rbln_compile_config = RBLNCompileConfig(input_info=input_info)
        rbln_config.set_compile_cfgs([rbln_compile_config])
        return rbln_config

    def forward(
        self,
        query_embeds: torch.FloatTensor,
        query_length: Optional[int] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        output = super().forward(query_embeds, encoder_hidden_states, encoder_attention_mask, return_dict=return_dict)
        return output

    def _prepare_output(self, output, return_dict):
        """
        Prepare model output based on return_dict flag.
        This method can be overridden by subclasses to provide task-specific output handling.
        """
        if not return_dict:
            return (output,) if not isinstance(output, (tuple, list)) else output
        else:
            return BaseModelOutputWithPoolingAndCrossAttentions(
                last_hidden_state=output[0],
                pooler_output=output[1],
            )


class RBLNBlip2ForConditionalGeneration(RBLNModel):
    auto_model_class = AutoModelForVisualQuestionAnswering
    _rbln_submodules = [{"name": "vision_model"}, {"name": "qformer"}, {"name": "language_model"}]

    def __getattr__(self, __name: str) -> Any:
        def redirect(func):
            return lambda *pargs, **kwargs: func(self, *pargs, **kwargs)

        val = getattr(Blip2ForConditionalGeneration, __name)

        if isinstance(val, Callable) and "self" in set(inspect.signature(val).parameters):
            return redirect(val)
        return val

    def can_generate(self):
        return True

    @classmethod
    def save_torch_artifacts(
        cls,
        model: "Blip2ForConditionalGeneration",
        save_dir_path: Path,
        subfolder: str,
        rbln_config: RBLNModelConfig,
    ):
        """
        If you are unavoidably running on a CPU rather than an RBLN device,
        store the torch tensor, weight, etc. in this function.
        """
        save_dict = {}
        save_dict["query_tokens"] = model.query_tokens
        torch.save(save_dict, save_dir_path / subfolder / "query_tokens.pth")

    def __post_init__(self, **kwargs):
        self.vision_model = self.rbln_submodules[0]
        self.language_model = self.rbln_submodules[2]
        self.qformer = self.rbln_submodules[1]
        self.language_projection = self.model[0]

        artifacts = torch.load(self.model_save_dir / self.subfolder / "query_tokens.pth", weights_only=False)
        self.query_tokens = artifacts["query_tokens"]

    def get_attn_impl(self) -> str:
        return self.rbln_config.language_model.attn_impl

    def get_kvcache_num_blocks(self) -> int:
        return self.rbln_config.language_model.kvcache_num_blocks

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    @classmethod
    def wrap_model_if_needed(cls, model, rbln_config):
        return model.language_projection

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Optional[Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"]],
        model: Optional["PreTrainedModel"] = None,
        model_config: Optional["PretrainedConfig"] = None,
        rbln_config: Optional[RBLNModelConfig] = None,
    ) -> RBLNModelConfig:
        input_info = [
            (
                "query_output",
                [
                    rbln_config.batch_size,
                    model_config.num_query_tokens,
                    model_config.qformer_config.hidden_size,
                ],
                "float32",
            ),
        ]

        rbln_compile_config = RBLNCompileConfig(input_info=input_info)
        rbln_config.set_compile_cfgs([rbln_compile_config])

        return rbln_config
