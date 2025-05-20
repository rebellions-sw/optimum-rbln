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
                    # rbln_config.batch_size,
                    1,
                    model_config.num_channels,
                    224,
                    224,
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
                    1,
                    32,
                    model_config.hidden_size,
                ],
                "float32",
            ),
            (
                "encoder_hidden_states",
                [
                    1,
                    257,
                    model_config.encoder_hidden_size,
                ],
                "float32",
            ),
            (
                "encoder_attention_mask",
                [1, 257],
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
    # _rbln_submodules = [{"name": "language_model"}]

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
                [1, 32, 768],
                "float32",
            ),
        ]

        rbln_compile_config = RBLNCompileConfig(input_info=input_info)
        rbln_config.set_compile_cfgs([rbln_compile_config])

        return rbln_config

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        interpolate_pos_encoding: bool = False,
        **generate_kwargs,
    ) -> torch.LongTensor:
        """
        Overrides `generate` function to be able to use the model as a conditional generator.

        Args:
            pixel_values (`torch.FloatTensor` of shape (batch_size, num_channels, height, width)):
                Input images to be processed.
            input_ids (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                The sequence used as a prompt for the generation.
            attention_mask (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                Mask to avoid performing attention on padding token indices

        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """

        batch_size = pixel_values.shape[0]
        image_embeds = self.vision_model(
            pixel_values,
            return_dict=True,
            interpolate_pos_encoding=interpolate_pos_encoding,
        ).last_hidden_state
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )
        query_output = query_outputs.last_hidden_state

        if query_output.dtype != image_embeds.dtype:
            query_output = query_output.to(image_embeds.dtype)

        language_model_inputs = self.language_projection(query_output)
        language_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )

        if input_ids is None:
            start_tokens = [self.config.text_config.bos_token_id]
            if getattr(self.config, "image_token_index", None) is not None:
                start_tokens = [self.config.image_token_index] * self.config.num_query_tokens + start_tokens
            input_ids = torch.tensor([start_tokens], dtype=torch.long, device=image_embeds.device)
            input_ids = input_ids.repeat(batch_size, 1)

        inputs_embeds = self.get_input_embeddings()(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        if getattr(self.config, "image_token_index", None) is not None:
            special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds[special_image_mask] = language_model_inputs.flatten()
        else:
            logger.warning_once(
                "Expanding inputs for image tokens in BLIP-2 should be done in processing. "
                "Please follow instruction here (https://gist.github.com/zucchini-nlp/e9f20b054fa322f84ac9311d9ab67042) to update your BLIP-2 model. "
                "Using processors without these attributes in the config is deprecated and will throw an error in v4.50."
            )
            inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)
            attention_mask = torch.cat(
                [language_attention_mask, attention_mask.to(language_attention_mask.device)], dim=1
            )
            if not self.language_model.config.is_encoder_decoder:
                generate_kwargs["max_length"] = (
                    generate_kwargs.get("max_length", 20) + language_model_inputs.shape[1] - 1
                )
                generate_kwargs["min_length"] = generate_kwargs.get("min_length", 0) + language_model_inputs.shape[1]

        inputs = {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask}
        if not self.language_model.config.is_encoder_decoder:
            inputs["input_ids"] = input_ids

        outputs = self.language_model.generate(**inputs, **generate_kwargs)
        return outputs
