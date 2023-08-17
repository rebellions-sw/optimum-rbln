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

import inspect
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Union

import torch
import transformers
from transformers import (
    AutoModelForTextEncoding,
    PretrainedConfig,
    T5EncoderModel,
    T5ForConditionalGeneration,
)
from transformers.modeling_outputs import BaseModelOutput

from ....diffusers.modeling_diffusers import RBLNDiffusionMixin
from ....modeling import RBLNModel
from ....modeling_config import RBLNCompileConfig, RBLNConfig
from ....utils.logging import get_logger
from ....utils.runtime_utils import RBLNPytorchRuntime
from ...models.seq2seq import RBLNModelForSeq2SeqLM
from .t5_architecture import T5Wrapper


logger = get_logger()

if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer, PreTrainedModel


class RBLNRuntimeModel(RBLNPytorchRuntime):
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor,
        head_mask: torch.FloatTensor,
        inputs_embeds: torch.FloatTensor,
        **kwargs,
    ):
        return super().forward(
            input_ids,
            attention_mask,
            head_mask,
            inputs_embeds,
            **kwargs,
        )


class T5EncoderWrapper(torch.nn.Module):
    def __init__(self, model: "T5EncoderModel") -> None:
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        kwargs.pop("return_dict", None)
        return self.model(*args, **kwargs, return_dict=False)


class RBLNT5EncoderModel(RBLNModel):
    auto_model_class = AutoModelForTextEncoding
    rbln_model_input_names = ["input_ids", "attention_mask"]

    def __post_init__(self, **kwargs):
        self.model = RBLNRuntimeModel(runtime=self.model[0])

    @classmethod
    def wrap_model_if_needed(self, model: "PreTrainedModel", rbln_config: "RBLNConfig"):
        return T5EncoderWrapper(model)

    @classmethod
    def update_rbln_config_using_pipe(cls, pipe: RBLNDiffusionMixin, rbln_config: Dict[str, Any]) -> Dict[str, Any]:
        batch_size = rbln_config.get("batch_size", 1)
        max_sequence_length = rbln_config.get("max_sequence_length", 256)
        model_input_names = ["input_ids"]

        rbln_config.update(
            {
                "batch_size": batch_size,
                "max_seq_len": max_sequence_length,
                "model_input_names": model_input_names,
            }
        )

        return rbln_config

    @classmethod
    def _get_rbln_config(
        cls,
        preprocessors: Optional[Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"]],
        model_config: Optional["PretrainedConfig"] = None,
        rbln_kwargs: Dict[str, Any] = {},
    ) -> RBLNConfig:
        rbln_max_seq_len = rbln_kwargs.get("max_seq_len", None)
        rbln_model_input_names = rbln_kwargs.get("model_input_names", None)
        rbln_batch_size = rbln_kwargs.get("batch_size", None)

        max_position_embeddings = getattr(model_config, "n_positions", None)

        if rbln_max_seq_len is None:
            rbln_max_seq_len = max_position_embeddings
            if rbln_max_seq_len is None:
                for tokenizer in preprocessors:
                    if hasattr(tokenizer, "model_max_length"):
                        rbln_max_seq_len = tokenizer.model_max_length
                        break
                if rbln_max_seq_len is None:
                    raise ValueError("`rbln_max_seq_len` should be specified!")

        if max_position_embeddings is not None and rbln_max_seq_len > max_position_embeddings:
            raise ValueError("`rbln_max_seq_len` should be less or equal than max_position_embeddings!")

        if rbln_model_input_names is None:
            for tokenizer in preprocessors:
                if hasattr(tokenizer, "model_input_names"):
                    rbln_model_input_names = tokenizer.model_input_names
                    break
            if rbln_model_input_names is None and hasattr(cls, "rbln_model_input_names"):
                rbln_model_input_names = cls.rbln_model_input_names
            elif rbln_model_input_names is None and hasattr(cls, "rbln_model_input_names") is False:
                original_model_class = getattr(transformers, model_config.architectures[0])
                input_names_order = inspect.signature(original_model_class.forward).parameters.keys()
                raise ValueError(
                    "Specify the model input names obtained by the tokenizer via `rbln_model_input_names`, "
                    f"and be sure to make the order of the inputs same as T5EncoderModel forward() arguments like ({list(input_names_order)})"
                )

        if rbln_batch_size is None:
            rbln_batch_size = 1

        input_info = [
            (model_input_name, [rbln_batch_size, rbln_max_seq_len], "int64")
            for model_input_name in rbln_model_input_names
        ]

        rbln_compile_config = RBLNCompileConfig(input_info=input_info)

        rbln_config = RBLNConfig(
            rbln_cls=cls.__name__,
            compile_cfgs=[rbln_compile_config],
            rbln_kwargs=rbln_kwargs,
        )

        rbln_config.model_cfg.update({"max_seq_len": rbln_max_seq_len})
        return rbln_config

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], BaseModelOutput]:
        encoder_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if not return_dict:
            return (encoder_outputs,)
        else:
            return BaseModelOutput(last_hidden_state=encoder_outputs)


class RBLNT5ForConditionalGeneration(RBLNModelForSeq2SeqLM):
    @classmethod
    def wrap_model_if_needed(self, model: "PreTrainedModel", rbln_config: "RBLNConfig"):
        enc_max_seq_len = rbln_config.model_cfg["enc_max_seq_len"]
        dec_max_seq_len = rbln_config.model_cfg["dec_max_seq_len"]

        return T5Wrapper(model, enc_max_seq_len=enc_max_seq_len, dec_max_seq_len=dec_max_seq_len)

    def __getattr__(self, __name: str) -> Any:
        def redirect(func):
            return lambda *pargs, **kwargs: func(self, *pargs, **kwargs)

        val = getattr(T5ForConditionalGeneration, __name)

        if isinstance(val, Callable) and "self" in set(inspect.signature(val).parameters):
            return redirect(val)

        return val
