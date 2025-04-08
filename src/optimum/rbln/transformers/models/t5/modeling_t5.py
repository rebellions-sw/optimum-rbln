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
from typing import TYPE_CHECKING, Any, Callable

import torch
from transformers import AutoModelForTextEncoding, T5EncoderModel, T5ForConditionalGeneration

from ...modeling_generic import RBLNTransformerEncoderForFeatureExtraction
from ...models.seq2seq import RBLNModelForSeq2SeqLM
from .configuration_t5 import RBLNT5EncoderModelConfig, RBLNT5ForConditionalGenerationConfig
from .t5_architecture import T5Wrapper


if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from ....diffusers.modeling_diffusers import RBLNDiffusionMixin, RBLNDiffusionMixinConfig


class T5EncoderWrapper(torch.nn.Module):
    def __init__(self, model: "T5EncoderModel") -> None:
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        kwargs.pop("return_dict", None)
        return self.model(*args, **kwargs, return_dict=False)


class RBLNT5EncoderModel(RBLNTransformerEncoderForFeatureExtraction):
    auto_model_class = AutoModelForTextEncoding
    rbln_model_input_names = ["input_ids", "attention_mask"]

    @classmethod
    def wrap_model_if_needed(self, model: "PreTrainedModel", rbln_config: RBLNT5EncoderModelConfig):
        return T5EncoderWrapper(model)

    @classmethod
    def update_rbln_config_using_pipe(
        cls,
        pipe: "RBLNDiffusionMixin",
        rbln_config: "RBLNDiffusionMixinConfig",
        submodule_name: str,
    ) -> "RBLNDiffusionMixinConfig":
        submodule_config = getattr(rbln_config, submodule_name)
        submodule_config.max_seq_len = rbln_config.max_seq_len or 256
        submodule_config.model_input_names = ["input_ids"]
        return rbln_config


class RBLNT5ForConditionalGeneration(RBLNModelForSeq2SeqLM):
    support_causal_attn = False

    @classmethod
    def wrap_model_if_needed(self, model: "PreTrainedModel", rbln_config: RBLNT5ForConditionalGenerationConfig):
        return T5Wrapper(
            model, enc_max_seq_len=rbln_config.enc_max_seq_len, dec_max_seq_len=rbln_config.dec_max_seq_len
        )

    def __getattr__(self, __name: str) -> Any:
        def redirect(func):
            return lambda *pargs, **kwargs: func(self, *pargs, **kwargs)

        val = getattr(T5ForConditionalGeneration, __name)

        if isinstance(val, Callable) and "self" in set(inspect.signature(val).parameters):
            return redirect(val)

        return val
