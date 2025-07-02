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

from typing import TYPE_CHECKING

import torch.nn as nn

from ....utils import logging
from ...models.decoderonly.decoderonly_architecture import (
    DecoderOnlyAttention,
    DecoderOnlyLayer,
    DecoderOnlyModel,
    DecoderOnlyWrapper,
)


if TYPE_CHECKING:
    from transformers import PreTrainedModel as ExaoneForCausalLM

logger = logging.get_logger(__name__)


class ExaoneForCausalLMWrapper(DecoderOnlyWrapper):
    """A wrapper class for the Exaone model with a language modeling head."""

    def get_decoder_layers(self, causal_lm: "ExaoneForCausalLM"):
        return causal_lm.transformer.h

    def get_attn_layer(self, layer: nn.Module):
        return layer.attn.attention

    def get_model_layer(self, causal_lm: "ExaoneForCausalLM"):
        return causal_lm.transformer

    def get_rbln_attn_class(self):
        return ExaoneAttention

    def get_rbln_layer_class(self):
        return ExaoneLayer

    def get_rbln_model_class(self):
        return ExaoneModel


class ExaoneModel(DecoderOnlyModel):
    def get_embedding(self) -> nn.Embedding:
        return self._original_mod.wte

    def get_last_layernorm(self) -> nn.LayerNorm:
        return self._original_mod.ln_f


class ExaoneLayer(DecoderOnlyLayer):
    def get_pre_attention_layernorm(self) -> nn.LayerNorm:
        return self._original_mod.ln_1

    def get_post_attention_layernorm(self) -> nn.LayerNorm:
        return self._original_mod.ln_2


class ExaoneAttention(DecoderOnlyAttention):
    def __post_init__(self):
        self.q_proj = self._original_mod.q_proj
        self.k_proj = self._original_mod.k_proj
        self.v_proj = self._original_mod.v_proj
        self.o_proj = self._original_mod.out_proj
