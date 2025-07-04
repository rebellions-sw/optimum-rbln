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
from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from ..decoderonly.decoderonly_architecture import (
    DecoderOnlyAttention,
    DecoderOnlyLayer,
    DecoderOnlyModel,
    DecoderOnlyWrapper,
)


if TYPE_CHECKING:
    from transformers import GPT2LMHeadModel


class GPT2Wrapper(DecoderOnlyWrapper):
    def get_rbln_attn_class(self):
        return GPT2Attention

    def get_rbln_layer_class(self):
        return GPT2Layer

    def get_rbln_model_class(self):
        return GPT2Model

    def get_attn_layer(self, layer: nn.Module):
        return layer.attn

    def get_model_layer(self, causal_lm: "GPT2LMHeadModel"):
        return causal_lm.transformer

    def get_decoder_layers(self, causal_lm: PreTrainedModel):
        return causal_lm.transformer.h


class GPT2Model(DecoderOnlyModel):
    def get_last_layernorm(self) -> nn.LayerNorm:
        return self._original_mod.ln_f

    def get_embedding(self) -> nn.Embedding:
        return self._original_mod.wte

    def get_pos_embedding(self) -> nn.Embedding:
        return self._original_mod.wpe


class GPT2Layer(DecoderOnlyLayer):
    def get_pre_attention_layernorm(self) -> nn.LayerNorm:
        return self._original_mod.ln_1

    def get_post_attention_layernorm(self) -> nn.LayerNorm:
        return self._original_mod.ln_2


class GPT2Attention(DecoderOnlyAttention):
    def __post_init__(self):
        self.c_attn = self._original_mod.c_attn
        self.o_proj = self._original_mod.c_proj
        self.split_size = self._original_mod.split_size

    def projection(self, hidden_states) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        return query_states, key_states, value_states

    def get_attn_scale(self):
        scale = 1.0
        if self._original_mod.scale_attn_weights:
            scale /= math.sqrt(self.head_dim)

        if self._original_mod.scale_attn_by_inverse_layer_idx:
            scale /= 1 + self.layer_idx

        return scale
