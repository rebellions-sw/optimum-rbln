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

from ..decoderonly.decoderonly_architecture import (
    DecoderOnlyAttention,
    DecoderOnlyForCausalLM,
    DecoderOnlyLayer,
    DecoderOnlyModel,
    DecoderOnlyWrapper,
)


if TYPE_CHECKING:
    from transformers import GPT2LMHeadModel


class GPT2Wrapper(DecoderOnlyWrapper):
    def convert_to_rbln_causal_lm(self, causal_lm: "GPT2LMHeadModel", max_seq_len: int):
        if self.attn_impl != "eager":
            raise NotImplementedError(f"flash attention ({self.attn_impl}) is not implemented for {self.__class__}")
        new_layers = []
        for layer in causal_lm.transformer.h:
            new_self_attn = GPT2Attention(
                layer.attn,
                self.use_attention_mask,
                kvcache_block_size=self.kvcache_block_size,
                use_position_ids=self.use_position_ids,
            )
            new_layer = GPT2Layer(layer, new_self_attn)
            new_layers.append(new_layer)
        new_model = GPT2Model(
            causal_lm.transformer,
            new_layers,
            max_seq_len=max_seq_len,
            sliding_window_layers=self.sliding_window_layers,
        )
        new_causal_lm = DecoderOnlyForCausalLM(causal_lm, new_model)
        return new_causal_lm


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
