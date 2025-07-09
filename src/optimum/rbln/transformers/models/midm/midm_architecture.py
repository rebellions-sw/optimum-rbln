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
    DecoderOnlyLayer,
    DecoderOnlyModel,
    DecoderOnlyWrapper,
    apply_rotary_pos_emb_partial,
    rotate_half,
)


if TYPE_CHECKING:
    from transformers import PreTrainedModel as MidmLMHeadModel


def apply_rotary_to_tensor(tensor, cos, sin, rot_dim):
    """Applies rotary position embedding to the specified dimension of the tensor."""
    tensor_, tensor_pass = tensor[..., :rot_dim], tensor[..., rot_dim:]
    tensor_embed = (tensor_ * cos) + (rotate_half(tensor_) * sin)
    return torch.cat((tensor_embed, tensor_pass), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Applies Rotary Position Embedding to the query and key tensors."""
    rot_dim = cos.shape[-1]
    q_embed = apply_rotary_to_tensor(q, cos, sin, rot_dim)
    k_embed = apply_rotary_to_tensor(k, cos, sin, rot_dim)
    return q_embed, k_embed


class MidmLMHeadModelWrapper(DecoderOnlyWrapper):
    def get_rotary_emb(self, max_seq_len):
        self.config.rope_theta = 10000
        self.config.head_dim = self.config.n_embd // self.config.n_head
        self.config.partial_rotary_factor = self.config.rotary_percentage
        return super().get_rotary_emb(max_seq_len=max_seq_len)

    def get_rbln_attn_class(self):
        return MidmAttention

    def get_rbln_layer_class(self):
        return MidmLayer

    def get_rbln_model_class(self):
        return MidmModel

    def get_model_layer(self, causal_lm: "MidmLMHeadModel"):
        return causal_lm.transformer

    def get_decoder_layers(self, causal_lm: "MidmLMHeadModel"):
        return causal_lm.transformer.h


class MidmModel(DecoderOnlyModel):
    def get_layernorm1p(self, module: nn.LayerNorm):
        def layernorm1p(input: torch.Tensor):
            """Applies Layer Normalization with a slight modification on the weights."""
            return torch.nn.functional.layer_norm(
                input, module.normalized_shape, module.weight + 1, module.bias, module.eps
            )

        return layernorm1p

    def get_last_layernorm(self) -> nn.LayerNorm:
        if self._original_mod.use_layernorm1p:
            return self.get_layernorm1p(self._original_mod.ln_f)
        else:
            return self._original_mod.ln_f

    def get_embedding(self) -> nn.Embedding:
        return self._original_mod.wte

    def get_pos_embedding(self) -> nn.Embedding:
        return self._original_mod.wpe


class MidmLayer(DecoderOnlyLayer):
    def get_layernorm1p(self, module: nn.LayerNorm):
        def layernorm1p(input: torch.Tensor):
            """Applies Layer Normalization with a slight modification on the weights."""
            return torch.nn.functional.layer_norm(
                input, module.normalized_shape, module.weight + 1, module.bias, module.eps
            )

        return layernorm1p

    def get_pre_attention_layernorm(self) -> nn.LayerNorm:
        if self._original_mod.use_layernorm1p:
            return self.get_layernorm1p(self._original_mod.ln_1)
        else:
            return self._original_mod.ln_1

    def get_post_attention_layernorm(self) -> nn.LayerNorm:
        if self._original_mod.use_layernorm1p:
            return self.get_layernorm1p(self._original_mod.ln_2)
        else:
            return self._original_mod.ln_2


class MidmAttention(DecoderOnlyAttention):
    def __post_init__(self):
        self.c_attn = self._original_mod.c_attn
        self.o_proj = self._original_mod.c_proj
        self.split_size = self._original_mod.split_size
        self.num_key_value_heads = self._original_mod.num_heads

    def projection(self, hidden_states) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        return query_states, key_states, value_states

    def get_attn_scale(self):
        scale = 1.0
        if self._original_mod.scale_attn_weights:
            scale /= math.sqrt(self.head_dim)

        if self._original_mod.scale_attn_by_inverse_layer_idx and not self._original_mod.scale_qk_by_inverse_layer_idx:
            scale /= 1 + self.layer_idx

        return scale

    def apply_rotary_pos_embed(self, query_states, key_states, cos, sin):
        return apply_rotary_pos_emb_partial(query_states, key_states, cos, sin, ndim=cos.shape[-1])
