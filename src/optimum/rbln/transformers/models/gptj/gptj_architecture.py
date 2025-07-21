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

from typing import TYPE_CHECKING, Optional, Tuple

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
    from transformers import GPTJModel


def rotate_every_two(x: torch.Tensor) -> torch.Tensor:
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    rot_dim = sin.shape[-1]
    q_rot, q_pass = q[..., :rot_dim], q[..., rot_dim:]
    k_rot, k_pass = k[..., :rot_dim], k[..., rot_dim:]

    q_rot = (q_rot * cos) + (rotate_every_two(q_rot) * sin)
    k_rot = (k_rot * cos) + (rotate_every_two(k_rot) * sin)
    query = torch.cat([q_rot, q_pass], dim=-1)
    key = torch.cat([k_rot, k_pass], dim=-1)
    return query, key


class GPTJWrapper(DecoderOnlyWrapper):
    def get_rbln_attn_class(self):
        return GPTJAttention

    def get_rbln_layer_class(self):
        return GPTJLayer

    def get_rbln_model_class(self):
        return GPTJModel

    def get_attn_layer(self, layer: nn.Module):
        return layer.attn

    def get_model_layer(self, causal_lm: "GPTJModel"):
        return causal_lm.transformer

    def get_decoder_layers(self, causal_lm: PreTrainedModel):
        return causal_lm.transformer.h


class GPTJModel(DecoderOnlyModel):
    def get_last_layernorm(self) -> nn.LayerNorm:
        return self._original_mod.ln_f

    def get_embedding(self) -> nn.Embedding:
        return self._original_mod.wte


class GPTJLayer(DecoderOnlyLayer):
    def get_pre_attention_layernorm(self) -> nn.LayerNorm:
        return self._original_mod.ln_1

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        seq_positions: torch.LongTensor,
        past_key_values: Tuple[Tuple[torch.Tensor]],
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        block_tables: Optional[torch.Tensor] = None,
    ):
        residual = hidden_states
        hidden_states = self.get_pre_attention_layernorm()(hidden_states)

        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            seq_positions=seq_positions,
            past_key_values=past_key_values,
            cos=cos,
            sin=sin,
            block_tables=block_tables,
        )
        attn_output = attn_outputs[0]
        # hidden_states = residual + hidden_states

        # residual = hidden_states
        feed_forward_hidden_states = self._original_mod.mlp(hidden_states)
        hidden_states = attn_output + feed_forward_hidden_states + residual
        # hidden_states = residual + hidden_states

        return hidden_states


class GPTJAttention(DecoderOnlyAttention):
    def __post_init__(self):
        self.k_proj = self._original_mod.k_proj
        self.v_proj = self._original_mod.v_proj
        self.q_proj = self._original_mod.q_proj
        self.o_proj = self._original_mod.out_proj

    def apply_rotary_pos_embed(self, query_states, key_states, cos, sin):
        return apply_rotary_pos_emb(query_states, key_states, cos, sin)
