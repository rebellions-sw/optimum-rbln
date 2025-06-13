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
from transformers import PhiForCausalLM

from ..decoderonly.decoderonly_architecture import (
    DecoderOnlyAttention,
    DecoderOnlyForCausalLM,
    DecoderOnlyLayer,
    DecoderOnlyModel,
    DecoderOnlyWrapper,
    apply_rotary_pos_emb_partial,
)


if TYPE_CHECKING:
    from transformers import PhiForCausalLM


class PhiWrapper(DecoderOnlyWrapper):
    def convert_to_rbln_causal_lm(self, causal_lm: "PhiForCausalLM", max_seq_len: int):
        new_layers = []
        for layer in causal_lm.model.layers:
            if self.attn_impl == "eager":
                new_self_attn = PhiAttention(
                    layer.self_attn,
                    self.use_attention_mask,
                    kvcache_block_size=self.kvcache_block_size,
                    use_position_ids=self.use_position_ids,
                )
            elif self.attn_impl == "flash_attn":
                raise NotImplementedError(f"flash attn for {self.__class__} is not implemented yet.")
            else:
                raise NotImplementedError(f"Unknwon attn : {self.attn_impl}")
            new_layer = PhiLayer(layer, new_self_attn)
            new_layers.append(new_layer)
        new_model = PhiModel(causal_lm.model, new_layers, sliding_window_layers=self.sliding_window_layers)
        new_causal_lm = DecoderOnlyForCausalLM(causal_lm, new_model)
        return new_causal_lm


class PhiAttention(DecoderOnlyAttention):
    def __post_init__(self):
        self.q_proj = self._original_mod.q_proj
        self.k_proj = self._original_mod.k_proj
        self.v_proj = self._original_mod.v_proj
        self.o_proj = self._original_mod.dense
        self.qk_layernorm = self._original_mod.qk_layernorm
        self.rotary_ndims = self._original_mod.rotary_ndims

    def projection(self, hidden_states) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        if self.qk_layernorm:
            query_states = self._original_mod.q_layernorm(query_states)
            key_states = self._original_mod.k_layernorm(key_states)

        return query_states, key_states, value_states

    def apply_rotary_pos_embed(self, query_states, key_states, cos, sin):
        return apply_rotary_pos_emb_partial(query_states, key_states, cos, sin, ndim=self.rotary_ndims)


class PhiLayer(DecoderOnlyLayer):
    def get_post_attention_layernorm(self):
        raise NotImplementedError

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

        attn_output = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            seq_positions=seq_positions,
            past_key_values=past_key_values,
            cos=cos,
            sin=sin,
            block_tables=block_tables,
        )

        feed_forward_hidden_states = self._original_mod.mlp(hidden_states)

        hidden_states = attn_output + feed_forward_hidden_states + residual

        return hidden_states


class PhiModel(DecoderOnlyModel):
    def get_last_layernorm(self):
        return self._original_mod.final_layernorm
