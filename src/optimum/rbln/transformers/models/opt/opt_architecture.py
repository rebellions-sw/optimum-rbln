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

from ...models.decoderonly.decoderonly_architecture import (
    DecoderOnlyAttention,
    DecoderOnlyForCausalLM,
    DecoderOnlyLayer,
    DecoderOnlyModel,
    DecoderOnlyWrapper,
    slice_and_unsqueeze_cos_sin,
)


if TYPE_CHECKING:
    from transformers import OPTForCausalLM


class OPTWrapper(DecoderOnlyWrapper):
    def convert_to_rbln_causal_lm(self, causal_lm: "OPTForCausalLM", max_seq_len: int):
        if self.attn_impl != "eager":
            raise NotImplementedError(f"flash attention ({self.attn_impl}) is not implemented for {self.__class__}")

        new_layers = []

        for layer in causal_lm.model.decoder.layers:
            new_self_attn = OPTAttention(
                layer.self_attn, self.use_attention_mask, kvcache_block_size=self.kvcache_block_size
            )
            new_layer = OPTDecoderLayer(layer, new_self_attn)
            new_layers.append(new_layer)
        new_model = OPTModel(causal_lm.model.decoder, new_layers, max_seq_len=max_seq_len)
        new_causal_lm = DecoderOnlyForCausalLM(causal_lm, new_model)
        return new_causal_lm


class OPTAttention(DecoderOnlyAttention):
    def __post_init__(self):
        self.k_proj = self._original_mod.k_proj
        self.v_proj = self._original_mod.v_proj
        self.q_proj = self._original_mod.q_proj
        self.o_proj = self._original_mod.out_proj


class OPTModel(DecoderOnlyModel):
    def get_embedding(self) -> nn.Embedding:
        return self._original_mod.embed_tokens

    def get_pos_embedding(self):
        return self._original_mod.embed_positions

    def get_last_layernorm(self) -> nn.LayerNorm:
        return self._original_mod.final_layer_norm

    def forward(
        self,
        input_ids: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        cache_position: torch.Tensor = None,
        past_key_values: Tuple[Tuple[torch.Tensor]] = None,
        rotary_emb: nn.Module = None,
        block_tables: Optional[torch.Tensor] = None,
    ):
        # retrieve input_ids and inputs_embeds
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        # embed positions
        if inputs_embeds is None:
            inputs_embeds = self.get_embedding()(input_ids)

        hidden_states = inputs_embeds * self.hidden_multiplier

        # get cos,sin vector if needed
        if rotary_emb is not None:
            if isinstance(rotary_emb, torch.Tensor):
                cos = rotary_emb[0]
                sin = rotary_emb[1]
            else:
                cos, sin = rotary_emb(hidden_states, self.max_seq_len)  # dtype carrier, max_seq_len
                cos, sin = slice_and_unsqueeze_cos_sin(cos, sin, cache_position)
        else:
            batch_size = inputs_embeds.shape[0]
            hidden_all = []
            for i in range(batch_size):
                positions_idx = cache_position[i]
                position_weight = self.get_pos_embedding().weight[2:]
                position = position_weight[positions_idx]
                batch_hidden = position + inputs_embeds[i]
                hidden_all.append(batch_hidden)
            hidden_states = torch.stack(hidden_all, dim=0)
            cos, sin = None, None

        if self.attn_impl == "flash_attn":
            seq_positions = cache_position[:, 0]
            seq_positions = self.convert_sequence_positions_for_flash_attn(
                seq_positions=seq_positions, max_seq_len=self.max_seq_len
            )
        else:
            seq_positions = cache_position[:, :1]

        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                seq_positions=seq_positions,
                past_key_values=past_key_values,
                cos=cos,
                sin=sin,
                block_tables=block_tables,
            )

        hidden_states = self.get_last_layernorm()(hidden_states)
        return hidden_states


class OPTDecoderLayer(DecoderOnlyLayer):
    def get_pre_attention_layernorm(self) -> nn.LayerNorm:
        return self._original_mod.self_attn_layer_norm

    def get_post_attention_layernorm(self) -> nn.LayerNorm:
        return self._original_mod.final_layer_norm
