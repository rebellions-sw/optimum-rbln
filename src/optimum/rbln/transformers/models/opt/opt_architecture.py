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

from ...models.decoderonly.decoderonly_architecture import (
    DecoderOnlyAttention,
    DecoderOnlyForCausalLM,
    DecoderOnlyLayer,
    DecoderOnlyModel,
    DecoderOnlyWrapper,
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
                layer.self_attn,
                self.use_attention_mask,
                kvcache_block_size=self.kvcache_block_size,
                use_position_ids=self.use_position_ids,
            )
            new_layer = OPTDecoderLayer(layer, new_self_attn)
            new_layers.append(new_layer)
        new_model = OPTModel(
            causal_lm.model.decoder,
            new_layers,
            max_seq_len=max_seq_len,
            use_learned_pos_emb=True,
            sliding_window_layers=self.sliding_window_layers,
        )
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


class OPTDecoderLayer(DecoderOnlyLayer):
    def get_pre_attention_layernorm(self) -> nn.LayerNorm:
        return self._original_mod.self_attn_layer_norm

    def get_post_attention_layernorm(self) -> nn.LayerNorm:
        return self._original_mod.final_layer_norm
