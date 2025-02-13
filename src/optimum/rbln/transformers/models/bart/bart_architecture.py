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

from typing import Tuple

import torch
from torch import nn
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_attention_mask,
)
from transformers.utils import logging

from ..seq2seq.seq2seq_architecture import (
    Seq2SeqDecoder,
    Seq2SeqDecoderLayer,
    Seq2SeqDecoderWrapper,
    Seq2SeqEncoderWrapper,
    Seq2SeqForConditionalGeneration,
    Seq2SeqSelfAttention,
)


logger = logging.get_logger(__name__)


class BartWrapper:
    def __init__(self, model: nn.Module, enc_max_seq_len: int):
        self.encoder = Seq2SeqEncoderWrapper(model, enc_max_seq_len)
        self.decoder = BartDecoderWrapper(model)


class BartDecoderWrapper(Seq2SeqDecoderWrapper):
    def convert_to_rbln_conditional_generation(self, model: nn.Module):
        new_layers = []
        for layer in model.get_decoder().layers:
            self_attn = BartSelfAttention(layer.self_attn)
            new_layers.append(BartDecoderLayer(layer, self_attn))

        decoder_model = BartDecoder(model.get_decoder(), new_layers)
        new_model = BartForConditionalGeneration(model, decoder_model)

        return new_model


class BartForConditionalGeneration(Seq2SeqForConditionalGeneration):
    has_rescaling = False

    def __post_init__(self):
        self.scaling = self.config.d_model**-0.5


class BartDecoder(Seq2SeqDecoder):
    has_pos_emb = True

    def __post_init__(self):
        self.embed_positions = self._original_mod.embed_positions
        self.layernorm_embedding = self._original_mod.layernorm_embedding
        self.embed_scale = getattr(self._original_mod, "embed_scale", None)

    def prepare_attn_mask(self, attention_mask, encoder_attention_mask, **kwargs):
        attention_mask = attention_mask[:, None, None, :]
        encoder_attention_mask = _prepare_4d_attention_mask(encoder_attention_mask, torch.float32, tgt_len=1)

        return attention_mask, encoder_attention_mask

    def apply_position_embedding(self, inputs_embeds, cache_position):
        hidden_all = []
        for i in range(inputs_embeds.shape[0]):
            positions_idx = cache_position[i]
            position_weight = self.embed_positions.weight[2:]
            position = position_weight[positions_idx]
            batch_hidden = position + inputs_embeds[i]
            hidden_all.append(batch_hidden)
        hidden_states = torch.stack(hidden_all, dim=0)

        hidden_states = self.layernorm_embedding(hidden_states)

        return hidden_states

    def get_embedding(self):
        if self.embed_scale is not None:
            return lambda x: self.embed_tokens(x) * self.embed_scale
        else:
            return self.embed_tokens


class BartLayerFF(nn.Module):
    def __init__(self, decoder_layer):
        super().__init__()
        self.fc1 = decoder_layer.fc1
        self.fc2 = decoder_layer.fc2
        self.activation_fn = decoder_layer.activation_fn
        self.layer_norm = decoder_layer.final_layer_norm

    def forward(self, hidden_states):
        # Residual Connection
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class BartDecoderLayer(Seq2SeqDecoderLayer):
    def __post_init__(self):
        self.self_attn_layer_norm = self._original_mod.self_attn_layer_norm
        self.encoder_attn = self._original_mod.encoder_attn
        self.encoder_attn_layer_norm = self._original_mod.encoder_attn_layer_norm
        self.ff_layer = BartLayerFF(self._original_mod)

    def pre_self_attn_layer_norm(self, hidden_states):
        return hidden_states

    def post_self_attn_layer_norm(self, hidden_states):
        return self.self_attn_layer_norm(hidden_states)

    def pre_cross_attn_layer_norm(self, hidden_states):
        return hidden_states

    def post_cross_attn_layer_norm(self, hidden_states):
        return self.encoder_attn_layer_norm(hidden_states)


class BartSelfAttention(Seq2SeqSelfAttention):
    def __post_init__(self):
        self.q_proj = self._original_mod.q_proj
        self.k_proj = self._original_mod.k_proj
        self.v_proj = self._original_mod.v_proj
        self.out_proj = self._original_mod.out_proj
        self.num_heads = self._original_mod.num_heads
        self.head_dim = self._original_mod.embed_dim // self._original_mod.num_heads
        self.scaling = self.head_dim**-0.5
        self.attn_decode = torch.ops.rbln_custom_ops.attn_decode

    def projection(self, hidden_states) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query_states = self.q_proj(hidden_states) * self.scaling
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        return query_states, key_states, value_states
