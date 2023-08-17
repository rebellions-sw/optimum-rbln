# Copyright 2024 Rebellions Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Portions of this software are licensed under the Apache License,
# Version 2.0. See the NOTICE file distributed with this work for
# additional information regarding copyright ownership.

# All other portions of this software, including proprietary code,
# are the intellectual property of Rebellions Inc. and may not be
# copied, modified, or distributed without prior written permission
# from Rebellions Inc.

from typing import Tuple

import torch
from torch import nn
from transformers.utils import logging

from ....ops import register_rbln_custom_attention_add_softmax
from ..seq2seq.seq2seq_architecture import (
    Seq2SeqDecoder,
    Seq2SeqDecoderLayer,
    Seq2SeqDecoderWrapper,
    Seq2SeqEncoderWrapper,
    Seq2SeqForConditionalGeneration,
    Seq2SeqSelfAttention,
)


logger = logging.get_logger(__name__)


class T5Wrapper:
    def __init__(self, model: nn.Module, enc_max_seq_len: int, dec_max_seq_len: int = None):
        self.encoder = T5EncoderWrapper(model, enc_max_seq_len)
        self.decoder = T5DecoderWrapper(model, dec_max_seq_len=dec_max_seq_len)


class T5EncoderWrapper(Seq2SeqEncoderWrapper):
    def __post_init__(self, model: nn.Module):
        self.n_layer = getattr(self.config, "num_layers")
        self.cross_k_projects, self.cross_v_projects = self._extract_cross_kv_projects(model.get_decoder().block)
        self.num_heads = self.config.num_heads
        self.d_kv = self.config.d_kv

    def _extract_cross_kv_projects(self, t5_block: nn.Module):
        return (
            # different from bart
            nn.ModuleList(t5_block[i].layer[1].EncDecAttention.k for i in range(self.n_layer)),
            nn.ModuleList(t5_block[i].layer[1].EncDecAttention.v for i in range(self.n_layer)),
        )


class T5DecoderWrapper(Seq2SeqDecoderWrapper):
    def __post_init__(self, model, dec_max_seq_len: int = None):
        register_rbln_custom_attention_add_softmax()
        self.num_layers = self.config.num_layers
        self.conditional_generation = self.convert_to_rbln_conditional_generation(model, dec_max_seq_len)

    def convert_to_rbln_conditional_generation(self, model: nn.Module, dec_max_seq_len: int):
        new_blocks = []
        for block in model.get_decoder().block:
            self_attn = T5LayerSelfAttention(block.layer[0].SelfAttention)
            block = T5Block(block, self_attn)
            new_blocks.append(block)

        decoder_model = T5Decoder(model.get_decoder(), new_blocks, dec_max_seq_len=dec_max_seq_len)
        new_model = T5ForConditionalGeneration(model, decoder_model)

        return new_model


class T5ForConditionalGeneration(Seq2SeqForConditionalGeneration):
    has_rescaling = True

    def __post_init__(self):
        self.scaling = self.config.d_model**-0.5


class T5Decoder(Seq2SeqDecoder):
    has_pos_emb = False

    def __post_init__(self, dec_max_seq_len: int = None):
        self.invert_attention_mask = self._original_mod.invert_attention_mask
        self._dec_position_bias = self.precompute_dec_position_bias(self._original_mod, dec_max_seq_len)

    def precompute_dec_position_bias(self, model, dec_max_length):
        attn_layer = model.block[0].layer[0].SelfAttention
        return attn_layer.compute_bias(dec_max_length, dec_max_length)

    def prepare_attn_mask(self, attention_mask, encoder_attention_mask, cache_position):
        attention_mask = self.invert_attention_mask(attention_mask)
        encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)

        b_size = attention_mask.shape[0]
        batch_decoder_position_bias = []
        for i in range(b_size):
            batch_position_bias = self._dec_position_bias[:, :, cache_position[i][0]].unsqueeze(2)
            batch_decoder_position_bias.append(batch_position_bias)
        position_bias = torch.cat(batch_decoder_position_bias, dim=0)

        attention_mask = position_bias + attention_mask

        return attention_mask, encoder_attention_mask


class T5Block(Seq2SeqDecoderLayer):
    def __post_init__(self):
        self.self_attn_layer_norm = self._original_mod.layer[0].layer_norm
        self.encoder_attn_layer_norm = self._original_mod.layer[1].layer_norm
        self.encoder_attn = T5CrossAttention(self._original_mod.layer[1].EncDecAttention)
        self.ff_layer = self._original_mod.layer[2]

    def pre_self_attn_layer_norm(self, hidden_states):
        return self.self_attn_layer_norm(hidden_states)

    def post_self_attn_layer_norm(self, hidden_states):
        return hidden_states

    def pre_cross_attn_layer_norm(self, hidden_states):
        return self.encoder_attn_layer_norm(hidden_states)

    def post_cross_attn_layer_norm(self, hidden_states):
        return hidden_states


class T5LayerSelfAttention(Seq2SeqSelfAttention):
    def __post_init__(self):
        self.q_proj = self._original_mod.q
        self.k_proj = self._original_mod.k
        self.v_proj = self._original_mod.v
        self.out_proj = self._original_mod.o
        self.num_heads = self._original_mod.n_heads
        self.head_dim = self._original_mod.key_value_proj_dim
        self.attn_decode = torch.ops.rbln_custom_ops.attn_decode_add_softmax

    def projection(self, hidden_states) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        return query_states, key_states, value_states


class T5CrossAttention(nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.attn = attn

    def forward(
        self,
        hidden_states: torch.Tensor = None,
        past_key_value: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        key_value_states: torch.Tensor = None,
    ):
        return self.attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            position_bias=attention_mask,
            key_value_states=key_value_states,
        )
