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

from typing import Optional, Tuple

import torch
from torch import nn
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_attention_mask,
)
from transformers.utils import logging

from ..seq2seq.seq2seq_architecture import (
    Seq2SeqCrossAttention,
    Seq2SeqDecoder,
    Seq2SeqDecoderLayer,
    Seq2SeqDecoderWrapper,
    Seq2SeqEncoderWrapper,
    Seq2SeqForConditionalGeneration,
    Seq2SeqSelfAttention,
)


logger = logging.get_logger(__name__)


class PegasusWrapper:
    def __init__(self, model: nn.Module, enc_max_seq_len: int, use_attention_mask: bool):
        self.encoder = Seq2SeqEncoderWrapper(model, enc_max_seq_len)
        # self.encoder = PegasusEncoderWrapper(model, enc_max_seq_len)
        self.decoder = PegasusDecoderWrapper(model, use_attention_mask=use_attention_mask)

class PegasusEncoderWrapper(Seq2SeqEncoderWrapper):
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        b_idx: torch.Tensor,
        *cross_key_values: Tuple[torch.Tensor],
    ) -> Tuple[torch.Tensor]:
        # 1. get encoder last_hidden_states
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = encoder_outputs[0]

        # 2. pre-compute cross_attention's past_key_value which used in decoder phase.
        cross_kv = []
        for k_proj, v_proj in zip(self.cross_k_projects, self.cross_v_projects):
            past_k = (
                k_proj(last_hidden_states).view(1, self.encoder_max_length, self.num_heads, self.d_kv).transpose(1, 2)
            )
            past_v = (
                v_proj(last_hidden_states).view(1, self.encoder_max_length, self.num_heads, self.d_kv).transpose(1, 2)
            )

            cross_kv.append(past_k)
            cross_kv.append(past_v)

        # 3. update the cross_attention's past_key_value direct to the device-dram for optimization.
        batch_axis = torch.tensor(0, dtype=torch.int16)
        cross_key_values = list(cross_key_values)
        # import pdb; pdb.set_trace() # last_hidden_states 계산은 문제없음 -> 결국 Decoder 문제?
        for i in range(self.n_layer * 2):
            cross_key_values[i] = torch.ops.rbln_custom_ops.rbln_cache_update(
                cross_key_values[i], cross_kv[i], b_idx[0], batch_axis
            )

        return last_hidden_states, cross_key_values


class PegasusDecoderWrapper(Seq2SeqDecoderWrapper):
    def convert_to_rbln_conditional_generation(self, model: nn.Module):
        new_layers = []
        for layer in model.get_decoder().layers:
            self_attn = PegasusSelfAttention(layer.self_attn, use_attention_mask=self.use_attention_mask)
            cross_attn = PegasusCrossAttention(layer.encoder_attn)
            new_layers.append(PegasusDecoderLayer(layer, self_attn, cross_attn))

        decoder_model = PegasusDecoder(model.get_decoder(), new_layers)
        new_model = PegasusForConditionalGeneration(model, decoder_model)

        return new_model


class PegasusForConditionalGeneration(Seq2SeqForConditionalGeneration):
    def forward(
        self,
        input_ids,
        attention_mask,
        encoder_attention_mask,
        self_past_key_values,
        cross_past_key_values,
        cache_position,
        block_tables: Optional[torch.Tensor] = None,
    ):
        hidden_states = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            self_past_key_values=self_past_key_values,
            cross_past_key_values=cross_past_key_values,
            cache_position=cache_position,
            block_tables=block_tables,
        )

        if self.has_rescaling and self.config.tie_word_embeddings:
            hidden_states = hidden_states * self.scaling

        lm_logits = self.lm_head(hidden_states[0])
        output_hidden_states = hidden_states[1:]

        return lm_logits, (output_hidden_states)


class PegasusDecoder(Seq2SeqDecoder):
    has_pos_emb = True

    def __post_init__(self):
        self.embed_positions = self._original_mod.embed_positions
        self.embed_scale = getattr(self._original_mod, "embed_scale", None)
        self.final_layer_norm = getattr(self._original_mod, "layer_norm", None)

    def prepare_attn_mask(self, attention_mask, encoder_attention_mask, **kwargs):
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]
        encoder_attention_mask = _prepare_4d_attention_mask(encoder_attention_mask, torch.float32, tgt_len=1)

        return attention_mask, encoder_attention_mask

    def apply_position_embedding(self, inputs_embeds, cache_position):
        hidden_all = []
        for i in range(inputs_embeds.shape[0]):
            positions_idx = cache_position[i]
            position_weight = self.embed_positions.weight
            position = position_weight[positions_idx]
            batch_hidden = position + inputs_embeds[i]
            hidden_all.append(batch_hidden)
        hidden_states = torch.stack(hidden_all, dim=0)

        return hidden_states

    def get_embedding(self):
        if self.embed_scale is not None:
            return lambda x: self.embed_tokens(x) * self.embed_scale
        else:
            return self.embed_tokens
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        self_past_key_values: torch.Tensor,
        cross_past_key_values: torch.Tensor,
        cache_position: torch.Tensor,
        block_tables: Optional[torch.Tensor] = None,
    ):
        output_hidden_states=True
        # embedding
        hidden_states = self.get_embedding()(input_ids)
        attention_mask, encoder_attention_mask = self.prepare_attn_mask(
            attention_mask, encoder_attention_mask, cache_position=cache_position
        )

        if self.has_pos_emb:
            hidden_states = self.apply_position_embedding(hidden_states, cache_position)

        all_hidden_states = () if output_hidden_states else None
        
        # iterate decoder_layer
        for decoder_layer, self_past_key_value, cross_past_key_value in zip(
            self.layers, self_past_key_values, cross_past_key_values
        ):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                self_past_key_value=self_past_key_value,
                cross_past_key_value=cross_past_key_value,
                cache_position=cache_position,
                block_tables=block_tables,
            )

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)
        
        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return hidden_states, all_hidden_states


class PegasusLayerFF(nn.Module):
    def __init__(self, decoder_layer):
        super().__init__()
        self.fc1 = decoder_layer.fc1
        self.fc2 = decoder_layer.fc2
        self.activation_fn = decoder_layer.activation_fn
        self.layer_norm = decoder_layer.final_layer_norm

    def forward(self, hidden_states):
        # Residual Connection
        residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class PegasusDecoderLayer(Seq2SeqDecoderLayer):
    def __post_init__(self):
        self.self_attn_layer_norm = self._original_mod.self_attn_layer_norm
        self.encoder_attn = self._original_mod.encoder_attn
        self.encoder_attn_layer_norm = self._original_mod.encoder_attn_layer_norm
        self.ff_layer = PegasusLayerFF(self._original_mod)

    def pre_self_attn_layer_norm(self, hidden_states):
        return self.self_attn_layer_norm(hidden_states)

    def post_self_attn_layer_norm(self, hidden_states):
        return hidden_states

    def pre_cross_attn_layer_norm(self, hidden_states):
        return self.encoder_attn_layer_norm(hidden_states)

    def post_cross_attn_layer_norm(self, hidden_states):
        return hidden_states


class PegasusSelfAttention(Seq2SeqSelfAttention):
    def __post_init__(self, use_attention_mask: bool = True):
        self.q_proj = self._original_mod.q_proj
        self.k_proj = self._original_mod.k_proj
        self.v_proj = self._original_mod.v_proj
        self.out_proj = self._original_mod.out_proj
        self.num_heads = self._original_mod.num_heads
        self.head_dim = self._original_mod.embed_dim // self._original_mod.num_heads
        self.scaling = self.head_dim**-0.5
        if use_attention_mask:
            self.attn_decode = torch.ops.rbln_custom_ops.paged_attn_decode
        else:
            self.attn_decode = torch.ops.rbln_custom_ops.paged_causal_attn_decode

    def projection(self, hidden_states) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query_states = self.q_proj(hidden_states) * self.scaling
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        return query_states, key_states, value_states

class PegasusCrossAttention(Seq2SeqCrossAttention):
    def __post_init__(self):
        self.q_proj = self._original_mod.q_proj
        self.k_proj = self._original_mod.k_proj
        self.v_proj = self._original_mod.v_proj
        self.out_proj = self._original_mod.out_proj
        self.num_heads = self._original_mod.num_heads
        self.head_dim = self._original_mod.embed_dim // self._original_mod.num_heads
        self.embed_dim = self._original_mod.embed_dim