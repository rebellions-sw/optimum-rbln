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

import copy
from typing import TYPE_CHECKING, Optional, Tuple, Union

import torch
from torch import nn
from transformers.models.gemma3.modeling_gemma3 import Gemma3RMSNorm

from ..decoderonly.decoderonly_architecture import (
    AttentionOp,
    DecoderOnlyAttention,
    DecoderOnlyFlashAttention,
    DecoderOnlyForCausalLM,
    DecoderOnlyLayer,
    DecoderOnlyModel,
    DecoderOnlyWrapper,
    RotaryEmbedding,
    SlidingWindowAttentionOp,
    slice_and_unsqueeze_cos_sin,
)


if TYPE_CHECKING:
    from transformers import Gemma3ForCausalLM


class Gemma3ForCausalLMWrapper(DecoderOnlyWrapper):
    def get_rotary_emb(self, max_seq_len):
        rotary_emb_global = RotaryEmbedding(config=self.config, max_seq_len_cached=max_seq_len)

        config = copy.deepcopy(self.config)
        config.rope_theta = config.rope_local_base_freq
        config.rope_scaling = {"rope_type": "default"}
        rotary_emb_local = RotaryEmbedding(config=config, max_seq_len_cached=max_seq_len)

        return (rotary_emb_global, rotary_emb_local)

    def convert_to_rbln_causal_lm(self, causal_lm: "Gemma3ForCausalLM", max_seq_len: int):
        new_layers = []
        for layer in causal_lm.model.layers:
            if layer.is_sliding:
                new_self_attn = Gemma3Attention(
                    layer.self_attn,
                    use_attention_mask=None,  # FIXME: no use in SWA
                    use_position_ids=self.use_position_ids,
                    kvcache_block_size=self.config.sliding_window,
                )
            else:
                if self.attn_impl == "eager":
                    new_self_attn = Gemma3Attention(
                        layer.self_attn,
                        use_attention_mask=self.use_attention_mask,
                        use_position_ids=self.use_position_ids,
                        kvcache_block_size=self.kvcache_block_size,
                    )
                elif self.attn_impl == "flash_attn":
                    new_self_attn = Gemma3FlashAttention(
                        layer.self_attn,
                        kvcache_partition_len=self.kvcache_partition_len,
                        use_attention_mask=self.use_attention_mask,
                        kvcache_block_size=self.kvcache_block_size,
                        use_position_ids=self.use_position_ids,
                    )
                else:
                    raise NotImplementedError(f"Unknwon attn : {self.attn_impl}")

            new_layer = Gemma3DecoderLayer(layer, new_self_attn)
            new_layers.append(new_layer)

        new_model = Gemma3TextModel(
            causal_lm.model,
            new_layers,
            partition_len=self.kvcache_partition_len,
            max_seq_len=max_seq_len,
        )
        new_causal_lm = Gemma3ForCausalLM(causal_lm, new_model)
        return new_causal_lm

    def forward(self, *args):
        if self.phase == "decode":
            (
                input_ids_or_inputs_embeds,
                attention_mask,  # used in global layer, 2D attn_mask for padded KVcache.
                cache_position,
                position_ids,
                golbal_block_tables,
                local_block_tables,
                *past_key_values,
            ) = args
            query_position = None

        elif "prefill" in self.phase:
            (
                input_ids_or_inputs_embeds,
                attention_mask,
                cache_position,
                position_ids,
                query_position,
                golbal_block_tables,
                local_block_tables,
                *past_key_values,
            ) = args

        else:
            raise ValueError(f"Unknown phase: {self.phase}")

        if input_ids_or_inputs_embeds.ndim == 2:
            input_ids = input_ids_or_inputs_embeds
            inputs_embeds = None
        elif input_ids_or_inputs_embeds.ndim == 3:
            input_ids = None
            inputs_embeds = input_ids_or_inputs_embeds
        else:
            raise NotImplementedError(f"Unknown ndim of input : {input_ids_or_inputs_embeds.ndim}")

        if len(past_key_values) != 2 * self.num_hidden_layers:
            raise ValueError(
                f"Different past_key_values to model's config. {len(past_key_values)} != {2 * self.num_hidden_layers}"
            )

        # [key, value] * n_layer -> ( (key, value) ) * n_layer
        # cache shape : batch, n_heads, 1, max_seq_len, head_dim
        _past_key_values = []
        for i in range(self.config.num_hidden_layers):
            key_states = past_key_values[i * 2]
            value_states = past_key_values[i * 2 + 1]
            past_key_value = [key_states, value_states]
            _past_key_values.append(past_key_value)
        past_key_values = _past_key_values

        logit = self.causal_lm(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            position_ids=position_ids,
            query_position=query_position,
            past_key_values=past_key_values,
            rotary_emb=(self.rotary_emb_global, self.rotary_emb_local),
            global_block_tables=golbal_block_tables,
            local_block_tables=local_block_tables,
        )

        return logit


class Gemma3ForCausalLM(DecoderOnlyForCausalLM):
    def forward(
        self,
        input_ids: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        cache_position: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        query_position: torch.Tensor = None,
        past_key_values: Tuple[Tuple[torch.Tensor]] = None,
        rotary_emb: nn.Module = None,
        global_block_tables: Optional[torch.Tensor] = None,
        local_block_tables: Optional[torch.Tensor] = None,
    ):
        # outputs
        hidden_states = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            position_ids=position_ids,
            query_position=query_position,
            past_key_values=past_key_values,
            rotary_emb=rotary_emb,
            global_block_tables=global_block_tables,
            local_block_tables=local_block_tables,
        )

        if "prefill" in self.phase:
            hidden_states = hidden_states[:, query_position.to(torch.int).unsqueeze(0)]

        logits = self.lm_head(hidden_states)

        # Apply final logit softmaxing if configured, e.g. for Gemma2
        if getattr(self.config, "final_logit_softcapping", None) is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

        return logits


class Gemma3TextModel(DecoderOnlyModel):
    def get_local_cache_positions(self, position_ids, query_position):
        max_cache_len = self._original_mod.config.sliding_window
        valid_input_len = 1 if query_position is None else query_position + 1
        cache_seq_len = torch.clamp(position_ids, max=max_cache_len)[:, :1]  # past seen tokens
        cache_offset = (
            torch.clamp(position_ids, max=max_cache_len)[:, :1] + valid_input_len
        )  # cache offset for next steps

        return cache_seq_len, cache_offset

    def forward(
        self,
        input_ids: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        cache_position: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        query_position: torch.Tensor = None,
        past_key_values: Tuple[Tuple[torch.Tensor]] = None,
        rotary_emb: torch.nn.Module = None,
        global_block_tables: Optional[torch.Tensor] = None,
        local_block_tables: Optional[torch.Tensor] = None,
    ):
        # retrieve input_ids and inputs_embeds
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        # embed positions
        if inputs_embeds is None:
            inputs_embeds = self.get_embedding()(input_ids)

        hidden_states = inputs_embeds

        # Global Position Embeddings
        cos_global, sin_global = rotary_emb[0](hidden_states, self.max_seq_len)
        cos_global, sin_global = slice_and_unsqueeze_cos_sin(cos_global, sin_global, position_ids)

        # Local Position Embeddings
        cos_local, sin_local = rotary_emb[1](hidden_states, self.max_seq_len)
        cos_local, sin_local = slice_and_unsqueeze_cos_sin(cos_local, sin_local, position_ids)

        # (batch, seq_len) -> (batch,)
        if self.attn_impl == "flash_attn":
            seq_positions = cache_position[:, 0]
            seq_positions = self.convert_sequence_positions_for_flash_attn(
                seq_positions=seq_positions, max_seq_len=self.max_seq_len
            )
        else:
            seq_positions = cache_position[:, :1]

        sliding_cache_pos = self.get_local_cache_positions(position_ids, query_position)

        for layer in self.layers:
            if layer.is_sliding:
                hidden_states = layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    seq_positions=sliding_cache_pos,
                    past_key_values=past_key_values,
                    cos=cos_local,
                    sin=sin_local,
                    block_tables=local_block_tables,
                )
            else:
                hidden_states = layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    seq_positions=seq_positions,
                    past_key_values=past_key_values,
                    cos=cos_global,
                    sin=sin_global,
                    block_tables=global_block_tables,
                )

        hidden_states = self.get_last_layernorm()(hidden_states)
        return hidden_states


class Gemma3DecoderLayer(DecoderOnlyLayer):
    def __init__(self, layer, self_attn: "DecoderOnlyAttention"):
        super().__init__(layer, self_attn)
        self.is_sliding = self._original_mod.is_sliding

    def get_pre_feedforward_layernorm(self) -> Gemma3RMSNorm:
        return self._original_mod.pre_feedforward_layernorm

    def get_post_feedforward_layernorm(self) -> Gemma3RMSNorm:
        return self._original_mod.post_feedforward_layernorm

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        seq_positions: Union[torch.LongTensor, Tuple[torch.LongTensor]],
        past_key_values: Tuple[Tuple[torch.Tensor]],
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        block_tables: Optional[torch.Tensor] = None,
    ):
        residual = hidden_states
        hidden_states = self.get_pre_attention_layernorm()(hidden_states)

        hidden_states = self.self_attn(
            hidden_states, attention_mask, seq_positions, past_key_values, cos, sin, block_tables
        )
        hidden_states = self.get_post_attention_layernorm()(hidden_states)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.get_pre_feedforward_layernorm()(hidden_states)
        hidden_states = self._original_mod.mlp(hidden_states)
        hidden_states = self.get_post_feedforward_layernorm()(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Gemma3Attention(DecoderOnlyAttention):
    def __post_init__(self):
        self.q_proj = self._original_mod.q_proj
        self.k_proj = self._original_mod.k_proj
        self.v_proj = self._original_mod.v_proj
        self.o_proj = self._original_mod.o_proj
        self.q_norm = self._original_mod.q_norm
        self.k_norm = self._original_mod.k_norm
        self.is_sliding = self._original_mod.is_sliding

    def get_attn_scale(self):
        return self._original_mod.config.query_pre_attn_scalar**-0.5

    def get_attention(self):
        if self._original_mod.is_sliding:
            return SlidingWindowAttentionOp(
                self.num_heads,
                self.head_dim,
                self.num_key_value_heads,
                self.use_attention_mask,
                self.use_position_ids,
            )
        else:
            return AttentionOp(
                self.num_heads, self.head_dim, self.num_key_value_heads, self.use_attention_mask, self.use_position_ids
            )

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
        batch_size, query_length, _ = hidden_states.size()

        query_states, key_states, value_states = self.projection(hidden_states=hidden_states)

        query_states = query_states.view(batch_size, query_length, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, query_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, query_length, self.num_key_value_heads, self.head_dim).transpose(
            1, 2
        )

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
        query_states, key_states = self.apply_rotary_pos_embed(query_states, key_states, cos, sin)

        batch_size = query_states.shape[0]
        if batch_size > 1 and "prefill" in self.phase:
            raise NotImplementedError(f"batch size should be 1 if prefill phase, but got {batch_size}.")

        attn_output = self.attention(
            query_states,
            key_states,
            value_states,
            attention_mask,
            past_key_state=past_key_values[self.layer_idx][0],
            past_value_state=past_key_values[self.layer_idx][1],
            seq_position=seq_positions,
            scale=self.scale,
            block_tables=block_tables,
            block_size=self.kvcache_block_size,
        )

        attn_outputs = self.o_proj(attn_output)
        return attn_outputs


class Gemma3FlashAttention(DecoderOnlyFlashAttention):
    def __post_init__(self):
        self.q_proj = self._original_mod.q_proj
        self.k_proj = self._original_mod.k_proj
        self.v_proj = self._original_mod.v_proj
        self.o_proj = self._original_mod.o_proj
        self.q_norm = self._original_mod.q_norm
        self.k_norm = self._original_mod.k_norm
        self.is_sliding = self._original_mod.is_sliding

    def get_attn_scale(self):
        return self._original_mod.config.query_pre_attn_scalar**-0.5

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
        batch_size, query_length, _ = hidden_states.size()

        query_states, key_states, value_states = self.projection(hidden_states=hidden_states)

        query_states = query_states.view(batch_size, query_length, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, query_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, query_length, self.num_key_value_heads, self.head_dim).transpose(
            1, 2
        )

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
        query_states, key_states = self.apply_rotary_pos_embed(query_states, key_states, cos, sin)

        attn_output = self.attention(
            query_states,
            key_states,
            value_states,
            attention_mask,
            past_key_state=past_key_values[self.layer_idx][0],
            past_value_state=past_key_values[self.layer_idx][1],
            seq_position=seq_positions,
            scale=self.scale,
            block_tables=block_tables,
            kvcache_block_size=self.kvcache_block_size,
        )

        attn_outputs = self.o_proj(attn_output)
        return attn_outputs
