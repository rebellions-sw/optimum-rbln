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

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from ..decoderonly.decoderonly_architecture import (
    DecoderOnlyAttention,
    DecoderOnlyLayer,
    DecoderOnlyWrapper,
    RotaryEmbedding,
)


class Qwen3Wrapper(DecoderOnlyWrapper):
    def get_rbln_attn_class(self):
        return Qwen3Attention


class Qwen3Attention(DecoderOnlyAttention):
    def __post_init__(self):
        self.k_proj = self._original_mod.k_proj
        self.v_proj = self._original_mod.v_proj
        self.q_proj = self._original_mod.q_proj
        self.o_proj = self._original_mod.o_proj
        self.q_norm = self._original_mod.q_norm
        self.k_norm = self._original_mod.k_norm


class Qwen3ModelWrapper(nn.Module):
    def __init__(
        self,
        model,
        attn_impl=None,
        use_inputs_embeds=None,
        use_attention_mask=None,
        use_rotary_emb=None,
        cache_impl=None,
        kvcache_partition_len=None,
        max_seq_len=None,
        kvcache_block_size=None,
        sliding_window=None,
        sliding_window_layers=None,
    ):
        super().__init__()
        self.config = model.config

        if use_rotary_emb:
            rotary_embs = self.get_rotary_emb(max_seq_len=max_seq_len)
            if isinstance(rotary_embs, tuple):
                self.rotary_emb_global, self.rotary_emb_local = rotary_embs
            else:
                self.rotary_emb = rotary_embs
        else:
            self.rotary_emb = None

        self._original_mod = model
        self.use_inputs_embeds = use_inputs_embeds
        self.attn_impl = attn_impl
        self.cache_impl = cache_impl
        self.use_attention_mask = use_attention_mask
        self.kvcache_partition_len = kvcache_partition_len
        self.kvcache_block_size = kvcache_block_size
        self.max_seq_len = max_seq_len
        self.sliding_window = sliding_window
        self.sliding_window_layers = sliding_window_layers
        self.model = self.convert_to_rbln_model(model)

    def get_rotary_emb(self, max_seq_len):
        return RotaryEmbedding(config=self.config, max_seq_len_cached=max_seq_len)

    def convert_to_rbln_model(self, base_model: PreTrainedModel):
        for layer_idx, layer in enumerate(base_model.layers):
            is_sliding = layer_idx in self.sliding_window_layers
            new_self_attn = Qwen3Attention(
                layer.self_attn,
                self.use_attention_mask if not is_sliding else True,
                use_position_ids=None,
                kvcache_block_size=self.sliding_window
                if layer_idx in self.sliding_window_layers
                else self.kvcache_block_size,
                is_sliding=is_sliding,
                attn_impl=self.attn_impl if not is_sliding else "eager",
                kvcache_partition_len=self.kvcache_partition_len,
            )
            base_model.layers[layer_idx] = DecoderOnlyLayer(layer, new_self_attn)

        return base_model

    @property
    def hidden_multiplier(self):
        return 1

    def get_last_layernorm(self) -> nn.LayerNorm:
        return self._original_mod.norm

    def get_embedding(self) -> nn.Embedding:
        return self._original_mod.embed_tokens

    def get_pos_embedding(self) -> nn.Embedding:
        raise NotImplementedError(
            "The 'get_pos_embedding' method is not implemented. Please define this method in a subclass."
        )

    def convert_sequence_positions_for_flash_attn(self, seq_positions, max_seq_len):
        if self.attn_impl not in ["flash_attn"]:
            raise NotImplementedError(f"Unknown attn_impl ({self.attn_impl}).")
        partition_len = self.kvcache_partition_len
        num_partition = max_seq_len // partition_len

        cs = seq_positions.repeat(num_partition, 1).transpose(0, 1)
        pidx = torch.arange(num_partition)
        cache_pos_for_partitions = torch.clamp(cs - pidx * partition_len, 0, partition_len)
        return cache_pos_for_partitions

    def get_local_cache_positions(self, position_ids, query_position):
        max_cache_len = self.model.config.sliding_window
        valid_input_len = 1 if query_position is None else query_position + 1
        cache_seq_len = torch.clamp(position_ids, max=max_cache_len)[:, :1]  # past seen tokens
        cache_offset = (
            torch.clamp(position_ids, max=max_cache_len)[:, :1] + valid_input_len
        )  # cache offset for next steps

        return cache_seq_len, cache_offset

    def prepare_forward_args(self, *args):
        args = list(args)
        input_ids = None if self.use_inputs_embeds else args.pop(0)
        inputs_embeds = args.pop(0) if self.use_inputs_embeds else None
        cache_position = args.pop(0)
        global_block_tables = args.pop(0) if self.cache_impl in ["hybrid", "static"] else None
        local_block_tables = args.pop(0) if self.cache_impl in ["hybrid", "sliding_window"] else None
        query_position = args.pop(0) if self.sliding_window else None
        attention_mask = args.pop(0) if self.use_attention_mask else None
        position_ids = None
        past_key_values = args

        if len(past_key_values) != 2 * self.config.num_hidden_layers:
            raise ValueError(
                f"Different past_key_values to model's config. {len(past_key_values)} != {2 * self.config.num_hidden_layers}"
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

        if hasattr(self, "rotary_emb_global") and hasattr(self, "rotary_emb_local"):
            rotary_emb = (self.rotary_emb_global, self.rotary_emb_local)
        else:
            rotary_emb = self.rotary_emb

        return (
            input_ids,
            inputs_embeds,
            cache_position,
            global_block_tables,
            local_block_tables,
            attention_mask,
            position_ids,
            query_position,
            past_key_values,
            rotary_emb,
        )

    def forward(self, *args):
        (
            input_ids,
            inputs_embeds,
            cache_position,
            global_block_tables,
            local_block_tables,
            attention_mask,
            position_ids,
            query_position,
            past_key_values,
            rotary_emb,
        ) = self.prepare_forward_args(*args)

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
        position_ids = position_ids if position_ids is not None else cache_position
        if rotary_emb is not None:
            if isinstance(rotary_emb, torch.Tensor):
                cos = rotary_emb[0]
                sin = rotary_emb[1]
            else:
                cos, sin = rotary_emb(hidden_states, self.max_seq_len)  # dtype carrier, max_seq_len
                cos, sin = slice_and_unsqueeze_cos_sin(cos, sin, position_ids)
        else:
            batch_size = inputs_embeds.shape[0]
            if position_ids.shape[0] > 1:
                position_embeds = []
                for b_idx in range(batch_size):
                    position_embed = self.get_pos_embedding()(position_ids[b_idx])
                    position_embeds.append(position_embed)

                position_embeds = torch.cat(position_embeds, dim=0).unsqueeze(1)
            else:
                position_embeds = self.get_pos_embedding()(position_ids)
            hidden_states = hidden_states + position_embeds
            cos, sin = None, None

        # Get sequence positions for flash attention
        if self.attn_impl == "flash_attn":
            seq_positions = cache_position[:, 0]
            seq_positions = self.convert_sequence_positions_for_flash_attn(
                seq_positions=seq_positions, max_seq_len=self.max_seq_len
            )
        else:
            seq_positions = cache_position[:, :1]

        # Get local cache positions for sliding window layers
        if len(self.sliding_window_layers) > 0:
            sliding_cache_pos = self.get_local_cache_positions(position_ids, query_position)

        for layer_idx, layer in enumerate(self.model.layers):
            is_sliding = True if layer_idx in self.sliding_window_layers else False
            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                seq_positions=sliding_cache_pos if is_sliding else seq_positions,
                past_key_values=past_key_values,
                cos=cos,
                sin=sin,
                block_tables=local_block_tables if is_sliding else global_block_tables,
            )

        hidden_states = self.get_last_layernorm()(hidden_states)
        return hidden_states


def slice_and_unsqueeze_cos_sin(cos, sin, cache_position, unsqueeze_dim=1):
    """Slice cos[cache_position], sin[cache_position] vector for the query."""
    if cache_position.shape[0] > 1:
        cos_all = []
        sin_all = []
        for i in range(cache_position.shape[0]):
            cos_all.append(cos[cache_position[i : i + 1]].unsqueeze(unsqueeze_dim))
            sin_all.append(sin[cache_position[i : i + 1]].unsqueeze(unsqueeze_dim))
        cos = torch.cat(cos_all, dim=0)
        sin = torch.cat(sin_all, dim=0)
    else:
        cos = cos[cache_position].unsqueeze(unsqueeze_dim)
        sin = sin[cache_position].unsqueeze(unsqueeze_dim)

    return cos, sin
