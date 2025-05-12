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

from typing import TYPE_CHECKING, Optional, Tuple, List

import torch
from transformers import PreTrainedModel
from transformers.models.gemma3.modeling_gemma3 import Gemma3RMSNorm

from ..decoderonly.decoderonly_architecture import (
    DEFAULT_FLASH_ATTN_PARTITION_LENGTH,
    DecoderOnlyAttention,
    DecoderOnlyFlashAttention,
    DecoderOnlyForCausalLM,
    DecoderOnlyLayer,
    DecoderOnlyModel,
    DecoderOnlyWrapper,
    DecoderOnlySlidingWindowAttention,
    slice_and_unsqueeze_cos_sin,
)


if TYPE_CHECKING:
    from transformers import Gemma3ForCausalLM


class Gemma3ForCausalLMWrapper(DecoderOnlyWrapper):
    def __init__(
        self,
        causal_lm: PreTrainedModel,
        max_seq_len: int,
        use_rotary_emb: bool,
        attn_impl: str,
        use_attention_mask: bool,
        kvcache_partition_len: Optional[int] = None,
        kvcache_block_size: Optional[int] = None,
    ):
        torch.nn.Module.__init__(self)
        self.config = causal_lm.config

        if use_rotary_emb:
            self.rotary_emb = self.get_rotary_emb(max_seq_len=max_seq_len)
        else:
            self.rotary_emb = None

        self.attn_impl = attn_impl
        self.kvcache_block_size = kvcache_block_size
        self.use_attention_mask = use_attention_mask
        if self.attn_impl == "flash_attn":
            self.kvcache_partition_len = kvcache_partition_len or DEFAULT_FLASH_ATTN_PARTITION_LENGTH
        elif self.attn_impl == "eager":
            self.kvcache_partition_len = None
        else:
            raise ValueError(f"Unknown attn_impl : {self.attn_impl}")

        if kvcache_partition_len and kvcache_partition_len > max_seq_len:
            raise ValueError(
                f"kvcache_partition_len({kvcache_partition_len}) should be lower"
                f" or equal to max_seq_len({max_seq_len})!"
            )

        sliding_window = self.config.sliding_window
        sliding_window_pattern = self.config.sliding_window_pattern

        self.causal_lm = self.convert_to_rbln_causal_lm(causal_lm, max_seq_len, sliding_window, sliding_window_pattern)

        self.num_hidden_layers = getattr(self.config, "num_hidden_layers", None) or getattr(self.config, "n_layer")
        self._phase = "prefill"

    def convert_to_rbln_causal_lm(
        self, causal_lm: "Gemma3ForCausalLM", max_seq_len: int, sliding_window: int, sliding_window_pattern: int
    ):
        new_layers = []
        for layer_idx, layer in enumerate(causal_lm.model.layers):
            # Global attention layer
            if (layer_idx + 1) % sliding_window_pattern == 0:
                if self.attn_impl == "eager":
                    new_self_attn = Gemma3Attention(
                        layer.self_attn, use_attention_mask=False, kvcache_block_size=self.kvcache_block_size
                    )
                elif self.attn_impl == "flash_attn":
                    new_self_attn = Gemma3FlashAttention(
                        layer.self_attn,
                        kvcache_partition_len=self.kvcache_partition_len,
                        use_attention_mask=False,
                        kvcache_block_size=self.kvcache_block_size,
                    )
                else:
                    raise NotImplementedError(f"Unknwon attn : {self.attn_impl}")
            # Local attention layer
            else:
                # TODO: implement SWA
                new_self_attn = Gemma3Attention(
                    layer.self_attn,
                    use_attention_mask=True,
                    kvcache_block_size=self.kvcache_block_size,
                )

            new_layer = Gemma3DecoderLayer(layer, new_self_attn)
            new_layers.append(new_layer)

        new_model = DecoderOnlyModel(
            causal_lm.model, new_layers, partition_len=self.kvcache_partition_len, max_seq_len=max_seq_len,
        )
        new_causal_lm = DecoderOnlyForCausalLM(causal_lm, new_model)
        return new_causal_lm


class Gemma3DecoderLayer(DecoderOnlyLayer):
    def get_pre_feedforward_layernorm(self) -> Gemma3RMSNorm:
        return self._original_mod.pre_feedforward_layernorm

    def get_post_feedforward_layernorm(self) -> Gemma3RMSNorm:
        return self._original_mod.post_feedforward_layernorm

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
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            seq_positions=seq_positions,
            past_key_values=past_key_values,
            cos=cos,
            sin=sin,
            block_tables=block_tables,
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
        self.q_norm = getattr(self._original_mod, "q_norm", None)
        self.k_norm = getattr(self._original_mod, "k_norm", None)

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

        if cos is not None and sin is not None:
            query_states, key_states = self.apply_rotary_pos_embed(query_states, key_states, cos, sin)

        batch_size = query_states.shape[0]
        if batch_size > 1 and self.phase == "prefill":
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
        self.q_norm = getattr(self._original_mod, "q_norm", None)
        self.k_norm = getattr(self._original_mod, "k_norm", None)

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

        if cos is not None and sin is not None:
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
