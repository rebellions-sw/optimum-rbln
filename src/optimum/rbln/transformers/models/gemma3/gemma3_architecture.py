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
        super().__init__()
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
            if (layer_idx + 1) % sliding_window_pattern == 0:
                if self.attn_impl == "eager":
                    new_self_attn = DecoderOnlyAttention(
                        layer.self_attn, self.use_attention_mask, kvcache_block_size=self.kvcache_block_size
                    )
                elif self.attn_impl == "flash_attn":
                    new_self_attn = DecoderOnlyFlashAttention(
                        layer.self_attn,
                        kvcache_partition_len=self.kvcache_partition_len,
                        use_attention_mask=self.use_attention_mask,
                        kvcache_block_size=self.kvcache_block_size,
                    )
                else:
                    raise NotImplementedError(f"Unknwon attn : {self.attn_impl}")
            else:
                # TODO: implement SWA
                new_self_attn = DecoderOnlyAttention(
                    layer.self_attn,
                    use_attention_mask=True,
                    kvcache_block_size=self.kvcache_block_size,
                    sliding_window_size=sliding_window,
                )

            new_layer = Gemma3DecoderLayer(layer, new_self_attn)
            new_layers.append(new_layer)

        new_model = DecoderOnlyModel(
            causal_lm.model, new_layers, partition_len=self.kvcache_partition_len, max_seq_len=max_seq_len
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

    def projection(self, hidden_states) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Projects input hidden states into query, key, and value representations.

        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_dim]

        Returns:
            Tuple of (query_states, key_states, value_states)
        """
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = self.q_norm(query_states) if self.q_norm is not None else query_states
        key_states = self.k_norm(key_states) if self.k_norm is not None else key_states

        return query_states, key_states, value_states

    def get_attn_scale(self):
        return self._original_mod.config.query_pre_attn_scalar**-0.5


class Gemma3FlashAttention(DecoderOnlyFlashAttention):
    def __init__(self, self_attn, kvcache_partition_len, kvcache_block_size, use_attention_mask):
        Gemma3Attention.__init__(self, self_attn, use_attention_mask, kvcache_block_size)
        self.kvcache_partition_size = kvcache_partition_len

    def projection(self, hidden_states) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = self.q_norm(query_states) if self.q_norm is not None else query_states
        key_states = self.k_norm(key_states) if self.k_norm is not None else key_states

        return query_states, key_states, value_states

    def get_attn_scale(self):
        return self._original_mod.config.query_pre_attn_scalar**-0.5
