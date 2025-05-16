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
    RotaryEmbedding,
    slice_and_unsqueeze_cos_sin,
    AttentionOp,
    FlashAttentionOp,
    SlidingWindowAttentionOp,
    
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
            self.rotary_emb_global, self.rotary_emb_local = self.get_rotary_emb(max_seq_len=max_seq_len)
        else:
            self.rotary_emb_global, self.rotary_emb_local = None

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

        self.causal_lm = self.convert_to_rbln_causal_lm(causal_lm, max_seq_len, sliding_window)

        self.num_hidden_layers = getattr(self.config, "num_hidden_layers", None) or getattr(self.config, "n_layer")
        self._phase = "prefill"

    def get_rotary_emb(self, max_seq_len):
        rotary_emb_global = RotaryEmbedding(config=self.config, max_seq_len_cached=max_seq_len)
        config = copy.deepcopy(self.config)
        config.rope_theta = config.rope_local_base_freq
        config.rope_scaling = {"rope_type": "default"}
        rotary_emb_local = RotaryEmbedding(config=config, max_seq_len_cached=max_seq_len)

        return (rotary_emb_global, rotary_emb_local)

    def convert_to_rbln_causal_lm(
        self, causal_lm: "Gemma3ForCausalLM", max_seq_len: int, sliding_window: int, sliding_window_pattern: int
    ):
        new_layers = []
        for layer in enumerate(causal_lm.model.layers):
            # Global attention layer
            if not layer.is_sliding:
                if self.attn_impl == "eager":
                    new_self_attn = Gemma3Attention(
                        layer.self_attn, use_attention_mask=True, kvcache_block_size=self.kvcache_block_size
                    )
                elif self.attn_impl == "flash_attn":
                    new_self_attn = Gemma3FlashAttention(
                        layer.self_attn,
                        kvcache_partition_len=self.kvcache_partition_len,
                        use_attention_mask=True,
                        kvcache_block_size=self.kvcache_block_size,
                    )
                else:
                    raise NotImplementedError(f"Unknwon attn : {self.attn_impl}")
            # Local attention layer
            else:
                # TODO: implement SWA
                new_self_attn = Gemma3SlidingWindowAttention(
                    layer.self_attn,
                    use_attention_mask=False,
                    kvcache_block_size=self.kvcache_block_size,
                )

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
                attention_mask,
                cache_position,
                position_ids,
                block_tables,
                *past_key_values,
            ) = args
            query_position = None
        
        elif self.phase == "prefill":
            (
                input_ids_or_inputs_embeds,
                attention_mask,
                cache_position,
                position_ids,
                query_position,
                batch_position,
                block_tables,
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
            batch_position=batch_position,
            past_key_values=past_key_values,
            rotary_emb=[self.rotary_emb_global, self.rotary_emb_local],
            block_tables=block_tables,
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
        batch_position: torch.Tensor = None,
        past_key_values: Tuple[Tuple[torch.Tensor]] = None,
        rotary_emb: nn.Module = None,
        block_tables: Optional[torch.Tensor] = None,
    ):
        # outputs
        hidden_states = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            position_ids=position_ids,
            query_position=query_position,
            batch_position=batch_position,
            past_key_values=past_key_values,
            rotary_emb=rotary_emb,
            block_tables=block_tables,
        )

        if self.phase == "prefill":
            hidden_states = hidden_states[:, query_position.to(torch.int).unsqueeze(0)]

        logits = self.lm_head(hidden_states)

        # Apply final logit softmaxing if configured, e.g. for Gemma2
        if getattr(self.config, "final_logit_softcapping", None) is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

        return logits


class Gemma3TextModel(DecoderOnlyModel):
    def get_sliding_cache_positions(self, position_ids, query_position):
        
        max_cache_len = self._original_mod.config.sliding_window
        valid_input_len = 1 if query_position is None else query_position + 1
        cache_seq_len = torch.clamp(position_ids, max=max_cache_len)[:, :1]
        cache_offset = torch.clamp(position_ids, max=max_cache_len)[:, :1] + valid_input_len

        return cache_seq_len, cache_offset

    def forward(
        self,
        input_ids: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        cache_position: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        query_position: torch.Tensor = None,
        batch_position: torch.Tensor = None,
        past_key_values: Tuple[Tuple[torch.Tensor]] = None,
        rotary_emb: torch.nn.Module = None,
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
    
        sliding_cache_pos = self.get_sliding_cache_positions(position_ids, query_position)

        for layer in self.layers:            
            if layer.is_sliding:
                hidden_states = layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    seq_positions=sliding_cache_pos,
                    past_key_values=past_key_values,
                    cos=cos_local,
                    sin=sin_local,
                    batch_position=batch_position,
                )
            else:
                hidden_states = layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    seq_positions=seq_positions,
                    past_key_values=past_key_values,
                    cos=cos_global,
                    sin=sin_global,
                    block_tables=block_tables,
                )

        hidden_states = self.get_last_layernorm()(hidden_states)
        return hidden_states


class Gemma3DecoderLayer(DecoderOnlyLayer):
    def __init__(self, layer, self_attn: "DecoderOnlyAttention"):
        super().__init__()
        self._original_mod = layer
        self.self_attn = self_attn
        self._phase = "prefill"
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
        batch_position: Optional[torch.Tensor] = None,
    ):
        residual = hidden_states
        hidden_states = self.get_pre_attention_layernorm()(hidden_states)
        
        if self.is_sliding:
            args = (
                hidden_states,
                seq_positions,
                past_key_values,
                cos,
                sin,
                batch_position
            )
        else:
            args = (
                hidden_states,
                attention_mask,
                seq_positions,
                past_key_values,
                cos,
                sin,
                block_tables  
            )
            
        
        hidden_states = self.self_attn(*args)
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
    
    
    def get_attention(self):
        return Gemma3FlashAttentionOp(
            self.num_heads,
            self.head_dim,
            self.num_key_value_heads,
            self.use_attention_mask,
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

class Gemma3SlidingWindowAttention(Gemma3Attention):
    def get_attention(self):
        return SlidingWindowAttentionOp(
            self.num_heads,
            self.head_dim,
            self.num_key_value_heads,
            self.use_attention_mask,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        seq_positions: torch.LongTensor,
        past_key_values: Tuple[Tuple[torch.Tensor]],
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        batch_position: Optional[torch.Tensor] = None,
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
            past_key_state=past_key_values[self.layer_idx][0],
            past_value_state=past_key_values[self.layer_idx][1],
            batch_position=batch_position,
            cache_seq_len=seq_positions[0],
            cache_offset=seq_positions[1],
            scale=self.scale,
        )

        attn_outputs = self.o_proj(attn_output)
        return attn_outputs



class Gemma3AttentionOp(AttentionOp):
    def forward(
        self,
        query_state: torch.Tensor,
        key_state: torch.Tensor,
        value_state: torch.Tensor,
        attn_mask: torch.Tensor,
        past_key_state: torch.Tensor,
        past_value_state: torch.Tensor,
        seq_position: torch.Tensor,
        scale: torch.Tensor,
        block_tables: torch.Tensor,
        block_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute attention with static shapes and explicit cache management.

        Args:
            query_state: Query tensor [1, num_heads, 1, head_dim]
            key_state: Key tensor [1, num_heads, seq_len, head_dim]
            value_state: Value tensor [1, num_heads, seq_len, head_dim]
            attn_mask: Attention mask tensor ∈ {0, 1}
            past_key_state: Previous key cache states
            past_value_state: Previous value cache states
            seq_position: Current position in sequence
            scale: Scale applied to attn weights

        Returns:
            Tensor: attention_output: [batch, num_heads, seq_len, head_dim]
        """
        # reshape for removing repeat_kv (batch=1 , num_head, 1, q_len=1, head_dim)
        key_state = key_state.unsqueeze(2)  # 1, 32, 1, 128, 128
        value_state = value_state.unsqueeze(2)

        if self.phase == "decode":
            batch_size = key_state.shape[0]
        else:
            batch_size = 1

        query_state = query_state.view(
            batch_size,
            self.num_key_value_heads,
            self.num_heads // self.num_key_value_heads,
            -1,  # seq len
            self.head_dim,
        )

        if self.phase == "decode":
            attn_output = torch.ops.rbln_custom_ops.paged_causal_attn_decode(
                q=query_state,
                k=key_state,
                v=value_state,
                kcache=past_key_state.unsqueeze(2),
                vcache=past_value_state.unsqueeze(2),
                seq=seq_position,
                scale=scale,
                block_table=block_tables,
                block_size=block_size,
                is_bidrectional=True,
                mask=attn_mask,
            )
        else:
            attn_output = torch.ops.rbln_custom_ops.paged_causal_attn_prefill(
                q=query_state,
                k=key_state,
                v=value_state,
                kcache=past_key_state.unsqueeze(2),
                vcache=past_value_state.unsqueeze(2),
                seq=seq_position,
                scale=scale,
                block_table=block_tables,
                block_size=block_size,
                is_bidrectional=True,
                mask=attn_mask,
            )

        attn_output = attn_output.view(batch_size, self.num_heads, -1, self.head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, -1, self.num_heads * self.head_dim)

        return attn_output



class Gemma3FlashAttentionOp(FlashAttentionOp):
    def forward(
        self,
        query_state,
        key_state,
        value_state,
        attn_mask,
        past_key_state,
        past_value_state,
        seq_position,
        scale,
        block_tables,
        kvcache_block_size,
    ):
        # reshape for removing repeat_kv (batch=1 , num_head, 1, q_len=1, head_dim)
        key_state = key_state.unsqueeze(2)
        value_state = value_state.unsqueeze(2)

        if self.phase == "decode":
            batch_size = key_state.shape[0]
        else:
            batch_size = 1

        query_state = query_state.view(
            batch_size,
            self.num_key_value_heads,
            self.num_heads // self.num_key_value_heads,
            -1,  # seq len
            self.head_dim,
        )

        if self.phase == "decode":
            attn_output = torch.ops.rbln_custom_ops.paged_flash_causal_attn_decode(
                q=query_state,
                k=key_state,
                v=value_state,
                kcache=past_key_state.unsqueeze(2),
                vcache=past_value_state.unsqueeze(2),
                seq=seq_position,
                scale=scale,
                block_table=block_tables,
                block_size=kvcache_block_size,
                partition=self.kvcache_partition_size,
                is_bidrectional=True,
                mask=attn_mask,
            )
        else:
            attn_output = torch.ops.rbln_custom_ops.paged_flash_causal_attn_prefill(
                q=query_state,
                k=key_state,
                v=value_state,
                kcache=past_key_state.unsqueeze(2),
                vcache=past_value_state.unsqueeze(2),
                seq=seq_position,
                scale=scale,
                block_table=block_tables,
                block_size=kvcache_block_size,
                partition=self.kvcache_partition_size,
                is_bidrectional=True,
                mask=attn_mask,
            )

        # reshape for removing repeat_kv
        attn_output = attn_output.view(batch_size, self.num_heads, -1, self.head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, -1, self.num_heads * self.head_dim)

        return attn_output