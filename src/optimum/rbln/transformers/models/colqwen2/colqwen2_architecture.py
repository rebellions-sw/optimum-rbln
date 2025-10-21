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

from typing import TYPE_CHECKING, List, Optional, Union, Tuple

import torch
import torch.nn as nn
from .configuration_colqwen2 import (
    RBLNColQwen2ForRetrievalConfig,
)
from optimum.rbln.configuration_utils import RBLNCompileConfig
from optimum.rbln.modeling import RBLNModel
from optimum.rbln.transformers.models.decoderonly.decoderonly_architecture import (
    DecoderOnlyWrapper,
    DecoderOnlyModel,
    DecoderOnlyLayer,
    DecoderOnlyAttention,
    apply_rotary_pos_emb,
)
from optimum.rbln.transformers.models.decoderonly.modeling_decoderonly import (
    set_default_values,
    validate_attention_method,
)
from optimum.rbln.transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    RBLNQwen2_5_VLForConditionalGeneration,
)
from rebel.compile_context import CompileContext
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    Qwen2_5_VLForConditionalGeneration,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLRotaryEmbedding,
)

if TYPE_CHECKING:
    from transformers import (
        AutoFeatureExtractor,
        AutoProcessor,
        AutoTokenizer,
        PretrainedConfig,
    )


def slice_and_unsqueeze_cos_sin(cos, sin, position_ids):
    """Slice cos[cache_position], sin[cache_position] vector for the query."""
    cos = cos[position_ids[0]][None, None, None, :, :]
    sin = sin[position_ids[0]][None, None, None, :, :]

    return cos, sin


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, extra_dim, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states.expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, 1, slen, head_dim)


class ColQwen2LanguageModelWrapper(DecoderOnlyWrapper):
    def __init__(self, model: PreTrainedModel, rbln_config: "RBLNDecoderOnlyModelConfig", use_rotary_emb: bool = True):
        model.config = model.config.vlm_config
        super().__init__(model, rbln_config, use_rotary_emb)
        # self.config = model.config
        # self.rbln_config = rbln_config

        # if use_rotary_emb:
        #     rotary_embs = self.get_rotary_emb(max_seq_len=rbln_config.max_seq_len)
        #     if isinstance(rotary_embs, tuple):
        #         self.rotary_emb_global, self.rotary_emb_local = rotary_embs
        #     else:
        #         self.rotary_emb = rotary_embs
        # else:
        #     self.rotary_emb = None

        # if rbln_config.kvcache_partition_len and rbln_config.kvcache_partition_len > rbln_config.max_seq_len:
        #     raise ValueError(
        #         f"kvcache_partition_len({rbln_config.kvcache_partition_len}) should be lower"
        #         f" or equal to max_seq_len({rbln_config.max_seq_len})!"
        #     )

        # self.language_model = self.convert_to_rbln_class(model, rbln_config.max_seq_len)
        # self.num_hidden_layers = getattr(self.config, "num_hidden_layers", None) or getattr(self.config, "n_layer")
        # self._phase = "prefill"

        # embedding_proj_layer from original model
        self.embedding_proj_layer = model.embedding_proj_layer

    def get_decoder_layers(self, model: PreTrainedModel):
        return model.language_model.layers

    def convert_to_rbln_class(self, model: PreTrainedModel, max_seq_len: int):
        new_layers = []
        for layer_idx, layer in enumerate(model.vlm.language_model.layers):
            is_sliding = layer_idx in self.rbln_config.sliding_window_layers
            new_self_attn = self.get_rbln_attn_class()(
                self.get_attn_layer(layer),
                self.rbln_config,
                is_sliding=is_sliding,
            )
            new_layer = self.get_rbln_layer_class()(layer, new_self_attn)
            new_layers.append(new_layer)

        new_model = self.get_rbln_model_class()(
            model.vlm.language_model,
            new_layers,
            self.rbln_config,
            use_learned_pos_emb=self.__class__._use_learned_pos_emb,
        )
        return new_model

    def get_rbln_model_class(self):
        return RBLNColQwen2LanguageModel

    def prepare_forward_args(self, *args):
        args = list(args)
        input_ids = None if self.rbln_config.use_inputs_embeds else args.pop(0)
        inputs_embeds = args.pop(0) if self.rbln_config.use_inputs_embeds else None
        cache_position = args.pop(0)
        global_block_tables = args.pop(0)
        local_block_tables = None
        position_embeds = args.pop(0)
        position_ids = None
        attention_mask = args.pop(0) if self.rbln_config.use_attention_mask else None
        past_key_values = args

        if len(past_key_values) != 2 * self.num_hidden_layers:
            raise ValueError(
                f"Different past_key_values to model's config. {len(past_key_values)} != {2 * self.num_hidden_layers}"
            )

        _past_key_values = []
        for i in range(self.config.num_hidden_layers):
            key_states = past_key_values[i * 2]
            value_states = past_key_values[i * 2 + 1]
            past_key_value = [key_states, value_states]
            _past_key_values.append(past_key_value)
        past_key_values = _past_key_values

        return (
            input_ids,
            inputs_embeds,
            cache_position,
            global_block_tables,
            local_block_tables,
            attention_mask,
            position_ids,
            past_key_values,
            position_embeds,
        )

    # def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor, position_ids: torch.Tensor):
    #     attention_mask = (1.0 - attention_mask) * torch.finfo(torch.float32).min
    #     attention_mask = attention_mask[:, None, None, None, :]
    def forward(self, *args):
        (
            input_ids,
            inputs_embeds,
            cache_position,
            global_block_tables,
            local_block_tables,
            attention_mask,
            position_ids,
            past_key_values,
            rotary_emb,
        ) = self.prepare_forward_args(*args)

        last_hidden_states = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            position_ids=position_ids,
            past_key_values=past_key_values,
            rotary_emb=rotary_emb,
            global_block_tables=global_block_tables,
            local_block_tables=local_block_tables,
        )
        # last_hidden_states = self.model(
        #     inputs_embeds=inputs_embeds,
        #     attention_mask=attention_mask,
        #     position_ids=position_ids,
        #     rotary_emb=self.rotary_emb,
        # )
        proj = self.embedding_proj_layer(last_hidden_states[0])

        return proj
        # return last_hidden_states

# class ColQwen2AttentionOp(nn.Module):
#     def __init__(self, self_attn):
#         super().__init__()
#         self._original_mod = self_attn
#         self.head_dim = self_attn.head_dim
#         self.num_key_value_groups = self_attn.num_key_value_groups
#         self.scaling = self.head_dim**-0.5

#     def forward(
#         self,
#         query_states: torch.Tensor,
#         key_states: torch.Tensor,
#         value_states: torch.Tensor,
#         attention_mask: torch.Tensor,
#     ):
#         batch_size, _, _, query_length, _ = query_states.size()

#         key_states = repeat_kv(key_states, self.num_key_value_groups)
#         value_states = repeat_kv(value_states, self.num_key_value_groups)

#         attn_weights = torch.matmul(query_states, key_states.transpose(3, 4)) * self.scaling
#         attn_weights = attn_weights + attention_mask
#         attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
#         attn_output = torch.matmul(attn_weights, value_states)
#         attn_output = attn_output.transpose(1, 3)
#         attn_output = attn_output.reshape(batch_size, query_length, -1)

#         return attn_output

# class ColQwen2Attention(nn.Module):
#     def __init__(self, self_attn):
#         super().__init__()
#         self._original_mod = self_attn
#         self.num_heads = getattr(self._original_mod, "num_heads", None) or getattr(
#             self._original_mod.config, "num_attention_heads"
#         )
#         self.head_dim = self._original_mod.head_dim
#         self.scaling = self.head_dim**-0.5

#         if hasattr(self._original_mod, "num_key_value_heads"):
#             self.num_key_value_heads = self._original_mod.num_key_value_heads
#         elif hasattr(self._original_mod, "config") and hasattr(self._original_mod.config, "num_key_value_heads"):
#             self.num_key_value_heads = self._original_mod.config.num_key_value_heads
#         else:
#             self.num_key_value_heads = self.num_heads

#         self.__post_init__()

#     def __post_init__(self):
#         self.q_proj = self._original_mod.q_proj
#         self.k_proj = self._original_mod.k_proj
#         self.v_proj = self._original_mod.v_proj
#         self.o_proj = self._original_mod.o_proj

#     def projection(self, hidden_states) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         query_states = self.q_proj(hidden_states)
#         key_states = self.k_proj(hidden_states)
#         value_states = self.v_proj(hidden_states)

#         return query_states, key_states, value_states

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         attention_mask: torch.Tensor,
#         cos: Optional[torch.Tensor] = None,
#         sin: Optional[torch.Tensor] = None,
#     ):
#         batch_size, query_length, _ = hidden_states.size()

#         query_states, key_states, value_states = self.projection(hidden_states=hidden_states)

#         query_states = query_states.view(batch_size, query_length, 1, self.num_heads, self.head_dim).transpose(1, 3)
#         key_states = key_states.view(batch_size, query_length, 1, self.num_key_value_heads, self.head_dim).transpose(
#             1, 3
#         )
#         value_states = value_states.view(
#             batch_size, query_length, 1, self.num_key_value_heads, self.head_dim
#         ).transpose(1, 3)

#         if cos is not None and sin is not None:
#             query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

#         # key_states = repeat_kv(key_states, self.num_heads // self.num_key_value_heads)
#         # value_states = repeat_kv(value_states, self.num_heads // self.num_key_value_heads)

#         # FIXME(seinpark) : broadcast?
#         attn_weights = torch.matmul(query_states, key_states.transpose(3, 4)) * self.scaling
#         attn_weights = attn_weights + attention_mask
#         attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
#         attn_output = torch.matmul(attn_weights, value_states)
#         attn_output = attn_output.transpose(1, 3)

#         attn_output = attn_output.reshape(batch_size, query_length, -1)
#         attn_output = self.o_proj(attn_output)

#         return attn_output

# class ColQwen2Layer(DecoderOnlyLayer):
#     def __init__(self, layer, self_attn):
#         super().__init__(layer, self_attn)

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         attention_mask: torch.Tensor,
#         seq_positions: torch.LongTensor,

#         cos: Optional[torch.Tensor] = None,
#         sin: Optional[torch.Tensor] = None,
#     ):
#         residual = hidden_states
#         hidden_states = self.get_pre_attention_layernorm()(hidden_states)
#         hidden_states = self.self_attn(
#             hidden_states=hidden_states,
#             attention_mask=attention_mask,
#             # seq_positions=seq_positions,
#             cos=cos,
#             sin=sin,
#         )
#         hidden_states = residual + hidden_states

#         # Fully Connected
#         residual = hidden_states
#         hidden_states = self.get_post_attention_layernorm()(hidden_states)
#         hidden_states = self.forward_mlp(hidden_states)
#         hidden_states = residual + hidden_states

#         return hidden_states

# TODO(seinpark) : 그냥 DecoderOnly 를 사용하는걸로!
class RBLNColQwen2LanguageModel(DecoderOnlyModel):
    def __init__(
        self,
        model,
        layers: List["DecoderOnlyLayer"],
        rbln_config: "RBLNDecoderOnlyModelConfig",
        use_learned_pos_emb=None,
    ):
        super().__init__(model, layers, rbln_config, use_learned_pos_emb)
        
        # self.output_hidden_states = rbln_config.output_hidden_states
        self.output_hidden_states = False
    
    def forward(
        self,
        input_ids: torch.Tensor = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: torch.Tensor = None,
        cache_position: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        query_position: torch.Tensor = None,
        past_key_values: Tuple[Tuple[torch.Tensor]] = None,
        rotary_emb: Optional[Union[nn.Module, torch.Tensor]] = None,
        global_block_tables: Optional[torch.Tensor] = None,
        local_block_tables: Optional[torch.Tensor] = None,
        lora_int_id: Optional[torch.Tensor] = None,
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
        position_ids = position_ids if position_ids is not None else cache_position
        if rotary_emb is not None:
            if isinstance(rotary_emb, torch.Tensor):
                cos = rotary_emb[0]
                sin = rotary_emb[1]
            else:
                cos, sin = rotary_emb(hidden_states, self.max_seq_len)  # dtype carrier, max_seq_len
                cos, sin = slice_and_unsqueeze_cos_sin(cos, sin, position_ids)

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

        all_hidden_states = () if self.output_hidden_states else None
        for layer_idx, layer in enumerate(self.layers):
            if self.output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            is_sliding = True if layer_idx in self.sliding_window_layers else False
            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                seq_positions=sliding_cache_pos if is_sliding else seq_positions,
                past_key_values=past_key_values,
                cos=cos,
                sin=sin,
                block_tables=local_block_tables if is_sliding else global_block_tables,
                lora_int_id=lora_int_id,
            )

        hidden_states = self.get_last_layernorm()(hidden_states)
        if self.output_hidden_states:
                all_hidden_states += (hidden_states,)

        return hidden_states, all_hidden_states
        # return hidden_states