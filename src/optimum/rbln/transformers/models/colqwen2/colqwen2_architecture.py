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

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from optimum.rbln.transformers.models.decoderonly.decoderonly_architecture import (
    DecoderOnlyLayer,
    DecoderOnlyModel,
    DecoderOnlyWrapper,
)

from .configuration_colqwen2 import (
    RBLNColQwen2ForRetrievalConfig,
)


def slice_and_unsqueeze_cos_sin(cos, sin, position_ids):
    """Slice cos[cache_position], sin[cache_position] vector for the query."""
    cos = cos[position_ids[0]][None, None, None, :, :]
    sin = sin[position_ids[0]][None, None, None, :, :]

    return cos, sin


class ColQwen2LanguageModelWrapper(DecoderOnlyWrapper):
    def __init__(
        self, model: PreTrainedModel, rbln_config: "RBLNColQwen2ForRetrievalConfig", use_rotary_emb: bool = True
    ):
        model.config = (
            model.config.vlm_config.text_config if hasattr(model.config, "vlm_config") else model.config.text_config
        )
        super().__init__(model, rbln_config, use_rotary_emb)

    def get_decoder_layers(self, model: PreTrainedModel):
        return model.language_model.layers

    def convert_to_rbln_class(self, model: PreTrainedModel, max_seq_len: int):
        new_layers = []
        for layer_idx, layer in enumerate(self.get_decoder_layers(model)):
            is_sliding = layer_idx in self.rbln_config.sliding_window_layers
            new_self_attn = self.get_rbln_attn_class()(
                self.get_attn_layer(layer),
                self.rbln_config,
                is_sliding=is_sliding,
            )
            new_layer = self.get_rbln_layer_class()(layer, new_self_attn)
            new_layers.append(new_layer)

        new_model = self.get_rbln_model_class()(
            model.language_model,
            new_layers,
            self.rbln_config,
            use_learned_pos_emb=self.__class__._use_learned_pos_emb,
        )

        # text_projection layer from model
        self.embedding_proj_layer = (
            model.embedding_proj_layer if hasattr(model, "embedding_proj_layer") else model.custom_text_proj
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

        proj = self.embedding_proj_layer(last_hidden_states[0])
        all_hidden_states = last_hidden_states[1] if self.rbln_config.output_hidden_states else None

        if self.rbln_config.output_hidden_states:
            return proj, all_hidden_states
        else:
            return proj


class RBLNColQwen2LanguageModel(DecoderOnlyModel):
    def __init__(
        self,
        model,
        layers: List["DecoderOnlyLayer"],
        rbln_config: "RBLNColQwen2ForRetrievalConfig",
        use_learned_pos_emb=None,
    ):
        super().__init__(model, layers, rbln_config, use_learned_pos_emb)

        self.output_hidden_states = rbln_config.output_hidden_states

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
