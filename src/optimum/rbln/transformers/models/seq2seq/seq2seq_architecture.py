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

from ....ops import register_rbln_custom_attention, register_rbln_custom_cache_update


logger = logging.get_logger(__name__)


class Seq2SeqWrapper:
    """A wrapper class for Seq2Seq models to support RBLN-specific optimizations.

    This wrapper divides the Seq2Seq model into separate encoder and decoder wrappers,
    enabling specific optimizations such as custom cache handling and attention mechanisms.

    Args:
        model (nn.Module): The Seq2Seq model to wrap.
        enc_max_seq_len (int): Maximum sequence length for the encoder's position embeddings and cache sizes.
        **kwargs: Additional arguments to pass to the decoder wrapper.
    """

    def __init__(self, model: nn.Module, enc_max_seq_len: int, **kwargs):
        self.encoder = Seq2SeqEncoderWrapper(model, enc_max_seq_len)
        self.decoder = Seq2SeqDecoderWrapper(model, **kwargs)


class Seq2SeqEncoderWrapper(nn.Module):
    """A wrapper for the encoder component of a Seq2Seq model, designed for RBLN optimization.

    This wrapper modifies the standard encoder-decoder architecture of Seq2Seq models to optimize
    memory usage and attention mechanisms, particularly in cross-attention layers. It supports custom
    cache handling to improve performance during decoding.

    Args:
        model (nn.Module): The Seq2Seq model containing the encoder.
        enc_max_seq_len (int): Maximum sequence length for encoder embeddings and cache sizes.
    """

    def __init__(self, model: nn.Module, enc_max_seq_len: int):
        super().__init__()
        register_rbln_custom_cache_update()
        self.config = model.config
        self.encoder = model.get_encoder()
        self.encoder_max_length = enc_max_seq_len
        self.__post_init__(model)

    def __post_init__(self, model: nn.Module):
        """
        Post-initialization to extract and configure encoder-related attributes.

        It is inspired by the BART architecture, but it is designed to be flexible and can be overridden
        by subclasses to modify or add custom attributes as necessary.
        """
        self.n_layer = getattr(self.config, "decoder_layers", None)
        self.cross_k_projects, self.cross_v_projects = self._extract_cross_kv_projects(model.get_decoder().layers)
        self.num_heads = self.config.decoder_attention_heads
        self.d_kv = self.config.d_model // self.num_heads

    def _extract_cross_kv_projects(self, decoder_layers: nn.Module):
        """
        Extract cross-attention key and value projection layers from the decoder.
        """
        return (
            nn.ModuleList(decoder_layers[i].encoder_attn.k_proj for i in range(self.n_layer)),
            nn.ModuleList(decoder_layers[i].encoder_attn.v_proj for i in range(self.n_layer)),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cross_key_values: torch.Tensor,
        batch_position: torch.Tensor,
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

        cross_kv = torch.stack(cross_kv, dim=0)

        # 3. update the cross_attention's past_key_value direct to the device-dram for optimization.
        batch_axis = torch.tensor(1, dtype=torch.int16)
        cross_key_values = torch.ops.rbln_custom_ops.rbln_cache_update(
            cross_key_values, cross_kv, batch_position, batch_axis
        )

        return cross_key_values


class Seq2SeqDecoderWrapper(nn.Module):
    """
    A wrapper for the decoder component of a Seq2Seq model, designed for RBLN optimization.

    This wrapper handles tasks such as:
    1. Converting decoder components to support RBLN-specific conditional generation.
    2. Customizing attention mechanisms, including self-attention and cross-attention.
    3. Managing the decoder's key-value caches for both self and cross-attention.

    Args:
        model (nn.Module): The Seq2Seq model containing the decoder.
        **kwargs: Additional arguments for decoder configuration.
    """

    def __init__(self, model: nn.Module, **kwargs):
        super().__init__()
        self.config = model.config
        self.__post_init__(model, **kwargs)

    def __post_init__(self, model: nn.Module, **kwargs):
        """
        Post-initialization to extract and configure encoder-related attributes.

        It is inspired by the BART architecture, but it is designed to be flexible and can be overridden
        by subclasses to modify or add custom attributes as necessary.
        """
        register_rbln_custom_attention()
        self.num_layers = self.config.decoder_layers
        self.conditional_generation = self.convert_to_rbln_conditional_generation(model)

    def convert_to_rbln_conditional_generation(self, model: nn.Module):
        new_layers = []
        for layer in model.get_decoder().layers:
            self_attn = Seq2SeqSelfAttention(layer.self_attn)
            new_layers.append(Seq2SeqDecoderLayer(layer, self_attn))

        decoder_model = Seq2SeqDecoder(model.get_decoder(), new_layers)
        new_model = Seq2SeqForConditionalGeneration(model, decoder_model)

        return new_model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        cache_position: torch.Tensor,
        cross_kv_cache: torch.Tensor,
        *self_kv_cache: torch.Tensor,
    ) -> Tuple[torch.FloatTensor, Tuple[torch.FloatTensor]]:
        self_past_key_values = ()
        cross_past_key_values = ()
        for i in range(0, self.num_layers * 2, 2):
            self_past_key_values = self_past_key_values + ((self_kv_cache[i], self_kv_cache[i + 1]),)
            cross_past_key_values = cross_past_key_values + ((cross_kv_cache[i], cross_kv_cache[i + 1]),)

        # decode
        lm_logits, self_present_key_values = self.conditional_generation(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            self_past_key_values=self_past_key_values,
            cross_past_key_values=cross_past_key_values,
            cache_position=cache_position,
        )

        outputs = (lm_logits,) + self_present_key_values

        return outputs


class Seq2SeqForConditionalGeneration(nn.Module):
    """
    A wrapper for Seq2Seq models supporting RBLN-specific optimizations for conditional generation.

    This class adapts a Seq2Seq model for tasks like machine translation, summarization, or text generation
    by:
    1. Wrapping and customizing the decoder component to support key RBLN features.
    2. Managing rescaling and output processing, if enabled.
    3. Aligning model behavior with RBLN's static and efficient execution requirements.

    Attributes:
        has_rescaling (bool): Indicates if output rescaling is applied.
        config (PretrainedConfig): Configuration from the original Seq2Seq model.
        lm_head (nn.Linear): The language modeling head for output logits.
        decoder (nn.Module): The wrapped decoder model.
    """

    has_rescaling = False

    def __init__(self, model, decoder_model):
        super().__init__()
        self.config = model.config
        self.lm_head = model.lm_head
        self.decoder = decoder_model
        self.__post_init__()

    def __post_init__(self):
        """
        Abstract method intended to be overridden by subclasses to modify or override
        the attributes of the original model after initialization.
        """

    def forward(
        self,
        input_ids,
        attention_mask,
        encoder_attention_mask,
        self_past_key_values,
        cross_past_key_values,
        cache_position,
    ):
        hidden_states, self_present_key_values = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            self_past_key_values=self_past_key_values,
            cross_past_key_values=cross_past_key_values,
            cache_position=cache_position,
        )

        if self.has_rescaling and self.config.tie_word_embeddings:
            hidden_states = hidden_states * self.scaling

        lm_logits = self.lm_head(hidden_states)

        return lm_logits, self_present_key_values


class Seq2SeqDecoder(torch.nn.Module):
    """A modified Seq2SeqDecoder implementation optimized for RBLN compilation.

    Args:
        model: Original Huggingface model to adapt
        layers (List[Seq2SeqDecoderLayer]): Modified transformer layers optimized for RBLN
    """

    has_pos_emb = True

    def __init__(self, model, layers, **kwargs):
        super().__init__()
        self._original_mod = model
        self.layers = nn.ModuleList(layers)
        self.embed_tokens = model.embed_tokens
        self.final_layer_norm = getattr(model, "final_layer_norm", None)
        self.__post_init__(**kwargs)

    def __post_init__(self, **kwargs):
        """
        Abstract method intended to be overridden by subclasses to modify or override
        the attributes of the original model after initialization.
        """
        pass

    def get_embedding(self):
        return self.embed_tokens

    def prepare_attn_mask(self, *args, **kwargs):
        raise NotImplementedError(
            "The 'prepare_attn_mask' method is not implemented. Please define this method in a subclass."
        )

    def apply_position_embedding(self, *args, **kwargs):
        raise NotImplementedError(
            "The 'apply_position_embedding' method is not implemented. Please define this method in a subclass."
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        self_past_key_values: torch.Tensor,
        cross_past_key_values: torch.Tensor,
        cache_position: torch.Tensor,
    ):
        # embedding
        hidden_states = self.get_embedding()(input_ids)
        attention_mask, encoder_attention_mask = self.prepare_attn_mask(
            attention_mask, encoder_attention_mask, cache_position=cache_position
        )

        if self.has_pos_emb:
            hidden_states = self.apply_position_embedding(hidden_states, cache_position)

        # iterate decoder_layer
        self_present_key_values = ()
        for decoder_layer, self_past_key_value, cross_past_key_value in zip(
            self.layers, self_past_key_values, cross_past_key_values
        ):
            hidden_states, self_present_key_value = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                self_past_key_value=self_past_key_value,
                cross_past_key_value=cross_past_key_value,
                cache_position=cache_position,
            )
            self_present_key_values += self_present_key_value

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states, self_present_key_values


class Seq2SeqDecoderLayer(torch.nn.Module):
    """A modified decoder-only model implementation optimized for RBLN compilation.

    Args:
        model: Original Huggingface model to adapt
        layers (List[DecoderOnlyLayer]): Modified transformer layers optimized for RBLN
        self_attn (Seq2SeqSelfAttention): Modified self-attention layer optimized for RBLN
    """

    def __init__(self, decoder_layer, self_attn):
        super().__init__()
        self._original_mod = decoder_layer
        self.self_attn = self_attn
        self.__post_init__()

    def __post_init__(self, **kwargs):
        """
        Abstract method intended to be overridden by subclasses to modify or override
        the attributes of the original model after initialization.
        """
        pass

    def pre_self_attn_layer_norm(self, hidden_states):
        raise NotImplementedError(
            "The 'pre_self_attn_layer_norm' method is not implemented. Please define this method in a subclass."
        )

    def post_self_attn_layer_norm(self, hidden_states):
        raise NotImplementedError(
            "The 'post_self_attn_layer_norm' method is not implemented. Please define this method in a subclass."
        )

    def pre_cross_attn_layer_norm(self, hidden_states):
        raise NotImplementedError(
            "The 'pre_cross_attn_layer_norm' method is not implemented. Please define this method in a subclass."
        )

    def post_cross_attn_layer_norm(self, hidden_states):
        raise NotImplementedError(
            "The 'post_cross_attn_layer_norm' method is not implemented. Please define this method in a subclass."
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        self_past_key_value: Tuple[torch.Tensor],
        cross_past_key_value: Tuple[torch.Tensor],
        cache_position: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
        dummy_encoder_hidden_states = torch.zeros(1, encoder_attention_mask.shape[-1])

        # Self Attention Block
        residual = hidden_states
        hidden_states = self.pre_self_attn_layer_norm(hidden_states)
        hidden_states, self_attn_past_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_past_key_value,
            attention_mask=attention_mask,
            cache_position=cache_position,
        )
        hidden_states = residual + hidden_states
        hidden_states = self.post_self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        residual = hidden_states
        hidden_states = self.pre_cross_attn_layer_norm(hidden_states)
        cross_attn_output = self.encoder_attn(
            hidden_states=hidden_states,
            past_key_value=cross_past_key_value,
            attention_mask=encoder_attention_mask,
            key_value_states=dummy_encoder_hidden_states,
        )
        hidden_states = residual + cross_attn_output[0]
        hidden_states = self.post_cross_attn_layer_norm(hidden_states)

        # Feed-Forward Block
        hidden_states = self.ff_layer(hidden_states)

        return hidden_states, self_attn_past_key_value


class Seq2SeqSelfAttention(nn.Module):
    def __init__(self, attn):
        super().__init__()
        self._original_mod = attn
        self.__post_init__()

    def __post_init__(self, **kwargs):
        """
        Abstract method intended to be overridden by subclasses to modify or override
        the attributes of the original model after initialization.
        """
        pass

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int) -> torch.Tensor:
        return tensor.view(bsz, 1, seq_len, 1, self.num_heads, self.head_dim).transpose(2, 4)

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
        return query_states, key_states, value_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Tuple[torch.Tensor],
        attention_mask: torch.Tensor,
        cache_position: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
        bsz, tgt_len, _ = hidden_states.size()

        query_states, key_states, value_states = self.projection(hidden_states=hidden_states)
        query_states = self._shape(query_states, tgt_len, bsz)
        key_states = self._shape(key_states, -1, bsz)
        value_states = self._shape(value_states, -1, bsz)

        all_key_states = []
        all_value_states = []
        all_attn_output = []
        for b_idx in range(bsz):
            query_state = query_states[b_idx]
            key_state = key_states[b_idx]
            value_state = value_states[b_idx]
            attn_mask = attention_mask[b_idx].unsqueeze(0).unsqueeze(2)
            past_key_state = past_key_value[0].view(bsz, self.num_heads, 1, -1, self.head_dim)
            past_value_state = past_key_value[1].view(bsz, self.num_heads, 1, -1, self.head_dim)

            attn_output, key_state, value_state = self.attn_decode(
                query_state,
                key_state,
                value_state,
                attn_mask,
                past_key_state,
                past_value_state,
                cache_position[b_idx][0],
                torch.tensor(1.0, dtype=torch.float32),  # scale
            )

            attn_output = attn_output.view(1, self.num_heads, -1, self.head_dim).transpose(1, 2)
            attn_output = attn_output.reshape(1, -1, self.num_heads * self.head_dim)

            all_key_states.append(key_state.squeeze(2))
            all_value_states.append(value_state.squeeze(2))
            all_attn_output.append(attn_output)

        key_states = torch.cat(all_key_states, dim=0)
        value_states = torch.cat(all_value_states, dim=0)
        attn_output = torch.cat(all_attn_output, dim=0)

        attn_output = self.out_proj(attn_output)
        present_key_value = (key_states, value_states)

        return attn_output, present_key_value
