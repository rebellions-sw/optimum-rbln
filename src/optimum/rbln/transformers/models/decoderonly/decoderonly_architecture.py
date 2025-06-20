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

import math
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel

from ....utils import logging
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS
from .configuration_decoderonly import CacheImplType


logger = logging.get_logger(__name__)

DEFAULT_FLASH_ATTN_PARTITION_LENGTH = 16_384
DEFAULT_MAX_EAGER_ATTN_SEQUENCE_LENGTH = 32_768
MIN_FLASH_ATTN_MAX_SEQ_LEN = 8_192
MIN_FLASH_ATTN_PARTITION_LENGTH = 4_096
MAX_FLASH_ATTN_PARTITION_LENGTH = 32_768
MAX_SLIDING_WINDOW_SIZE = 32_768


def set_default_values(
    attn_impl: Optional[str] = None,
    kvcache_partition_len: Optional[int] = None,
    kvcache_block_size: Optional[int] = None,
    max_seq_len: Optional[int] = None,
) -> Tuple[str, int, int]:
    if attn_impl is None:
        attn_impl = "eager"

    if kvcache_partition_len is not None:
        if attn_impl == "eager":
            attn_impl = "flash_attn"
            logger.warning(
                "A non-null `kvcache_partition_len` was provided, but `attn_impl` was not explicitly set or "
                "set to 'eager'. Since KV cache partitioning is only supported with flash attention, "
                "`attn_impl` has been automatically switched to 'flash_attn'."
            )

    if kvcache_partition_len is None and attn_impl == "flash_attn":
        kvcache_partition_len = DEFAULT_FLASH_ATTN_PARTITION_LENGTH

    if kvcache_block_size is None:
        if attn_impl == "eager":
            kvcache_block_size = max_seq_len
        else:
            kvcache_block_size = kvcache_partition_len

    return attn_impl, kvcache_partition_len, kvcache_block_size


def validate_attention_method(attn_impl: str, kvcache_partition_len: int, kvcache_block_size: int, max_seq_len: int):
    if attn_impl not in ["eager", "flash_attn"]:
        raise ValueError(f"Unknown `attn_impl` : {attn_impl}. (Available : 'eager', 'flash_attn`)")

    ## Checking Constraints...
    # Constraint of eager attention:
    # - `max_seq_len` <= 32k

    # Constraints of flash attention:
    # 1. `max_seq_len` should be multiple of `partition_len`.
    # 2. 4k <= `partition_len` <= 32k.
    # 3. `max_seq_len` should be larger then 8k.
    if attn_impl == "eager" and max_seq_len > DEFAULT_MAX_EAGER_ATTN_SEQUENCE_LENGTH:
        raise ValueError(
            f"`max_seq_len` is set to {max_seq_len}, "
            f"which exceeds the limit of {DEFAULT_MAX_EAGER_ATTN_SEQUENCE_LENGTH} for 'eager' attention. "
            f"Please reduce the `max_seq_len` to {DEFAULT_MAX_EAGER_ATTN_SEQUENCE_LENGTH} or lower,"
            " or consider switching `attn_impl` to 'flash_attn' for larger sequence lengths."
        )

    if attn_impl == "flash_attn":
        if max_seq_len // kvcache_partition_len < 2 or max_seq_len % kvcache_partition_len != 0:
            raise ValueError(
                f"`max_seq_len` ({max_seq_len}) must be a multiple of `kvcache_partition_len` ({kvcache_partition_len}) "
                f"when using 'flash_attn'. Please adjust either value to meet this requirement."
            )
        elif not (MIN_FLASH_ATTN_PARTITION_LENGTH <= kvcache_partition_len <= MAX_FLASH_ATTN_PARTITION_LENGTH):
            raise ValueError(
                f"`kvcache_partition_len` ({kvcache_partition_len}) is out of the supported range for 'flash_attn' "
                f"({MIN_FLASH_ATTN_PARTITION_LENGTH} <= `kvcache_partition_len` <= {MAX_FLASH_ATTN_PARTITION_LENGTH}). "
                f"Please provide a valid value within this range."
            )
        elif max_seq_len < MIN_FLASH_ATTN_MAX_SEQ_LEN:
            raise ValueError(
                f"`max_seq_len` ({max_seq_len}) is too small for 'flash_attn'. The minimum "
                f"supported value is {MIN_FLASH_ATTN_MAX_SEQ_LEN}. Please increase `max_seq_len` to meet "
                "this requirement, or consider switching `attn_impl` to 'eager' for shorter lengths."
            )

    if kvcache_block_size is not None:
        if attn_impl == "flash_attn" and kvcache_partition_len != kvcache_block_size:
            raise ValueError(
                f" When using 'flash attention', the `kvcache_block_size` ({kvcache_block_size})  "
                f"must always be set equal to the `kvcache_partition_len` {kvcache_partition_len}."
            )
        elif attn_impl == "eager" and kvcache_block_size != max_seq_len:
            raise ValueError(
                f" When using 'eager attention', the `kvcache_block_size` ({kvcache_block_size})  "
                f"must always be set equal to the `max_seq_len` {max_seq_len}."
            )


def validate_sliding_window_size(sliding_window: int, prefill_chunk_size: int):
    if sliding_window > MAX_SLIDING_WINDOW_SIZE - prefill_chunk_size:
        raise ValueError(
            f"Sliding window size ({sliding_window}) must be less than 32768 - prefill_chunk_size ({32768 - prefill_chunk_size})"
        )


class DecoderOnlyWrapper(nn.Module):
    """A wrapper class for decoder-only language models that handles RBLN-specific optimizations and requirements.

    This wrapper is designed to:
    1. Convert Huggingface decoder models for RBLN compilation with static shapes
    2. Handle input/model mapping and additional information supply (e.g., positional embeddings)
    3. Manage different attention implementations (standard/flash attention)
    4. Support both prefill and decode phases

    Notes:
    - Wrapper must only receive positional arguments in forward() due to torch.jit.trace dependency
    - Wrapper should not contain neural network graph operations (including memory view handling)

    Args:
        causal_lm (PreTrainedModel): The Huggingface causal language model to wrap
        max_seq_len (int): Maximum sequence length for position embeddings and cache sizes
        use_rotary_emb (bool): Whether to use rotary position embeddings
        attn_impl (str): The attention implementation to use.
            - "eager": Uses the standard attention.
            - "flash_attn": Uses flash attention. When set,
              the key/value cache is partitioned into chunks of length
              `kvcache_partition_len`.
        kvcache_partition_len (Optional[int]): Length of KV cache partitions for flash attention.
            This is only relevant if `attn_impl` is set to "flash_attn`
    """

    def __init__(
        self,
        causal_lm: PreTrainedModel,
        max_seq_len: int,
        use_rotary_emb: bool,
        attn_impl: str,
        cache_impl: CacheImplType,
        use_inputs_embeds: bool,
        use_attention_mask: bool,
        use_position_ids: bool,
        use_learned_pos_emb: Optional[bool] = None,
        kvcache_partition_len: Optional[int] = None,
        kvcache_block_size: Optional[int] = None,
        sliding_window: Optional[int] = None,
        sliding_window_layers: Optional[List[int]] = None,
    ):
        super().__init__()
        self.config = causal_lm.config

        if use_rotary_emb:
            rotary_embs = self.get_rotary_emb(max_seq_len=max_seq_len)
            if isinstance(rotary_embs, tuple):
                self.rotary_emb_global, self.rotary_emb_local = rotary_embs
            else:
                self.rotary_emb = rotary_embs
        else:
            self.rotary_emb = None

        self.attn_impl = attn_impl
        self.kvcache_block_size = kvcache_block_size
        self.use_attention_mask = use_attention_mask
        self.use_position_ids = use_position_ids
        self.use_inputs_embeds = use_inputs_embeds
        self.use_learned_pos_emb = use_learned_pos_emb
        self.sliding_window_layers = sliding_window_layers
        self.cache_impl = cache_impl
        self.sliding_window = sliding_window

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

        self.causal_lm = self.convert_to_rbln_causal_lm(causal_lm, max_seq_len)
        self.num_hidden_layers = getattr(self.config, "num_hidden_layers", None) or getattr(self.config, "n_layer")
        self._phase = "prefill"

    def get_rotary_emb(self, max_seq_len):
        return RotaryEmbedding(config=self.config, max_seq_len_cached=max_seq_len)

    def convert_to_rbln_causal_lm(self, causal_lm: PreTrainedModel, max_seq_len: int):
        new_layers = []
        for layer_idx, layer in enumerate(causal_lm.model.layers):
            if layer_idx in self.sliding_window_layers:
                # Flash attention is not yet supported for sliding window attention.
                new_self_attn = DecoderOnlyAttention(
                    layer.self_attn,
                    self.use_attention_mask,
                    self.use_position_ids,
                    kvcache_block_size=self.sliding_window,
                    is_sliding=True,
                )
            else:
                if self.attn_impl == "eager":
                    new_self_attn = DecoderOnlyAttention(
                        layer.self_attn,
                        self.use_attention_mask,
                        self.use_position_ids,
                        kvcache_block_size=self.kvcache_block_size,
                        is_sliding=False,
                    )
                elif self.attn_impl == "flash_attn":
                    new_self_attn = DecoderOnlyFlashAttention(
                        layer.self_attn,
                        kvcache_partition_len=self.kvcache_partition_len,
                        kvcache_block_size=self.kvcache_block_size,
                        use_attention_mask=self.use_attention_mask,
                        use_position_ids=self.use_position_ids,
                    )
                else:
                    raise NotImplementedError(f"Unknwon attn : {self.attn_impl}")

            new_layer = DecoderOnlyLayer(layer, new_self_attn)
            new_layers.append(new_layer)

        new_model = DecoderOnlyModel(
            causal_lm.model,
            new_layers,
            partition_len=self.kvcache_partition_len,
            max_seq_len=max_seq_len,
            kvcache_block_size=self.kvcache_block_size,
            use_learned_pos_emb=self.use_learned_pos_emb,
            sliding_window_layers=self.sliding_window_layers,
        )
        new_causal_lm = DecoderOnlyForCausalLM(causal_lm, new_model)
        return new_causal_lm

    @property
    def phase(self) -> str:
        return self._phase

    @phase.setter
    def phase(self, phase: str):
        self._phase = phase
        self.causal_lm.phase = phase

    def prepare_forward_args(self, *args):
        args = list(args)
        input_ids = None if self.use_inputs_embeds else args.pop(0)
        inputs_embeds = args.pop(0) if self.use_inputs_embeds else None
        cache_position = args.pop(0)
        global_block_tables = args.pop(0) if self.cache_impl in ["hybrid", "static"] else None
        local_block_tables = args.pop(0) if self.cache_impl in ["hybrid", "sliding_window"] else None
        query_position = args.pop(0) if "prefill" in self.phase else None
        attention_mask = args.pop(0) if self.use_attention_mask else None
        position_ids = args.pop(0) if self.use_position_ids else None
        past_key_values = args

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
            query_position,
            attention_mask,
            position_ids,
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
            query_position,
            attention_mask,
            position_ids,
            past_key_values,
            rotary_emb,
        ) = self.prepare_forward_args(*args)

        logit = self.causal_lm(
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

        return logit


class DecoderOnlyForCausalLM(nn.Module):
    """A specialized wrapper for Causal Language Models optimized for RBLN compilation.

    This class adapts Huggingface's CausalLM (or similar models) for RBLN deployment by:
    1. Managing model phases (prefill/decode) throughout the computation graph
    2. Handling output shape alignments for static compilation
    3. Coordinating between the original model and RBLN-optimized components

    The class serves as an intermediate layer between DecoderOnlyWrapper and the core model,
    focusing on maintaining correct model behavior while enabling RBLN-specific optimizations.

    Args:
        causal_lm (PreTrainedModel): Original Huggingface causal language model
        model (DecoderOnlyModel): RBLN-optimized model instance

    Attributes:
        config: Configuration from the original causal language model
        _original_mod: Reference to the original model for components like lm_head
        model: RBLN-optimized decoder model instance
        _phase: Current processing phase ("prefill" or "decode")
    """

    def __init__(self, causal_lm: PreTrainedModel, model: nn.Module):
        super().__init__()
        self.config = causal_lm.config
        self._original_mod = causal_lm
        self.model = model
        self._phase = "prefill"
        self.lm_head = self._original_mod.lm_head

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, phase: str):
        self._phase = phase
        self.model.phase = phase

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


class DecoderOnlyModel(nn.Module):
    """A modified decoder-only model implementation optimized for RBLN compilation.

    Args:
        model: Original Huggingface model to adapt
        layers (List[DecoderOnlyLayer]): Modified transformer layers optimized for RBLN

    Attributes:
        _original_mod: Reference to original Huggingface model
        layers: ModuleList of RBLN-optimized transformer layers
        _phase: Current processing phase ("prefill" or "decode")
    """

    def __init__(
        self,
        model,
        layers: List["DecoderOnlyLayer"],
        partition_len=None,
        max_seq_len=None,
        kvcache_block_size=None,
        use_learned_pos_emb=None,
        sliding_window_layers=None,
    ):
        super().__init__()
        self._original_mod = model
        self.layers = nn.ModuleList(layers)
        self._phase = "prefill"
        self.partition_len = partition_len
        self.kvcache_block_size = kvcache_block_size
        self.max_seq_len = max_seq_len
        self.use_learned_pos_emb = use_learned_pos_emb
        self.sliding_window_layers = sliding_window_layers

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, phase: str):
        self._phase = phase
        for layer in self.layers:
            layer.phase = phase

    @property
    def attn_impl(self) -> str:
        return "eager" if self.partition_len is None else "flash_attn"

    @property
    def hidden_multiplier(self):
        return 1

    def convert_sequence_positions_for_flash_attn(self, seq_positions, max_seq_len):
        if self.attn_impl not in ["flash_attn"]:
            raise NotImplementedError(f"Unknown attn_impl ({self.attn_impl}).")
        partition_len = self.partition_len
        num_partition = max_seq_len // partition_len

        cs = seq_positions.repeat(num_partition, 1).transpose(0, 1)
        pidx = torch.arange(num_partition)
        cache_pos_for_partitions = torch.clamp(cs - pidx * partition_len, 0, partition_len)
        return cache_pos_for_partitions

    def get_local_cache_positions(self, position_ids, query_position):
        max_cache_len = self._original_mod.config.sliding_window
        valid_input_len = 1 if query_position is None else query_position + 1
        cache_seq_len = torch.clamp(position_ids, max=max_cache_len)[:, :1]  # past seen tokens
        cache_offset = (
            torch.clamp(position_ids, max=max_cache_len)[:, :1] + valid_input_len
        )  # cache offset for next steps

        return cache_seq_len, cache_offset

    def get_last_layernorm(self) -> nn.LayerNorm:
        return self._original_mod.norm

    def get_embedding(self) -> nn.Embedding:
        return self._original_mod.embed_tokens

    def get_pos_embedding(self) -> nn.Embedding:
        raise NotImplementedError(
            "The 'get_pos_embedding' method is not implemented. Please define this method in a subclass."
        )

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

        elif self.use_learned_pos_emb:
            batch_size = inputs_embeds.shape[0]
            hidden_all = []
            for i in range(batch_size):
                positions_idx = position_ids[i]
                position_weight = self.get_pos_embedding().weight[2:]
                position = position_weight[positions_idx]
                batch_hidden = position + inputs_embeds[i]
                hidden_all.append(batch_hidden)
            hidden_states = torch.stack(hidden_all, dim=0)
            cos, sin = None, None

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

        for layer_idx, layer in enumerate(self.layers):
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


class DecoderOnlyLayer(nn.Module):
    """A single transformer layer adapted for RBLN compilation with static shapes.

    This layer implements a modified transformer block that includes:
    1. Self-attention mechanism (either standard or flash attention)
    2. Feed-forward network (FFN)
    3. Layer normalization
    4. Residual connections

    The layer is specifically designed to:
    - Support compilation to RBLN custom ops
    - Maintain static tensor shapes throughout computations
    - Handle both prefill and decode phases efficiently
    - Manage attention state transitions properly

    Args:
        layer: Original transformer layer module to wrap
        self_attn (DecoderOnlyAttention): Modified attention module optimized for RBLN

    Attributes:
        _original_mod: Reference to original layer for accessing components
        self_attn: Modified attention mechanism mapped to RBLN ops at compile time
        phase: Current operation phase ("prefill" or "decode")
    """

    def __init__(self, layer, self_attn: "DecoderOnlyAttention"):
        super().__init__()
        self._original_mod = layer
        self.self_attn = self_attn
        self._phase = "prefill"

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, phase: str):
        self._phase = phase
        self.self_attn.phase = phase

    def get_pre_attention_layernorm(self) -> nn.LayerNorm:
        return self._original_mod.input_layernorm

    def get_post_attention_layernorm(self) -> nn.LayerNorm:
        return self._original_mod.post_attention_layernorm

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
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.get_post_attention_layernorm()(hidden_states)
        hidden_states = self._original_mod.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class DecoderOnlyAttention(nn.Module):
    """Attention implementation for decoder-only models optimized for RBLN compilation.

    This class implements a modified version of the standard attention mechanism that:
    1. Supports static shape requirements for RBLN compilation
    2. Handles explicit batch and position management

    Args:
        self_attn: Original attention module from the base model
    """

    def __init__(self, self_attn, use_attention_mask, use_position_ids, kvcache_block_size, is_sliding=False):
        super().__init__()
        self._original_mod = self_attn
        self.layer_idx = self_attn.layer_idx
        self.num_heads = getattr(self._original_mod, "num_heads", None) or getattr(
            self._original_mod.config, "num_attention_heads"
        )
        self.head_dim = self._original_mod.head_dim
        self._phase = "prefill"
        self.scale = torch.tensor(self.get_attn_scale())

        if hasattr(self._original_mod, "num_key_value_heads"):
            self.num_key_value_heads = self._original_mod.num_key_value_heads
        elif hasattr(self._original_mod, "config") and hasattr(self._original_mod.config, "num_key_value_heads"):
            self.num_key_value_heads = self._original_mod.config.num_key_value_heads
        else:
            self.num_key_value_heads = self.num_heads

        self.use_attention_mask = use_attention_mask
        self.use_position_ids = use_position_ids
        self.is_sliding = is_sliding
        self.attention = self.get_attention()
        self.kvcache_block_size = kvcache_block_size
        self.__post_init__()

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, phase: str):
        self._phase = phase
        self.attention.phase = phase

    def get_attention(self):
        if self.is_sliding:
            return SlidingWindowAttentionOp(
                self.num_heads, self.head_dim, self.num_key_value_heads, self.use_attention_mask, self.use_position_ids
            )
        else:
            return AttentionOp(
                self.num_heads, self.head_dim, self.num_key_value_heads, self.use_attention_mask, self.use_position_ids
            )

    def __post_init__(self):
        self.q_proj = self._original_mod.q_proj
        self.k_proj = self._original_mod.k_proj
        self.v_proj = self._original_mod.v_proj
        self.o_proj = self._original_mod.o_proj

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

    def apply_rotary_pos_embed(self, query_states, key_states, cos, sin):
        return apply_rotary_pos_emb(query_states, key_states, cos, sin)

    def get_attn_scale(self):
        return 1 / math.sqrt(self.head_dim)

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
        if hasattr(self, "q_norm") and hasattr(self, "k_norm"):
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        if cos is not None and sin is not None:
            query_states, key_states = self.apply_rotary_pos_embed(query_states, key_states, cos, sin)

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


class AttentionOp(nn.Module):
    def __init__(
        self, num_heads: int, head_dim: int, num_key_value_heads: int, use_attention_mask: bool, use_position_ids: bool
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.phase = "prefill"
        self.use_attention_mask = use_attention_mask
        self.use_position_ids = use_position_ids

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

        if self.use_attention_mask and not self.use_position_ids:
            attn_mask = attn_mask.unsqueeze(2)

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
            if self.use_attention_mask and not self.use_position_ids:
                attn_output = torch.ops.rbln_custom_ops.paged_attn_decode(
                    q=query_state,
                    k=key_state,
                    v=value_state,
                    mask=attn_mask,
                    kcache=past_key_state.unsqueeze(2),
                    vcache=past_value_state.unsqueeze(2),
                    seq=seq_position,
                    scale=scale,
                    block_table=block_tables,
                    block_size=block_size,
                )
            else:
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
                    mask=attn_mask if self.use_position_ids else None,
                )

        else:
            if self.use_attention_mask and not self.use_position_ids:
                attn_output = torch.ops.rbln_custom_ops.paged_attn_prefill(
                    q=query_state,
                    k=key_state,
                    v=value_state,
                    mask=attn_mask,
                    kcache=past_key_state.unsqueeze(2),
                    vcache=past_value_state.unsqueeze(2),
                    seq=seq_position,
                    scale=scale,
                    block_table=block_tables,
                    block_size=block_size,
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
                    is_bidirectional=True if self.phase == "image_prefill" else False,  # FIXME, Hard-coded for Gemma3.
                    mask=attn_mask if self.use_position_ids else None,
                )

        attn_output = attn_output.view(batch_size, self.num_heads, -1, self.head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, -1, self.num_heads * self.head_dim)

        return attn_output


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


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Applies Rotary Position Embedding to the query and key tensors."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb_partial(query_states, key_states, cos, sin, ndim) -> Tuple[torch.Tensor, torch.Tensor]:
    # Partial rotary embedding
    query_rot, query_pass = (
        query_states[..., :ndim],
        query_states[..., ndim:],
    )
    key_rot, key_pass = (
        key_states[..., :ndim],
        key_states[..., ndim:],
    )

    # [batch_size, seq_length, num_heads, head_dim // config.partial_rotary_factor]
    query_rot, key_rot = apply_rotary_pos_emb(query_rot, key_rot, cos, sin)

    # [batch_size, seq_length, num_heads, head_dim]
    query_states = torch.cat((query_rot, query_pass), dim=-1)
    key_states = torch.cat((key_rot, key_pass), dim=-1)
    return query_states, key_states


class RotaryEmbedding(nn.Module):
    """RotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(
        self,
        config: PretrainedConfig,
        max_seq_len_cached: int,
    ):
        super().__init__()

        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            rope_type = "default"

        inv_freq, attention_scaling = ROPE_INIT_FUNCTIONS[rope_type](config, max_seq_len_cached)
        cache_position = torch.arange(0, max_seq_len_cached, dtype=torch.float32)
        cache_position_expanded = cache_position[:, None]

        if rope_type == "dynamic":
            freqs = cache_position_expanded.float() * inv_freq.float()
        else:
            inv_freq_expanded = inv_freq[None, :]
            freqs = cache_position_expanded.float() @ inv_freq_expanded.float()

        emb = torch.cat((freqs, freqs), dim=-1)

        cos = emb.cos() * attention_scaling
        sin = emb.sin() * attention_scaling

        self.register_buffer("_cos_cached", cos, persistent=False)
        self.register_buffer("_sin_cached", sin, persistent=False)

    def forward(self, x, seq_len):
        return (
            self._cos_cached[:seq_len].to(dtype=x.dtype),
            self._sin_cached[:seq_len].to(dtype=x.dtype),
        )


class DecoderOnlyFlashAttention(DecoderOnlyAttention):
    def __init__(self, self_attn, kvcache_partition_len, kvcache_block_size, use_attention_mask, use_position_ids):
        self.kvcache_partition_size = kvcache_partition_len
        super().__init__(
            self_attn=self_attn,
            use_attention_mask=use_attention_mask,
            use_position_ids=use_position_ids,
            kvcache_block_size=kvcache_block_size,
        )

    def get_attention(self):
        return FlashAttentionOp(
            self.num_heads,
            self.head_dim,
            self.num_key_value_heads,
            self.kvcache_partition_size,
            self.use_attention_mask,
            self.use_position_ids,
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

        if hasattr(self, "q_norm") and hasattr(self, "k_norm"):
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


class FlashAttentionOp(AttentionOp):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        num_key_value_heads: int,
        kvcache_partition_len: int,
        use_attention_mask: bool,
        use_position_ids: bool,
    ):
        super().__init__(
            num_heads=num_heads,
            head_dim=head_dim,
            num_key_value_heads=num_key_value_heads,
            use_attention_mask=use_attention_mask,
            use_position_ids=use_position_ids,
        )
        self.kvcache_partition_size = kvcache_partition_len

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
        if self.use_attention_mask and not self.use_position_ids:
            attn_mask = attn_mask.unsqueeze(2)

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
            if self.use_attention_mask and not self.use_position_ids:
                attn_output = torch.ops.rbln_custom_ops.paged_flash_attn_decode(
                    q=query_state,
                    k=key_state,
                    v=value_state,
                    mask=attn_mask,
                    kcache=past_key_state.unsqueeze(2),
                    vcache=past_value_state.unsqueeze(2),
                    seq=seq_position,
                    scale=scale,
                    block_table=block_tables,
                    block_size=kvcache_block_size,
                    partition=self.kvcache_partition_size,
                )
            else:
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
                    mask=attn_mask if self.use_position_ids else None,
                )
        else:
            if self.use_attention_mask and not self.use_position_ids:
                attn_output = torch.ops.rbln_custom_ops.paged_flash_attn_prefill(
                    q=query_state,
                    k=key_state,
                    v=value_state,
                    mask=attn_mask,
                    kcache=past_key_state.unsqueeze(2),
                    vcache=past_value_state.unsqueeze(2),
                    seq=seq_position,
                    scale=scale,
                    block_table=block_tables,
                    block_size=kvcache_block_size,
                    partition=self.kvcache_partition_size,
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
                    is_bidirectional=True if self.phase == "image_prefill" else False,
                    mask=attn_mask if self.use_position_ids else None,
                )

        # reshape for removing repeat_kv
        attn_output = attn_output.view(batch_size, self.num_heads, -1, self.head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, -1, self.num_heads * self.head_dim)

        return attn_output


class SlidingWindowAttentionOp(AttentionOp):
    def forward(
        self,
        query_state: torch.Tensor,
        key_state: torch.Tensor,
        value_state: torch.Tensor,
        attn_mask: torch.Tensor,
        past_key_state: torch.Tensor,
        past_value_state: torch.Tensor,
        seq_position: Tuple[torch.Tensor],
        scale: torch.Tensor,
        block_tables: torch.Tensor,
        block_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            attn_output = torch.ops.rbln_custom_ops.paged_sliding_window_attn_decode(
                q=query_state,
                k=key_state,
                v=value_state,
                kcache=past_key_state.unsqueeze(2),
                vcache=past_value_state.unsqueeze(2),
                cache_seq_len=seq_position[0],
                cache_offset=seq_position[1],
                scale=scale,
                block_table=block_tables,
                block_size=block_size,
            )
        else:
            attn_output = torch.ops.rbln_custom_ops.paged_sliding_window_attn_prefill(
                q=query_state,
                k=key_state,
                v=value_state,
                kcache=past_key_state.unsqueeze(2),
                vcache=past_value_state.unsqueeze(2),
                cache_seq_len=seq_position[0],
                cache_offset=seq_position[1],
                scale=scale,
                block_table=block_tables,
                block_size=block_size,
                is_bidirectional=True if self.phase == "image_prefill" else False,
            )

        # reshape for removing repeat_kv
        attn_output = attn_output.view(batch_size, self.num_heads, -1, self.head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, -1, self.num_heads * self.head_dim)

        return attn_output
