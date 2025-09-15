import math
from typing import Tuple

import torch
import torch.nn as nn

from ..decoderonly.decoderonly_architecture import (
    DecoderOnlyWrapper,
    apply_rotary_pos_emb,
)


class Qwen2VisionTransformerWrapper(nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self._original_mod = model
        self.merger = model.merger
        self.blocks = self.wrap_vision_blocks(model.blocks)

    def wrap_vision_blocks(self, blocks: torch.nn.ModuleList):
        wrapped_blocks = []
        for i, block in enumerate(blocks):
            wrapped_blocks.append(Qwen2VLVisionBlock(block))
        return nn.ModuleList(wrapped_blocks)

    def forward(
        self,
        hidden_states: torch.Tensor,
        full_attn_masks: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ):
        full_attn_masks = (1 - full_attn_masks) * torch.finfo(torch.float32).min

        for block in self.blocks:
            hidden_states = block(hidden_states, full_attn_masks, [cos, sin])

        return self.merger(hidden_states)


class Qwen2VLVisionBlock(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self._origin_model = model
        self.norm1 = model.norm1
        self.norm2 = model.norm2

        self.attn = VisionAttention(model.attn)
        self.mlp = model.mlp

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_masks: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            attn_masks,
            position_embeddings,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class VisionAttention(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self._origin_model = model
        self.num_heads = model.num_heads
        self.head_dim = getattr(model, "head_dim", model.proj.in_features // model.num_heads)
        self.qkv = model.qkv
        self.proj = model.proj

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_masks: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        hidden_states = hidden_states.unsqueeze(0)
        q, k, v = (
            self.qkv(hidden_states).reshape(1, seq_length, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4).unbind(0)
        )

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        attn_weights = attn_weights + attn_masks
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(1, seq_length, -1)
        attn_output = self.proj(attn_output).squeeze(0)

        return attn_output


class Qwen2VL_LanguageModelWrapper(DecoderOnlyWrapper):
    def prepare_forward_args(self, *args):
        args = list(args)
        input_ids = None if self.rbln_config.use_inputs_embeds else args.pop(0)
        inputs_embeds = args.pop(0) if self.rbln_config.use_inputs_embeds else None
        cache_position = args.pop(0)
        global_block_tables = args.pop(0)
        local_block_tables = None
        position_embeds = args.pop(0)
        query_position = args.pop(0) if self.phase == "prefill" else None
        position_ids = None
        attention_mask = args.pop(0) if self.rbln_config.use_attention_mask else None
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
            position_embeds,
        )
