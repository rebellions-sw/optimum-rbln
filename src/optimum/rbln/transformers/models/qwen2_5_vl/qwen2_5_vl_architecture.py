import math
from typing import Tuple, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    import rebel


class Qwen2_5_VisionTransformerWrapper(nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self._origin_model = model
        self.spatial_merge_size = model.spatial_merge_size
        self.patch_size = model.patch_size
        self.fullatt_block_indexes = model.fullatt_block_indexes
        self.window_size = model.window_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size
        self.patch_embed = model.patch_embed
        # self.rotary_pos_emb = model.rotary_pos_embed
        self.merger = model.merger

        # Wrap blocks
        self.blocks = self.wrap_vision_blocks(model.blocks)

    def wrap_vision_blocks(self, blocks: torch.nn.ModuleList):
        wrapped_blocks = []
        for i, block in enumerate(blocks):
            is_full_attn = True if i in self.fullatt_block_indexes else False
            wrapped_blocks.append(Qwen2_5_VLVisionBlock(block, is_full_attn))
        return nn.ModuleList(wrapped_blocks)

    def forward(
        self,
        hidden_states: torch.Tensor,
        full_attn_masks: torch.Tensor,
        window_attn_masks: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ):
        full_attn_masks = (1 - full_attn_masks) * torch.finfo(torch.float32).min
        window_attn_masks = (1 - window_attn_masks) * torch.finfo(torch.float32).min

        for i, block in enumerate(self.blocks):
            attn_masks = full_attn_masks if i in self.fullatt_block_indexes else window_attn_masks
            hidden_states = block(hidden_states, attn_masks, [cos, sin])

        hidden_states = self.merger(hidden_states)

        return hidden_states


class Qwen2_5_VLVisionBlock(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, is_full_attn: bool):
        super().__init__()
        self._origin_model = model
        self.norm1 = model.norm1
        self.norm2 = model.norm2
        if is_full_attn:
            self.attn = Qwen2_5_VLVisionFullAttention(model.attn)
        else:
            self.attn = Qwen2_5_VLVisionWindowAttention(model.attn)
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


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    # tensor_a.shape: (8, 1, 64, 80)
    # tenosr_b.shape: (8, 16, 64, 80)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class Qwen2_5_VLVisionFullAttention(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self._origin_model = model
        self.num_heads = model.num_heads
        self.head_dim = model.head_dim
        self.qkv = model.qkv
        self.proj = model.proj

    def _shape(self, tensor: torch.Tensor):
        return tensor[:, :, :, :].transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_masks: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        hidden_states = hidden_states.unsqueeze(0)
        qkv = self.qkv(hidden_states).reshape(1, seq_length, 3, self.num_heads, -1).permute(2, 0, 1, 3, 4)
        q = self._shape(qkv[0])
        k = self._shape(qkv[1])
        v = self._shape(qkv[2])

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        attn_weights = attn_weights + attn_masks
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(1, seq_length, -1)
        attn_output = self.proj(attn_output).squeeze(0)

        return attn_output


class Qwen2_5_VLVisionWindowAttention(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self._origin_model = model
        self.num_heads = model.num_heads
        self.head_dim = model.head_dim
        self.qkv = model.qkv
        self.proj = model.proj
        self.window_size = 64

    def _shape(self, tensor: torch.Tensor):
        return tensor[:, :, :, :].transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_masks: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]

        window_hidden_states = []
        for i in range(0, seq_length, self.window_size):
            window_hidden_states.append(hidden_states[i : i + self.window_size])
        hidden_states = torch.stack(window_hidden_states)
        batch_size = hidden_states.shape[0]

        qkv = (
            self.qkv(hidden_states).reshape(batch_size, self.window_size, 3, self.num_heads, -1).permute(2, 0, 1, 3, 4)
        )
        q = self._shape(qkv[0])
        k = self._shape(qkv[1])
        v = self._shape(qkv[2])

        cos, sin = position_embeddings
        cos = cos.reshape(batch_size, 1, seq_length // batch_size, -1)
        sin = sin.reshape(batch_size, 1, seq_length // batch_size, -1)
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)

        # breakpoint()
        attn_weights = attn_weights + attn_masks
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(1, seq_length, -1)
        attn_output = self.proj(attn_output).squeeze(0)

        return attn_output
