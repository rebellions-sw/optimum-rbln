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


from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from ..decoderonly.configuration_decoderonly import RBLNLoRAConfig
from ..decoderonly.decoderonly_architecture import (
    DecoderOnlyAttention,
    DecoderOnlyLayer,
    DecoderOnlyWrapper,
    DecoderOnlyAttention,
)


class RBLNGptOssWrapper(DecoderOnlyWrapper):
    def get_rbln_attn_class(self):
        return RBLNGptOssAttention

    def get_rbln_layer_class(self):
        return RBLNGptOssLayer


class RBLNGptOssAttention(DecoderOnlyAttention):
    def __post_init__(self):
        # Initialize LoRA weights if configured, which will replace linear layers
        if self.lora_config:
            self._init_lora_weights()
        else:
            # Use original linear layers if no LoRA
            self.q_proj = self._original_mod.q_proj
            self.k_proj = self._original_mod.k_proj
            self.v_proj = self._original_mod.v_proj
            self.o_proj = self._original_mod.o_proj
            self.sinks = self._original_mod.sinks.data[:, None]


class RBLNGptOssLayer(DecoderOnlyLayer):
    def __init__(self, layer, self_attn: DecoderOnlyAttention, lora_config: Optional[RBLNLoRAConfig] = None):
        super().__init__(layer, self_attn, lora_config)
        self.mlp = RBLNGptOssMLP(layer.mlp)

    def get_mlp(self) -> nn.Module:
        return self.mlp


class RBLNGptOssTopKRouter(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.top_k = model.top_k
        self.num_experts = model.num_experts
        self.hidden_dim = model.hidden_dim
        self.weight = model.weight
        self.bias = model.bias

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight, self.bias)  # (seq_len, num_experts)
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)  # (seq_len, top_k)
        router_top_value = torch.nn.functional.softmax(router_top_value, dim=1, dtype=router_top_value.dtype)
        router_scores = torch.zeros_like(router_logits).scatter_(1, router_indices, router_top_value)

        zeros = torch.zeros(self.num_experts, dtype=torch.int32)
        ones = torch.ones_like(router_indices.view(-1), dtype=torch.int32)
        expert_select_count = torch.scatter_add(zeros, dim=0, index=router_indices.view(-1), src=ones)

        return router_scores, router_indices, expert_select_count


class RBLNGptOssExperts(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.intermediate_size = model.intermediate_size
        self.num_experts = model.num_experts
        self.hidden_size = model.hidden_size

        self.gate_proj_blocks = model.gate_up_proj_blocks.data[:, ::2, :, :].reshape(
            self.num_experts, self.intermediate_size, -1
        )
        self.gate_proj_scales = model.gate_up_proj_scales.data[:, ::2, :]
        self.gate_proj_bias = model.gate_up_proj_bias.data[:, ::2].reshape(self.num_experts, self.intermediate_size)

        self.up_proj_blocks = model.gate_up_proj_blocks[:, 1::2, :, :].reshape(
            self.num_experts, self.intermediate_size, -1
        )
        self.up_proj_scales = model.gate_up_proj_scales[:, 1::2, :]
        self.up_proj_bias = model.gate_up_proj_bias[:, 1::2].reshape(self.num_experts, self.intermediate_size)

        self.down_proj_blocks = model.down_proj_blocks.data.reshape(self.num_experts, self.hidden_size, -1)
        self.down_proj_scales = model.down_proj_scales.data
        self.down_proj_bias = model.down_proj_bias.data
        self.alpha = model.alpha  # 1.702
        self.limit = model.limit  # 7.0

    def forward(
        self, hidden_states: torch.Tensor, router_indices=None, routing_weights=None, expert_select_count=None
    ) -> torch.Tensor:
        next_states = torch.ops.rbln_custom_ops.custom_moe_glu_mxfp4(
            hidden_states,
            self.gate_proj_blocks,
            self.gate_proj_scales,
            self.gate_proj_bias,
            self.up_proj_blocks,
            self.up_proj_scales,
            self.up_proj_bias,
            self.down_proj_blocks,
            self.down_proj_scales,
            self.down_proj_bias,
            routing_weights,
            expert_select_count,
            torch.tensor(self.alpha, dtype=hidden_states.dtype),
            torch.tensor(self.limit, dtype=hidden_states.dtype),
        )

        return next_states


class RBLNGptOssMLP(nn.Module):
    def __init__(self, model):
        super().__init__()
        self._original_mod = model
        self.router = RBLNGptOssTopKRouter(model.router)
        self.experts = RBLNGptOssExperts(model.experts)

    def forward(self, hidden_states):
        router_scores, router_indices, expert_select_count = self.router(hidden_states)
        routed_out = self.experts(
            hidden_states,
            router_indices=router_indices,
            routing_weights=router_scores,
            expert_select_count=expert_select_count,
        )
        return routed_out
