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
from ..decoderonly.decoderonly_architecture import DecoderOnlyAttention, DecoderOnlyLayer, DecoderOnlyWrapper


class RBLNGptOssWrapper(DecoderOnlyWrapper):
    def get_rbln_layer_class(self):
        return RBLNGptOssLayer


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

        return router_scores, router_indices

        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight, self.bias)  # (seq_len, num_experts)
        routing_weights = torch.nn.functional.softmax(router_logits, dim=1, dtype=torch.float)

        # selected_experts: (batch * sequence_length, top_k)
        selected_weights, selected_experts = torch.topk(routing_weights, k=self.top_k, dim=-1)
        mask = torch.zeros_like(routing_weights, dtype=torch.float32)
        un_mask = torch.ones_like(selected_experts, dtype=torch.float32)
        mask.scatter_(1, selected_experts, un_mask)

        masked_routing_weights = routing_weights * mask
        ## get size per expert
        expert = router_logits.shape[1]
        zeros = torch.zeros(expert, dtype=torch.int32)
        ones = torch.ones_like(selected_experts.view(-1), dtype=torch.int32)
        expert_select_count = torch.scatter_add(zeros, dim=0, index=selected_experts.view(-1), src=ones)

        return masked_routing_weights, expert_select_count


class RBLNGptOssExperts(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.intermediate_size = model.intermediate_size
        self.num_experts = model.num_experts
        self.hidden_size = model.hidden_size
        self.expert_dim = model.expert_dim

        self.gate_up_proj = model.gate_up_proj
        self.gate_up_proj_bias = model.gate_up_proj_bias
        self.down_proj = model.down_proj
        self.down_proj_bias = model.down_proj_bias
        self.alpha = model.alpha  # 1.702
        self.limit = model.limit  # 7.0

    def forward(self, hidden_states: torch.Tensor, router_indices=None, routing_weights=None) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)  # (num_tokens, hidden_size)
        num_experts = routing_weights.shape[1]

        hidden_states = hidden_states.repeat(num_experts, 1)
        hidden_states = hidden_states.view(num_experts, -1, self.hidden_size)

        gate_up = torch.bmm(hidden_states, self.gate_up_proj.to(hidden_states.dtype)) + self.gate_up_proj_bias[..., None, :].to(hidden_states.dtype)
        gate, up = gate_up[..., ::2], gate_up[..., 1::2]
        gate = gate.clamp(min=None, max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        glu = gate * torch.sigmoid(gate * self.alpha)
        next_states = torch.bmm(((up + 1.0) * glu), self.down_proj.to(hidden_states.dtype))
        next_states = next_states + self.down_proj_bias[..., None, :].to(hidden_states.dtype)
        next_states = next_states.view(num_experts, batch_size, -1, self.hidden_size)
        next_states = next_states * routing_weights.transpose(0, 1).view(num_experts, batch_size, -1)[..., None]
        next_states = next_states.sum(dim=0)

        return next_states


class RBLNGptOssMLP(nn.Module):
    def __init__(self, model):
        super().__init__()
        self._original_mod = model
        self.router = RBLNGptOssTopKRouter(model.router)
        self.experts = RBLNGptOssExperts(model.experts)

    def forward(self, hidden_states):
        router_scores, router_indices = self.router(hidden_states)  # (num_experts, seq_len)
        routed_out = self.experts(hidden_states, router_indices=router_indices, routing_weights=router_scores)
        return routed_out  # , router_scores
