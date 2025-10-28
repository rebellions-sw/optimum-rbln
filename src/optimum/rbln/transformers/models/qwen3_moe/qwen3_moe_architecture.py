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
from torch import nn
from transformers.activations import ACT2FN

from ..decoderonly.configuration_decoderonly import RBLNLoRAConfig
from ..decoderonly.decoderonly_architecture import DecoderOnlyAttention, DecoderOnlyLayer, DecoderOnlyWrapper


class QWEN3MoeWrapper(DecoderOnlyWrapper):
    def get_rbln_layer_class(self):
        return Qwen3MoeLayer

    def get_rbln_attn_class(self):
        return Qwen3MoeAttention


class Qwen3MoeAttention(DecoderOnlyAttention):
    def __post_init__(self):
        self.q_proj = self._original_mod.q_proj
        self.k_proj = self._original_mod.k_proj
        self.v_proj = self._original_mod.v_proj
        self.o_proj = self._original_mod.o_proj
        self.q_norm = self._original_mod.q_norm
        self.k_norm = self._original_mod.k_norm


class Qwen3MoeLayer(DecoderOnlyLayer):
    def __init__(self, layer, self_attn: DecoderOnlyAttention, lora_config: Optional[RBLNLoRAConfig] = None):
        super().__init__(layer, self_attn, lora_config)
        self.mlp = (
            Qwen3MoeSparseMoeBlock(self._original_mod.mlp)
            if self._original_mod.mlp.__class__.__name__ == "Qwen3MoeSparseMoeBlock"
            else self._original_mod.mlp
        )

    def get_mlp(self) -> nn.Module:
        return self.mlp


class Qwen3MoeSparseMoeBlock(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.num_experts = model.num_experts
        self.top_k = model.top_k
        self.norm_topk_prob = model.norm_topk_prob
        self.gate = model.gate
        # self.shared_expert = model.shared_expert
        # self.shared_expert_gate = model.shared_expert_gate
        self.experts = Qwen3MoeMLP(model.experts)

    def get_masked_routing_weights(self, router_logits):
        if self.norm_topk_prob:
            selected_weights, selected_experts = torch.topk(router_logits, k=self.top_k, dim=-1)
            selected_weights = torch.nn.functional.softmax(selected_weights, dim=1, dtype=torch.float)
            masked_routing_weights = torch.zeros_like(router_logits, dtype=torch.float32)
            masked_routing_weights.scatter_(1, selected_experts, selected_weights)
        else:
            # routing_weights: (batch * sequence_length, n_experts)
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

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)
        masked_routing_weights, expert_select_count = self.get_masked_routing_weights(router_logits)
        final_hidden_states = self.experts(hidden_states, masked_routing_weights, expert_select_count)

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states


class Qwen3MoeMLP(nn.Module):
    def __init__(self, expert_list):
        super().__init__()
        self.config = expert_list[0].config
        self.hidden_size = expert_list[0].hidden_size
        self.intermediate_size = expert_list[0].intermediate_size
        self.act_fn = ACT2FN[self.config.hidden_act]
        self.act_fn_name = self.config.hidden_act

        # RBLN-optimized MLP
        self.num_experts = len(expert_list)
        self.gate_proj = nn.Linear(self.hidden_size, self.num_experts * self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.num_experts * self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.num_experts * self.intermediate_size, self.hidden_size, bias=False)
        self.gate_proj.weight.data = torch.stack([expert.gate_proj.weight.data for expert in expert_list], dim=0)
        self.up_proj.weight.data = torch.stack([expert.up_proj.weight.data for expert in expert_list], dim=0)
        self.down_proj.weight.data = torch.stack([expert.down_proj.weight.data for expert in expert_list], dim=0)

    def forward(self, x, masked_routing_weights, expert_select_count):
        # # masked_routing_weights: (batch * sequence_length, num_experts)
        # # x: (batch * sequence_length, hidden_size)
        # # y: (batch * sequence_length, num_experts * intermediate_size)
        # y = self.act_fn(self.gate_proj(x)) * self.up_proj(x)

        # # elementwise multiplication of y and routing_weights
        # y = y.reshape(-1, self.num_experts, self.intermediate_size) * masked_routing_weights[:, :, None]
        # y = y.reshape(-1, self.num_experts * self.intermediate_size)

        # return self.down_proj(y)
        return torch.ops.rbln_custom_ops.custom_moe_glu(
            x,
            self.gate_proj.weight,
            self.up_proj.weight,
            self.down_proj.weight,
            masked_routing_weights,
            expert_select_count,  # count for each expert
            # self.act_fn_name,
        )
