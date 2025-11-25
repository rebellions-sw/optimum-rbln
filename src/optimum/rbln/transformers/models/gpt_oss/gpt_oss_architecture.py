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
        router_logits = F.linear(hidden_states, self.weight, self.bias)  # (seq_len, num_experts)
        return router_logits


class RBLNGptOssExperts(nn.Module):
    def __init__(self, model, k: int = 2):
        super().__init__()
        self.intermediate_size = model.intermediate_size
        self.num_experts = model.num_experts
        self.hidden_size = model.hidden_size

        if hasattr(model, "gate_up_proj_blocks"):
            self._support_mxfp4 = True
        else:
            self._support_mxfp4 = False

        if self._support_mxfp4:
            self.register_buffer(
                "gate_proj_blocks",
                model.gate_up_proj_blocks.data[:, ::2, :, :].reshape(self.num_experts, self.intermediate_size, -1),
            )
            self.register_buffer("gate_proj_scales", model.gate_up_proj_scales.data[:, ::2, :])
            self.register_buffer(
                "gate_proj_bias",
                model.gate_up_proj_bias.data[:, ::2].reshape(self.num_experts, self.intermediate_size),
            )

            self.register_buffer(
                "up_proj_blocks",
                model.gate_up_proj_blocks.data[:, 1::2, :, :].reshape(self.num_experts, self.intermediate_size, -1),
            )
            self.register_buffer("up_proj_scales", model.gate_up_proj_scales.data[:, 1::2, :])
            self.register_buffer(
                "up_proj_bias", model.gate_up_proj_bias.data[:, 1::2].reshape(self.num_experts, self.intermediate_size)
            )

            self.register_buffer(
                "down_proj_blocks", model.down_proj_blocks.data.reshape(self.num_experts, self.hidden_size, -1)
            )
            self.register_buffer("down_proj_scales", model.down_proj_scales.data)
            self.register_buffer("down_proj_bias", model.down_proj_bias.data)
        else:
            self.gate_up_proj = model.gate_up_proj
            self.gate_up_proj_bias = model.gate_up_proj_bias
            self.down_proj = model.down_proj
            self.down_proj_bias = model.down_proj_bias

        self.alpha = model.alpha  # 1.702
        self.limit = model.limit  # 7.0

        self.k = k

    def forward(
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor
    ) -> torch.Tensor:
        if self._support_mxfp4:
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
                router_logits,
                torch.tensor(self.alpha, dtype=hidden_states.dtype),
                torch.tensor(self.limit, dtype=hidden_states.dtype),
                k=self.k,
            )
        else:
            batch_size = hidden_states.shape[0]
            hidden_states = hidden_states.reshape(-1, self.hidden_size)  # (num_tokens, hidden_size)
            num_experts = routing_weights.shape[1]

            hidden_states = hidden_states.repeat(num_experts, 1)
            hidden_states = hidden_states.view(num_experts, -1, self.hidden_size)

            gate_up = torch.bmm(hidden_states, self.gate_up_proj.to(hidden_states.dtype)) + self.gate_up_proj_bias[
                ..., None, :
            ].to(hidden_states.dtype)
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
        self.experts = RBLNGptOssExperts(model.experts, k=self.router.top_k)

    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits = self.router(hidden_states)

        routed_out = self.experts(
            hidden_states,
            router_logits=router_logits
        )
        routed_out = routed_out.reshape(batch_size, sequence_length, hidden_dim)
        return routed_out
