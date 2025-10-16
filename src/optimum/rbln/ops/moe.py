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

from typing import Literal, Optional

import torch
from torch import Tensor


ACT_TYPES = Literal[
    "gelu",
    "gelu_10",
    "gelu_fast",
    "gelu_new",
    "gelu_python",
    "gelu_pytorch_tanh",
    "gelu_accurate",
    "laplace",
    "leaky_relu",
    "linear",
    "mish",
    "quick_gelu",
    "relu",
    "relu2",
    "relu6",
    "sigmoid",
    "silu",
    "swish",
    "tanh",
    "prelu",
]


@torch.library.custom_op(
    "rbln_custom_ops::custom_moe_glu",
    mutates_args=(),
)
def custom_moe_glu(
    hidden_states: Tensor,
    gate_proj_weight: Tensor,
    up_proj_weight: Tensor,
    down_proj_weight: Tensor,
    masked_routing_weight: Tensor,
    # act_fn: str,
    expert_select_count: Tensor,
    gate_proj_bias: Optional[Tensor] = None,
    up_proj_bias: Optional[Tensor] = None,
    down_proj_bias: Optional[Tensor] = None,
) -> Tensor:
    """
    Customized MoE GLU operation.

    Expected tensor shapes:
    - hidden_states: [batch*seq_len, hidden_size]
    - gate_proj_weight: [num_experts, hidden_size, intermediate_size]
    - up_proj_weight: [num_experts, hidden_size, intermediate_size]
    - down_proj_weight: [num_experts, intermediate_size, hidden_size]

    - gate_proj_bias: [num_experts, intermediate_size]
    - up_proj_bias: [num_experts, intermediate_size]
    - down_proj_bias: [num_experts, hidden_size]

    - masked_routing_weight: [batch * seq_len, num_experts]

    Returns:
        Tensor: [batch * seq_len, hidden_size]
    """

    out = torch.zeros_like(hidden_states)
    expert_cnt = gate_proj_weight.shape[0]
    for i in range(expert_cnt):
        gate = torch.nn.functional.linear(hidden_states, gate_proj_weight[i])
        up = torch.nn.functional.linear(hidden_states, up_proj_weight[i])
        mul = torch.nn.functional.silu(gate) * up
        down = torch.nn.functional.linear(mul, down_proj_weight[i])
        out += down * masked_routing_weight[:, i : i + 1]

    return out


@custom_moe_glu.register_fake
def custom_moe_glu_fake(
    hidden_states: Tensor,
    gate_proj_weight: Tensor,
    up_proj_weight: Tensor,
    down_proj_weight: Tensor,
    masked_routing_weight: Tensor,
    expert_select_count: Tensor,
    # act_fn: ACT_TYPES,
    gate_proj_bias: Optional[Tensor] = None,
    up_proj_bias: Optional[Tensor] = None,
    down_proj_bias: Optional[Tensor] = None,
    # gate_proj_scale: Optional[Tensor] = None,
    # up_proj_scale: Optional[Tensor] = None,
    # down_proj_scale: Optional[Tensor] = None,
) -> Tensor:
    return torch.empty_like(hidden_states)


@torch.library.custom_op(
    "rbln_custom_ops::custom_moe_ff",
    mutates_args=(),
)
def custom_moe_ff(
    hidden_states: Tensor,
    gate_proj_weight: Tensor,
    down_proj_weight: Tensor,
    masked_routing_weight: Tensor,
    # act_fn: str,
    gate_proj_bias: Optional[Tensor] = None,
    down_proj_bias: Optional[Tensor] = None,
) -> Tensor:
    """
    Customized MoE FF operation.

    Expected tensor shapes:
    - hidden_states: [batch * seq_len, hidden_size]
    - gate_proj_weight: [hidden_size, num_experts * intermediate_size]
    - down_proj_weight: [num_experts * intermediate_size, hidden_size]
    - masked_routing_weight: [batch * seq_len, num_experts]

    Returns:
        Tensor: [batch * seq_len, hidden_size]
    """
    return torch.empty_like(hidden_states)


@custom_moe_ff.register_fake
def custom_moe_ff_fake(
    hidden_states: Tensor,
    gate_proj_weight: Tensor,
    down_proj_weight: Tensor,
    masked_routing_weight: Tensor,
    # act_fn: str,
    gate_proj_bias: Optional[Tensor] = None,
    down_proj_bias: Optional[Tensor] = None,
) -> Tensor:
    return torch.empty_like(hidden_states)
