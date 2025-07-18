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
from torch import Tensor


@torch.library.custom_op("rbln_custom_ops::linear", mutates_args=())
def linear(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    output_shape = list(input.shape[:-1])
    output_shape += [weight.shape[0]]
    return torch.empty(size=output_shape, dtype=input.dtype, device=input.device, requires_grad=input.requires_grad)


@linear.register_fake
def linear_fake(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    output_shape = list(input.shape[:-1])
    output_shape += [weight.shape[0]]
    return torch.empty(size=output_shape, dtype=input.dtype, device=input.device, requires_grad=input.requires_grad)
