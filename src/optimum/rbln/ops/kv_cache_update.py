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

from functools import lru_cache

import torch
from transformers.pytorch_utils import is_torch_greater_or_equal_than_2_4


if is_torch_greater_or_equal_than_2_4:
    register_fake = torch.library.register_fake
else:
    register_fake = torch.library.impl_abstract


@lru_cache
def register_rbln_custom_cache_update():
    # Define the RBLN custom operation "rbln_cache_update" which updates a cache tensor with a given state tensor.
    # This operation is designed to perform in-place updates directly on the device without needing to transfer the cache back to the host.
    # The `position` parameter specifies the start index for the update along the specified axis, allowing flexible updates to any part of the cache tensor.
    torch.library.define("rbln_custom_ops::rbln_cache_update", "(Tensor x, Tensor y, Tensor z, Tensor w) -> Tensor")

    # Implementation of the "rbln_cache_update" operation for the CPU.
    @torch.library.impl("rbln_custom_ops::rbln_cache_update", "cpu")
    def rbln_cache_update_cpu(cache, state, position, axis):
        assert position.dim() == 0
        assert axis.dim() == 0

        # Calculate the start (s) and end (e) indices for the update based on the position and the shape of the state tensor along the specified axis.
        s = position  # Start index for the update, specified by the position.
        e = (
            position + state.shape[axis]
        )  # End index is determined by adding the size of the state along the given axis.

        # Update the specified portion of the cache tensor with the state tensor, using `slice_scatter`.
        # This operation modifies the cache tensor in-place directly on the device, avoiding any unnecessary transfers between host and device.
        updated_cache = cache.slice_scatter(state, dim=axis, start=s, end=e)

        # Return the updated cache tensor.
        return updated_cache

    # Register a "fake" implementation of the "rbln_cache_update" operation.
    # This serves as an abstract definition for the RBLN compiler to recognize the operation and generate an optimized implementation.
    @register_fake("rbln_custom_ops::rbln_cache_update")
    def rbln_cache_update_abstract(cache, state, position, axis):
        # Return a tensor with the same shape as the input cache tensor.
        # This is a placeholder for the abstract implementation and does not perform any actual computation.
        # Like the actual implementation, the abstraction assumes in-place device-side updates.
        return torch.empty_like(cache)
