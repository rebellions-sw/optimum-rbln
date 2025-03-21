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
    torch.library.define("rbln_custom_ops::rbln_cache_update", "(Tensor x, Tensor y, Tensor z, Tensor w) -> Tensor")

    @torch.library.impl("rbln_custom_ops::rbln_cache_update", "cpu")
    def rbln_cache_update_cpu(cache, state, position, axis):
        # 'rbln_cache_update' is an in-place operation that isn't tracked in JIT trace, so a dummy output was added to the return value.
        return torch.empty([256])

    @register_fake("rbln_custom_ops::rbln_cache_update")
    def rbln_cache_update_abstract(cache, state, position, axis):
        return torch.empty([256])
