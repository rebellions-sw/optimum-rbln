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
from packaging import version


if version.parse(torch.__version__) > version.parse("2.4.0"):
    register_fake = torch.library.register_fake
else:
    register_fake = torch.library.impl_abstract


@lru_cache
def register_rbln_custom_flash_attention():
    torch.library.define(
        "rbln_custom_ops::flash_attn_decode",
        "(Tensor x, Tensor y, Tensor z, Tensor w, Tensor a, Tensor b, Tensor c, Tensor d, int e) -> Tensor[]",
    )

    @torch.library.impl("rbln_custom_ops::flash_attn_decode", "cpu")
    def flash_attn_decode_cpu(q, k, v, mask, kcache, vcache, seq, scale, partition):
        return (
            q,
            torch.empty(1, *kcache.shape[1:], device=kcache.device),
            torch.empty(1, *vcache.shape[1:], device=vcache.device),
        )

    @register_fake("rbln_custom_ops::flash_attn_decode")
    def flash_attn_decode_abstract(q, k, v, m, kcache, vcache, seq, scale, partition):
        return (
            q,
            torch.empty(1, *kcache.shape[1:], device=kcache.device),
            torch.empty(1, *vcache.shape[1:], device=vcache.device),
        )

    torch.library.define(
        "rbln_custom_ops::flash_attn_prefill",
        "(Tensor x, Tensor y, Tensor z, Tensor w, Tensor a, Tensor b, Tensor c, Tensor d, Tensor e, int f) -> Tensor[]",
    )

    @torch.library.impl("rbln_custom_ops::flash_attn_prefill", "cpu")
    def flash_attn_prefill_cpu(q, k, v, mask, kcache, vcache, batch, seq, scale, partition):
        return q, kcache, vcache

    @register_fake("rbln_custom_ops::flash_attn_prefill")
    def flash_attn_prefill_abstract(q, k, v, m, kcache, vcache, batch, seq, scale, partition):
        return q, kcache, vcache
