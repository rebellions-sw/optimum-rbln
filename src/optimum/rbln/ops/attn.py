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
def register_rbln_custom_attention():
    torch.library.define(
        "rbln_custom_ops::attn_decode",
        "(Tensor x, Tensor y, Tensor z, Tensor w, Tensor a, Tensor b, Tensor c, Tensor d) -> Tensor[]",
    )

    @torch.library.impl("rbln_custom_ops::attn_decode", "cpu")
    def attn_decode_cpu(q, k, v, mask, kcache, vcache, seq, scale):
        """Defines the computation pattern for fused attention with KV cache updates.

        IMPORTANT: This op serves as a pattern definition for the RBLN compiler to generate
        a single optimized NPU operation. It is NOT meant for CPU execution.

        Pattern components that compiler fuses into a single op:
        1. KV cache updates with new key/value states
        2. Scaled dot-product attention computation
        3. Masked softmax operation
        4. Final attention output computation

        Expected tensor shapes:
        - q: [batch=1, n_heads, n_groups, 1, head_dim] - Query states for single token
        - k: [batch=1, n_heads, 1, 1, head_dim] - Key states for current input
        - v: [batch=1, n_heads, 1, 1, head_dim] - Value states for current input
        - mask: [batch=1, n_heads, 1, 1, max_seq_len] - Attention mask
        - kcache: [batch_size, n_heads, 1, max_seq_len, head_dim] - Key cache
        - vcache: [batch_size, n_heads, 1, max_seq_len, head_dim] - Value cache
        - seq: [1] - Current sequence position
        - scale: [] - Attention scale factor

        Returns:
            Tuple[Tensor, Tensor, Tensor]:
            - attn_output: [batch=1, n_heads, n_groups, 1, head_dim] - Attention output
            - kcache: Same shape as input kcache, batch=1 - Placeholder for compiler
            - vcache: Same shape as input vcache, batch=1 - Placeholder for compiler
        """
        return (
            q,
            torch.empty(1, *kcache.shape[1:], device=kcache.device),
            torch.empty(1, *vcache.shape[1:], device=vcache.device),
        )

    @register_fake("rbln_custom_ops::attn_decode")
    def attn_decode_abstract(q, k, v, m, kcache, vcache, seq, partition):
        return (
            q,
            torch.empty(1, *kcache.shape[1:], device=kcache.device),
            torch.empty(1, *vcache.shape[1:], device=vcache.device),
        )

    torch.library.define(
        "rbln_custom_ops::attn_prefill",
        "(Tensor x, Tensor y, Tensor z, Tensor w, Tensor a, Tensor b, Tensor c, Tensor d, Tensor e) -> Tensor[]",
    )

    @torch.library.impl("rbln_custom_ops::attn_prefill", "cpu")
    def attn_prefill_cpu(q, k, v, mask, kcache, vcache, batch, seq, scale):
        """Defines the computation pattern for prefill phase attention with KV cache updates.

        IMPORTANT: This op serves as a pattern definition for the RBLN compiler to generate
        a single optimized NPU operation. It is NOT meant for CPU execution.

        Key differences from decode pattern:
        - Handles prefill phase with multiple input tokens
        - Takes explicit batch index for continuous batching

        Expected tensor shapes:
        - q: [batch=1, n_heads, n_groups, seq_len, head_dim] - Query states for multiple tokens
        - k: [batch=1, n_heads, 1, seq_len, head_dim] - Key states for current input
        - v: [batch=1, n_heads, 1, seq_len, head_dim] - Value states for current input
        - mask: [batch=1, 1, 1, seq_len, max_seq_len] - Attention mask
        - kcache: [batch_size, n_heads, 1, max_seq_len, head_dim] - Key cache
        - vcache: [batch_size, n_heads, 1, max_seq_len, head_dim] - Value cache
        - batch: [1] - Batch index for cache access
        - seq: [1] - Starting sequence position
        - scale: [] - Attention scale factor

        Returns:
            Tuple[Tensor, Tensor, Tensor]:
            - attn_output: [batch=1, n_heads, n_groups, seq_len, head_dim] - Attention output
            - empty_kcache: Same shape as input kcache - Placeholder for compiler
            - empty_vcache: Same shape as input vcache - Placeholder for compiler
        """
        return q, kcache, vcache

    @register_fake("rbln_custom_ops::attn_prefill")
    def attn_prefill_abstract(q, k, v, m, kcache, vcache, batch, seq, partition):
        return q, kcache, vcache


@lru_cache
def register_rbln_custom_attention_add_softmax():
    torch.library.define(
        "rbln_custom_ops::attn_decode_add_softmax",
        "(Tensor x, Tensor y, Tensor z, Tensor w, Tensor a, Tensor b, Tensor c, Tensor d) -> Tensor[]",
    )

    @torch.library.impl("rbln_custom_ops::attn_decode_add_softmax", "cpu")
    def attn_decode_add_softmax_cpu(q, k, v, mask, kcache, vcache, seq, scale):
        """Defines the computation pattern for fused attention with KV cache updates.

        IMPORTANT: This op serves as a pattern definition for the RBLN compiler to generate
        a single optimized NPU operation. It is NOT meant for CPU execution.

        Pattern components that compiler fuses into a single op:
        1. KV cache updates with new key/value states
        2. Scaled dot-product attention computation
        3. add-softmax operation
        4. Final attention output computation

        Expected tensor shapes:
        - q: [batch=1, n_heads, n_groups, 1, head_dim] - Query states for single token
        - k: [batch=1, n_heads, 1, 1, head_dim] - Key states for current input
        - v: [batch=1, n_heads, 1, 1, head_dim] - Value states for current input
        - mask: [batch=1, n_heads, 1, 1, max_seq_len] - Attention mask
        - kcache: [batch_size, n_heads, 1, max_seq_len, head_dim] - Key cache
        - vcache: [batch_size, n_heads, 1, max_seq_len, head_dim] - Value cache
        - seq: [1] - Current sequence position
        - scale: [] - Attention scale factor

        Returns:
            Tuple[Tensor, Tensor, Tensor]:
            - attn_output: [batch=1, n_heads, 1, 1, head_dim] - Attention output
            - kcache: Same shape as input kcache, batch=1 - Placeholder for compiler
            - vcache: Same shape as input vcache, batch=1 - Placeholder for compiler
        """
        return (
            q,
            torch.empty(1, *kcache.shape[1:], device=kcache.device),
            torch.empty(1, *vcache.shape[1:], device=vcache.device),
        )

    @register_fake("rbln_custom_ops::attn_decode_add_softmax")
    def attn_decode_add_softmax_abstract(q, k, v, m, kcache, vcache, seq, partition):
        return (
            q,
            torch.empty(1, *kcache.shape[1:], device=kcache.device),
            torch.empty(1, *vcache.shape[1:], device=vcache.device),
        )

    torch.library.define(
        "rbln_custom_ops::attn_prefill_add_softmax",
        "(Tensor x, Tensor y, Tensor z, Tensor w, Tensor a, Tensor b, Tensor c, Tensor d, Tensor e) -> Tensor[]",
    )

    @torch.library.impl("rbln_custom_ops::attn_prefill_add_softmax", "cpu")
    def attn_prefill_add_softmax_cpu(q, k, v, mask, kcache, vcache, batch, seq, scale):
        """Defines the computation pattern for prefill phase attention with KV cache updates.

        IMPORTANT: This op serves as a pattern definition for the RBLN compiler to generate
        a single optimized NPU operation. It is NOT meant for CPU execution.

        Key differences from decode pattern:
        - Handles prefill phase with multiple input tokens
        - Takes explicit batch index for continuous batching

        Expected tensor shapes:
        - q: [batch=1, n_heads, n_groups, seq_len, head_dim] - Query states for multiple tokens
        - k: [batch=1, n_heads, 1, seq_len, head_dim] - Key states for current input
        - v: [batch=1, n_heads, 1, seq_len, head_dim] - Value states for current input
        - mask: [batch=1, 1, 1, seq_len, max_seq_len] - Attention mask
        - kcache: [batch_size, n_heads, 1, max_seq_len, head_dim] - Key cache
        - vcache: [batch_size, n_heads, 1, max_seq_len, head_dim] - Value cache
        - batch: [1] - Batch index for cache access
        - seq: [1] - Starting sequence position
        - scale: [] - Attention scale factor

        Returns:
            Tuple[Tensor, Tensor, Tensor]:
            - attn_output: [batch=1, n_heads, seq_len, 1, head_dim] - Attention output
            - empty_kcache: Same shape as input kcache - Placeholder for compiler
            - empty_vcache: Same shape as input vcache - Placeholder for compiler
        """
        return (
            q,
            torch.empty(1, *kcache.shape[1:], device=kcache.device),
            torch.empty(1, *vcache.shape[1:], device=vcache.device),
        )

    @register_fake("rbln_custom_ops::attn_prefill_add_softmax")
    def attn_prefill_add_softmax_abstract(q, k, v, m, kcache, vcache, batch, seq, partition):
        return (
            q,
            torch.empty(1, *kcache.shape[1:], device=kcache.device),
            torch.empty(1, *vcache.shape[1:], device=vcache.device),
        )
