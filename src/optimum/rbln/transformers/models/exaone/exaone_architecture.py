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

from typing import TYPE_CHECKING

import torch.nn as nn

from ....utils import logging
from ...models.decoderonly.decoderonly_architecture import (
    DecoderOnlyAttention,
    DecoderOnlyFlashAttention,
    DecoderOnlyForCausalLM,
    DecoderOnlyLayer,
    DecoderOnlyModel,
    DecoderOnlyWrapper,
)


if TYPE_CHECKING:
    from transformers import PreTrainedModel as ExaoneForCausalLM

logger = logging.get_logger(__name__)


class ExaoneForCausalLMWrapper(DecoderOnlyWrapper):
    """A wrapper class for the Exaone model with a language modeling head."""

    def convert_to_rbln_causal_lm(self, causal_lm: "ExaoneForCausalLM", max_seq_len: int):
        new_layers = []
        for layer in causal_lm.transformer.h:
            if self.attn_impl == "eager":
                new_self_attn = ExaoneAttention(
                    layer.attn.attention,
                    self.use_attention_mask,
                    kvcache_block_size=self.kvcache_block_size,
                    use_position_ids=self.use_position_ids,
                )
            elif self.attn_impl == "flash_attn":
                new_self_attn = ExaoneFlashAttention(
                    layer.attn.attention,
                    kvcache_partition_len=self.kvcache_partition_len,
                    use_attention_mask=self.use_attention_mask,
                    kvcache_block_size=self.kvcache_block_size,
                    use_position_ids=self.use_position_ids,
                )
            else:
                raise NotImplementedError(f"Unknwon attn : {self.attn_impl}")

            new_layer = ExaoneLayer(layer, new_self_attn)
            new_layers.append(new_layer)
        new_model = ExaoneModel(
            causal_lm.transformer,
            new_layers,
            partition_len=self.kvcache_partition_len,
            max_seq_len=max_seq_len,
            sliding_window_layers=self.sliding_window_layers,
        )
        new_causal_lm = DecoderOnlyForCausalLM(causal_lm, new_model)
        return new_causal_lm


class ExaoneModel(DecoderOnlyModel):
    def get_embedding(self) -> nn.Embedding:
        return self._original_mod.wte

    def get_last_layernorm(self) -> nn.LayerNorm:
        return self._original_mod.ln_f


class ExaoneLayer(DecoderOnlyLayer):
    def get_pre_attention_layernorm(self) -> nn.LayerNorm:
        return self._original_mod.ln_1

    def get_post_attention_layernorm(self) -> nn.LayerNorm:
        return self._original_mod.ln_2


class ExaoneAttention(DecoderOnlyAttention):
    def __post_init__(self):
        self.q_proj = self._original_mod.q_proj
        self.k_proj = self._original_mod.k_proj
        self.v_proj = self._original_mod.v_proj
        self.o_proj = self._original_mod.out_proj


class ExaoneFlashAttention(DecoderOnlyFlashAttention):
    def __post_init__(self):
        self.q_proj = self._original_mod.q_proj
        self.k_proj = self._original_mod.k_proj
        self.v_proj = self._original_mod.v_proj
        self.o_proj = self._original_mod.out_proj
