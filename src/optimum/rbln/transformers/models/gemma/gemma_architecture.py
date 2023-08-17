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

from typing import TYPE_CHECKING

from ...models.decoderonly.decoderonly_architecture import (
    DecoderOnlyAttention,
    DecoderOnlyFlashAttention,
    DecoderOnlyForCausalLM,
    DecoderOnlyLayer,
    DecoderOnlyModel,
    DecoderOnlyWrapper,
)


if TYPE_CHECKING:
    from transformers import GemmaForCausalLM


class GemmaWrapper(DecoderOnlyWrapper):
    def convert_to_rbln_causal_lm(self, causal_lm: "GemmaForCausalLM"):
        new_layers = []
        for layer in causal_lm.model.layers:
            if self.attn_impl == "eager":
                new_self_attn = DecoderOnlyAttention(layer.self_attn)
            elif self.attn_impl == "flash_attn":
                new_self_attn = DecoderOnlyFlashAttention(
                    layer.self_attn, kvcache_partition_len=self.kvcache_partition_len
                )
            else:
                raise NotImplementedError(f"Unknwon attn : {self.attn_impl}")
            new_layer = DecoderOnlyLayer(layer, new_self_attn)
            new_layers.append(new_layer)
        new_model = GemmaModel(causal_lm.model, new_layers, partition_len=self.kvcache_partition_len)
        new_causal_lm = DecoderOnlyForCausalLM(causal_lm, new_model)
        return new_causal_lm


class GemmaModel(DecoderOnlyModel):
    @property
    def hidden_multiplier(self):
        return self._original_mod.config.hidden_size**0.5
