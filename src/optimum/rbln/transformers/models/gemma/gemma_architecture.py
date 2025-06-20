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
    def convert_to_rbln_causal_lm(self, causal_lm: "GemmaForCausalLM", max_seq_len: int):
        new_layers = []
        for layer in causal_lm.model.layers:
            if self.attn_impl == "eager":
                new_self_attn = DecoderOnlyAttention(
                    layer.self_attn,
                    self.use_attention_mask,
                    kvcache_block_size=self.kvcache_block_size,
                    use_position_ids=self.use_position_ids,
                )
            elif self.attn_impl == "flash_attn":
                new_self_attn = DecoderOnlyFlashAttention(
                    layer.self_attn,
                    kvcache_partition_len=self.kvcache_partition_len,
                    use_attention_mask=self.use_attention_mask,
                    kvcache_block_size=self.kvcache_block_size,
                    use_position_ids=self.use_position_ids,
                )
            else:
                raise NotImplementedError(f"Unknwon attn : {self.attn_impl}")
            new_layer = DecoderOnlyLayer(layer, new_self_attn)
            new_layers.append(new_layer)
        new_model = GemmaModel(
            causal_lm.model,
            new_layers,
            partition_len=self.kvcache_partition_len,
            max_seq_len=max_seq_len,
            sliding_window_layers=self.sliding_window_layers,
        )
        new_causal_lm = DecoderOnlyForCausalLM(causal_lm, new_model)
        return new_causal_lm


class GemmaModel(DecoderOnlyModel):
    @property
    def hidden_multiplier(self):
        return self._original_mod.config.hidden_size**0.5
