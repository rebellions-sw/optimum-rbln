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

from typing import TYPE_CHECKING, Any, Dict, Optional

import torch
from transformers.generation.utils import GenerationMixin


if TYPE_CHECKING:
    from ...modeling_outputs import RBLNDecoderOnlyOutput


class RBLNDecoderOnlyGenerationMixin(GenerationMixin):
    _supports_cache_class = False  # Needed for GenerationMixin
    _is_stateful = False  # Needed for GenerationMixin

    def _reorder_cache(self, past_key_values, beam_idx):
        raise NotImplementedError

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        generate_idx: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        padded_cache_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        model_inputs = {}
        is_prefill_phase = generate_idx is None

        if is_prefill_phase:
            generate_idx = attention_mask.sum(dim=-1, keepdim=True).int()
            padded_cache_lengths = torch.zeros_like(generate_idx)
            cache_position = None
            position_ids = None
        else:
            if inputs_embeds is not None:
                # if `inputs_embeds` are passed, only use them in the 1st generation step for every prompt.
                inputs_embeds = None

            input_ids = input_ids[:, -1:]
            position_ids = generate_idx
            cache_position = generate_idx + padded_cache_lengths if padded_cache_lengths is not None else generate_idx
            generate_idx = generate_idx + 1
            model_inputs.update({"input_ids": input_ids})

        if inputs_embeds is not None:
            if self.rbln_config.use_inputs_embeds:
                model_inputs.update({"inputs_embeds": inputs_embeds})
            else:
                raise ValueError(
                    "The specifying inputs_embeds is only supported when using a compiled RBLN model with 'rbln_use_inputs_embeds' set to True."
                )
        else:
            model_inputs.update({"input_ids": input_ids})

        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "generate_idx": generate_idx,
                "position_ids": position_ids,
                "padded_cache_lengths": padded_cache_lengths,
            }
        )

        return model_inputs

    def _update_model_kwargs_for_generation(
        self, outputs: "RBLNDecoderOnlyOutput", model_kwargs: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        # update generate_idx
        model_kwargs["generate_idx"] = outputs.generate_idx
        model_kwargs["padded_cache_lengths"] = outputs.padded_cache_lengths
        return model_kwargs
