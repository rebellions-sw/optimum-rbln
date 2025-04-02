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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import torch
from transformers.utils import ModelOutput

from ....utils.logging import get_logger


logger = get_logger()

if TYPE_CHECKING:
    pass


@dataclass
class RBLNDecoderOnlyOutput(ModelOutput):
    logits: torch.FloatTensor = None
    generate_idx: torch.Tensor = None


class DecoderOnlyGenerationUtils:
    """
    A utility class that overrides methods required for the `generate()` method of Hugging Face's Transformers library
    when using an Optimum-RBLN compiled model. This class facilitates text generation by adapting the compiled model's
    behavior to work seamlessly with the Transformers generation API, supporting both prefill and decode phases.

    The class handles input preparation, model execution, and cache management for decoder-only causal language models
    optimized with the RBLN framework.
    """

    def get_decoder(self):
        return self.decoder

    def can_generate(self):
        return True

    def _reorder_cache(self, past_key_values, beam_idx):
        raise NotImplementedError

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        generate_idx: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        model_inputs = {}
        is_prefill_phase = generate_idx is None

        if is_prefill_phase:
            generate_idx = attention_mask.sum(dim=-1, keepdim=True).int()
            cache_position = None
        else:
            if inputs_embeds is not None:
                raise NotImplementedError("Specifying inputs_embeds in decoder phase is not supported.")

            input_ids = input_ids[:, -1:]
            cache_position = generate_idx
            generate_idx = generate_idx + 1
            model_inputs.update({"input_ids": input_ids})

        if inputs_embeds is not None:
            if self.rbln_config.model_cfg["use_inputs_embeds"]:
                model_inputs.update({"inputs_embeds": inputs_embeds})
            else:
                raise ValueError(
                    "The specifying inputs_embedst is only supported when using a compiled RBLN model with 'rbln_use_inputs_embeds' set to True."
                )
        else:
            model_inputs.update({"input_ids": input_ids})

        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "generate_idx": generate_idx,
            }
        )

        return model_inputs

    def _update_model_kwargs_for_generation(
        self,
        outputs: RBLNDecoderOnlyOutput,
        model_kwargs: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, Any]:
        # update generate_idx
        model_kwargs["generate_idx"] = outputs.generate_idx

        return model_kwargs

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        generate_idx: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor]:
        """
        Forward method for the RBLN-optimized model, designed for integration with the HuggingFace generate API.
        For continuous batching, the prefill stage processes one batch at a time and updates the KV cache using batch_idx.
        A for-loop ensures synchronization with the HuggingFace generate API.
        The decoder stage operates as usual, processing inputs in batch mode.
        """
        # Prefll
        if cache_position is None:
            logits = []
            inputs = inputs_embeds if inputs_embeds is not None else input_ids
            batch_size = inputs.shape[0]

            for b_idx in range(batch_size):
                cache_position = torch.arange(0, generate_idx[b_idx].item(), dtype=torch.int32).unsqueeze(0)
                logit = self.prefill_decoder(
                    input_ids=inputs[b_idx : b_idx + 1] if inputs_embeds is None else None,
                    inputs_embeds=inputs[b_idx : b_idx + 1] if inputs_embeds is not None else None,
                    attention_mask=attention_mask[b_idx] if attention_mask is not None else None,
                    cache_position=cache_position,
                    batch_idx=b_idx,
                )
                logits.append(logit)

            logits = torch.cat(logits, dim=0)
        # Decoder
        else:
            logits = self.decoder(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
            )

        return RBLNDecoderOnlyOutput(
            logits=logits,
            generate_idx=generate_idx,
        )
