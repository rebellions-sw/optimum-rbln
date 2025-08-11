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

import inspect
from typing import Any, Callable

from transformers import BartForConditionalGeneration, PreTrainedModel

from ....utils.logging import get_logger
from ...modeling_generic import RBLNTransformerEncoderForFeatureExtraction
from ...models.seq2seq import RBLNModelForSeq2SeqLM
from .bart_architecture import BartWrapper
from .configuration_bart import RBLNBartForConditionalGenerationConfig


logger = get_logger()


class RBLNBartModel(RBLNTransformerEncoderForFeatureExtraction):
    """
    RBLN optimized BART model for feature extraction tasks.

    This class provides hardware-accelerated inference for BART encoder models
    on RBLN devices, optimized for feature extraction use cases.
    """


class RBLNBartForConditionalGeneration(RBLNModelForSeq2SeqLM):
    """
    RBLN optimized BART model for conditional text generation tasks.

    This class provides hardware-accelerated inference for BART models
    on RBLN devices, supporting sequence-to-sequence generation tasks
    such as summarization, translation, and text generation.
    """

    support_causal_attn = True

    @classmethod
    def wrap_model_if_needed(self, model: PreTrainedModel, rbln_config: RBLNBartForConditionalGenerationConfig):
        return BartWrapper(
            model, enc_max_seq_len=rbln_config.enc_max_seq_len, use_attention_mask=rbln_config.use_attention_mask
        )

    def __getattr__(self, __name: str) -> Any:
        def redirect(func):
            return lambda *pargs, **kwargs: func(self, *pargs, **kwargs)

        val = getattr(BartForConditionalGeneration, __name)

        if isinstance(val, Callable) and "self" in set(inspect.signature(val).parameters):
            return redirect(val)

        return val
