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

from transformers.utils import _LazyModule


_import_structure = {
    "cache_utils": ["RebelDynamicCache"],
    "models": [
        "RBLNAutoModel",
        "RBLNAutoModelForAudioClassification",
        "RBLNAutoModelForCausalLM",
        "RBLNAutoModelForCTC",
        "RBLNAutoModelForDepthEstimation",
        "RBLNAutoModelForImageClassification",
        "RBLNAutoModelForMaskedLM",
        "RBLNAutoModelForQuestionAnswering",
        "RBLNAutoModelForSeq2SeqLM",
        "RBLNAutoModelForSequenceClassification",
        "RBLNAutoModelForSpeechSeq2Seq",
        "RBLNAutoModelForVision2Seq",
        "RBLNBartForConditionalGeneration",
        "RBLNBartModel",
        "RBLNBertModel",
        "RBLNCLIPTextModel",
        "RBLNCLIPTextModelWithProjection",
        "RBLNCLIPVisionModel",
        "RBLNDPTForDepthEstimation",
        "RBLNExaoneForCausalLM",
        "RBLNGemmaForCausalLM",
        "RBLNGPT2LMHeadModel",
        "RBLNQwen2ForCausalLM",
        "RBLNWav2Vec2ForCTC",
        "RBLNWhisperForConditionalGeneration",
        "RBLNLlamaForCausalLM",
        "RBLNPhiForCausalLM",
        "RBLNT5EncoderModel",
        "RBLNT5ForConditionalGeneration",
        "RBLNLlavaNextForConditionalGeneration",
        "RBLNMidmLMHeadModel",
        "RBLNXLMRobertaModel",
        "RBLNMistralForCausalLM",
    ],
    "modeling_alias": [
        "RBLNASTForAudioClassification",
        "RBLNBertForQuestionAnswering",
        "RBLNDistilBertForQuestionAnswering",
        "RBLNResNetForImageClassification",
        "RBLNXLMRobertaForSequenceClassification",
        "RBLNRobertaForSequenceClassification",
        "RBLNRobertaForMaskedLM",
        "RBLNViTForImageClassification",
    ],
}

if TYPE_CHECKING:
    from .cache_utils import RebelDynamicCache
    from .modeling_alias import (
        RBLNASTForAudioClassification,
        RBLNBertForQuestionAnswering,
        RBLNDistilBertForQuestionAnswering,
        RBLNResNetForImageClassification,
        RBLNRobertaForMaskedLM,
        RBLNRobertaForSequenceClassification,
        RBLNViTForImageClassification,
        RBLNXLMRobertaForSequenceClassification,
    )
    from .models import (
        RBLNAutoModel,
        RBLNAutoModelForAudioClassification,
        RBLNAutoModelForCausalLM,
        RBLNAutoModelForCTC,
        RBLNAutoModelForDepthEstimation,
        RBLNAutoModelForImageClassification,
        RBLNAutoModelForMaskedLM,
        RBLNAutoModelForQuestionAnswering,
        RBLNAutoModelForSeq2SeqLM,
        RBLNAutoModelForSequenceClassification,
        RBLNAutoModelForSpeechSeq2Seq,
        RBLNAutoModelForVision2Seq,
        RBLNBartForConditionalGeneration,
        RBLNBartModel,
        RBLNBertModel,
        RBLNCLIPTextModel,
        RBLNCLIPTextModelWithProjection,
        RBLNCLIPVisionModel,
        RBLNDPTForDepthEstimation,
        RBLNExaoneForCausalLM,
        RBLNGemmaForCausalLM,
        RBLNGPT2LMHeadModel,
        RBLNLlamaForCausalLM,
        RBLNLlavaNextForConditionalGeneration,
        RBLNMidmLMHeadModel,
        RBLNMistralForCausalLM,
        RBLNPhiForCausalLM,
        RBLNQwen2ForCausalLM,
        RBLNT5EncoderModel,
        RBLNT5ForConditionalGeneration,
        RBLNWav2Vec2ForCTC,
        RBLNWhisperForConditionalGeneration,
        RBLNXLMRobertaModel,
    )
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
