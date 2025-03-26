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

from transformers.utils import _LazyModule


_import_structure = {
    "configuration_alias": [
        "RBLNASTForAudioClassificationConfig",
        "RBLNDistilBertForQuestionAnsweringConfig",
        "RBLNResNetForImageClassificationConfig",
        "RBLNXLMRobertaForSequenceClassificationConfig",
        "RBLNRobertaForSequenceClassificationConfig",
        "RBLNRobertaForMaskedLMConfig",
        "RBLNViTForImageClassificationConfig",
    ],
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
        "RBLNBertForMaskedLM",
        "RBLNBertForQuestionAnswering",
        "RBLNBertModel",
        "RBLNCLIPTextModel",
        "RBLNCLIPTextModelConfig",
        "RBLNCLIPTextModelWithProjection",
        "RBLNCLIPTextModelWithProjectionConfig",
        "RBLNCLIPVisionModel",
        "RBLNCLIPVisionModelConfig",
        "RBLNCLIPVisionModelWithProjection",
        "RBLNCLIPVisionModelWithProjectionConfig",
        "RBLNDecoderOnlyModelForCausalLM",
        "RBLNDecoderOnlyModelForCausalLMConfig",
        "RBLNDPTForDepthEstimation",
        "RBLNExaoneForCausalLM",
        "RBLNGemmaForCausalLM",
        "RBLNGPT2LMHeadModel",
        "RBLNLlamaForCausalLM",
        "RBLNLlamaForCausalLMConfig",
        "RBLNLlavaNextForConditionalGeneration",
        "RBLNLlavaNextForConditionalGenerationConfig",
        "RBLNMidmLMHeadModel",
        "RBLNMistralForCausalLM",
        "RBLNMistralForCausalLMConfig",
        "RBLNPhiForCausalLM",
        "RBLNQwen2ForCausalLM",
        "RBLNT5EncoderModel",
        "RBLNT5ForConditionalGeneration",
        "RBLNWav2Vec2ForCTC",
        "RBLNWhisperForConditionalGeneration",
        "RBLNXLMRobertaModel",
    ],
    "modeling_alias": [
        "RBLNASTForAudioClassification",
        "RBLNDistilBertForQuestionAnswering",
        "RBLNResNetForImageClassification",
        "RBLNXLMRobertaForSequenceClassification",
        "RBLNRobertaForSequenceClassification",
        "RBLNRobertaForMaskedLM",
        "RBLNViTForImageClassification",
    ],
}

if TYPE_CHECKING:
    from .configuration_alias import (
        RBLNASTForAudioClassificationConfig,
        RBLNDistilBertForQuestionAnsweringConfig,
        RBLNResNetForImageClassificationConfig,
        RBLNRobertaForMaskedLMConfig,
        RBLNRobertaForSequenceClassificationConfig,
        RBLNViTForImageClassificationConfig,
        RBLNXLMRobertaForSequenceClassificationConfig,
    )
    from .modeling_alias import (
        RBLNASTForAudioClassification,
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
        RBLNBertForMaskedLM,
        RBLNBertForQuestionAnswering,
        RBLNBertModel,
        RBLNCLIPTextModel,
        RBLNCLIPTextModelConfig,
        RBLNCLIPTextModelWithProjection,
        RBLNCLIPTextModelWithProjectionConfig,
        RBLNCLIPVisionModel,
        RBLNCLIPVisionModelConfig,
        RBLNCLIPVisionModelWithProjection,
        RBLNCLIPVisionModelWithProjectionConfig,
        RBLNDecoderOnlyModelForCausalLM,
        RBLNDecoderOnlyModelForCausalLMConfig,
        RBLNDPTForDepthEstimation,
        RBLNExaoneForCausalLM,
        RBLNGemmaForCausalLM,
        RBLNGPT2LMHeadModel,
        RBLNLlamaForCausalLM,
        RBLNLlamaForCausalLMConfig,
        RBLNLlavaNextForConditionalGeneration,
        RBLNLlavaNextForConditionalGenerationConfig,
        RBLNMidmLMHeadModel,
        RBLNMistralForCausalLM,
        RBLNMistralForCausalLMConfig,
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
