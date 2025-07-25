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
    "models": [
        "RBLNASTForAudioClassification",
        "RBLNASTForAudioClassificationConfig",
        "RBLNAutoModel",
        "RBLNAutoModelForAudioClassification",
        "RBLNAutoModelForCausalLM",
        "RBLNAutoModelForCTC",
        "RBLNAutoModelForDepthEstimation",
        "RBLNAutoModelForImageClassification",
        "RBLNAutoModelForImageTextToText",
        "RBLNAutoModelForMaskedLM",
        "RBLNAutoModelForQuestionAnswering",
        "RBLNAutoModelForSeq2SeqLM",
        "RBLNAutoModelForSequenceClassification",
        "RBLNAutoModelForSpeechSeq2Seq",
        "RBLNAutoModelForVision2Seq",
        "RBLNBartForConditionalGeneration",
        "RBLNBartForConditionalGenerationConfig",
        "RBLNBartModel",
        "RBLNBartModelConfig",
        "RBLNBertForMaskedLM",
        "RBLNBertForMaskedLMConfig",
        "RBLNBertForQuestionAnswering",
        "RBLNBertForQuestionAnsweringConfig",
        "RBLNBertModel",
        "RBLNBertModelConfig",
        "RBLNBlip2ForConditionalGeneration",
        "RBLNBlip2ForConditionalGenerationConfig",
        "RBLNBlip2QFormerModel",
        "RBLNBlip2QFormerModelConfig",
        "RBLNBlip2VisionModel",
        "RBLNBlip2VisionModelConfig",
        "RBLNColPaliForRetrieval",
        "RBLNColPaliForRetrievalConfig",
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
        "RBLNDecoderOnlyModelConfig",
        "RBLNDecoderOnlyModel",
        "RBLNDistilBertForQuestionAnswering",
        "RBLNDistilBertForQuestionAnsweringConfig",
        "RBLNDPTForDepthEstimation",
        "RBLNDPTForDepthEstimationConfig",
        "RBLNExaoneForCausalLM",
        "RBLNExaoneForCausalLMConfig",
        "RBLNGemmaModel",
        "RBLNGemmaModelConfig",
        "RBLNGemma3ForCausalLM",
        "RBLNGemma3ForCausalLMConfig",
        "RBLNGemma3ForConditionalGeneration",
        "RBLNGemma3ForConditionalGenerationConfig",
        "RBLNGemmaForCausalLM",
        "RBLNGemmaForCausalLMConfig",
        "RBLNGPT2LMHeadModel",
        "RBLNGPT2LMHeadModelConfig",
        "RBLNGPT2Model",
        "RBLNGPT2ModelConfig",
        "RBLNIdefics3ForConditionalGeneration",
        "RBLNIdefics3ForConditionalGenerationConfig",
        "RBLNIdefics3VisionTransformer",
        "RBLNIdefics3VisionTransformerConfig",
        "RBLNLlamaForCausalLM",
        "RBLNLlamaForCausalLMConfig",
        "RBLNLlavaForConditionalGeneration",
        "RBLNLlavaForConditionalGenerationConfig",
        "RBLNLlamaModel",
        "RBLNLlamaModelConfig",
        "RBLNOPTForCausalLM",
        "RBLNOPTForCausalLMConfig",
        "RBLNPegasusForConditionalGeneration",
        "RBLNPegasusForConditionalGenerationConfig",
        "RBLNPegasusModel",
        "RBLNPegasusModelConfig",
        "RBLNLlavaNextForConditionalGeneration",
        "RBLNLlavaNextForConditionalGenerationConfig",
        "RBLNMidmLMHeadModel",
        "RBLNMidmLMHeadModelConfig",
        "RBLNMistralForCausalLM",
        "RBLNMistralForCausalLMConfig",
        "RBLNMistralModel",
        "RBLNMistralModelConfig",
        "RBLNOPTForCausalLM",
        "RBLNOPTForCausalLMConfig",
        "RBLNOPTModel",
        "RBLNOPTModelConfig",
        "RBLNPhiForCausalLM",
        "RBLNPhiForCausalLMConfig",
        "RBLNPixtralVisionModelConfig",
        "RBLNPixtralVisionModel",
        "RBLNPhiModel",
        "RBLNPhiModelConfig",
        "RBLNQwen2_5_VisionTransformerPretrainedModel",
        "RBLNQwen2_5_VisionTransformerPretrainedModelConfig",
        "RBLNQwen2_5_VLForConditionalGeneration",
        "RBLNQwen2_5_VLForConditionalGenerationConfig",
        "RBLNQwen2Model",
        "RBLNQwen2ModelConfig",
        "RBLNQwen2ForCausalLM",
        "RBLNQwen2ForCausalLMConfig",
        "RBLNQwen3ForCausalLM",
        "RBLNQwen3ForCausalLMConfig",
        "RBLNQwen3Model",
        "RBLNQwen3ModelConfig",
        "RBLNResNetForImageClassification",
        "RBLNResNetForImageClassificationConfig",
        "RBLNRobertaForMaskedLM",
        "RBLNRobertaForMaskedLMConfig",
        "RBLNRobertaForSequenceClassification",
        "RBLNRobertaForSequenceClassificationConfig",
        "RBLNSiglipVisionModel",
        "RBLNSiglipVisionModelConfig",
        "RBLNT5EncoderModel",
        "RBLNT5EncoderModelConfig",
        "RBLNT5ForConditionalGeneration",
        "RBLNT5ForConditionalGenerationConfig",
        "RBLNTimeSeriesTransformerForPrediction",
        "RBLNTimeSeriesTransformerForPredictionConfig",
        "RBLNViTForImageClassification",
        "RBLNViTForImageClassificationConfig",
        "RBLNWav2Vec2ForCTC",
        "RBLNWav2Vec2ForCTCConfig",
        "RBLNWhisperForConditionalGeneration",
        "RBLNWhisperForConditionalGenerationConfig",
        "RBLNXLMRobertaForSequenceClassification",
        "RBLNXLMRobertaForSequenceClassificationConfig",
        "RBLNXLMRobertaModel",
        "RBLNXLMRobertaModelConfig",
    ],
}

if TYPE_CHECKING:
    from .models import (
        RBLNASTForAudioClassification,
        RBLNASTForAudioClassificationConfig,
        RBLNAutoModel,
        RBLNAutoModelForAudioClassification,
        RBLNAutoModelForCausalLM,
        RBLNAutoModelForCTC,
        RBLNAutoModelForDepthEstimation,
        RBLNAutoModelForImageClassification,
        RBLNAutoModelForImageTextToText,
        RBLNAutoModelForMaskedLM,
        RBLNAutoModelForQuestionAnswering,
        RBLNAutoModelForSeq2SeqLM,
        RBLNAutoModelForSequenceClassification,
        RBLNAutoModelForSpeechSeq2Seq,
        RBLNAutoModelForVision2Seq,
        RBLNBartForConditionalGeneration,
        RBLNBartForConditionalGenerationConfig,
        RBLNBartModel,
        RBLNBartModelConfig,
        RBLNBertForMaskedLM,
        RBLNBertForMaskedLMConfig,
        RBLNBertForQuestionAnswering,
        RBLNBertForQuestionAnsweringConfig,
        RBLNBertModel,
        RBLNBertModelConfig,
        RBLNBlip2ForConditionalGeneration,
        RBLNBlip2ForConditionalGenerationConfig,
        RBLNBlip2QFormerModel,
        RBLNBlip2QFormerModelConfig,
        RBLNBlip2VisionModel,
        RBLNBlip2VisionModelConfig,
        RBLNCLIPTextModel,
        RBLNCLIPTextModelConfig,
        RBLNCLIPTextModelWithProjection,
        RBLNCLIPTextModelWithProjectionConfig,
        RBLNCLIPVisionModel,
        RBLNCLIPVisionModelConfig,
        RBLNCLIPVisionModelWithProjection,
        RBLNCLIPVisionModelWithProjectionConfig,
        RBLNColPaliForRetrieval,
        RBLNColPaliForRetrievalConfig,
        RBLNDecoderOnlyModel,
        RBLNDecoderOnlyModelConfig,
        RBLNDecoderOnlyModelForCausalLM,
        RBLNDecoderOnlyModelForCausalLMConfig,
        RBLNDistilBertForQuestionAnswering,
        RBLNDistilBertForQuestionAnsweringConfig,
        RBLNDPTForDepthEstimation,
        RBLNDPTForDepthEstimationConfig,
        RBLNExaoneForCausalLM,
        RBLNExaoneForCausalLMConfig,
        RBLNGemma3ForCausalLM,
        RBLNGemma3ForCausalLMConfig,
        RBLNGemma3ForConditionalGeneration,
        RBLNGemma3ForConditionalGenerationConfig,
        RBLNGemmaForCausalLM,
        RBLNGemmaForCausalLMConfig,
        RBLNGemmaModel,
        RBLNGemmaModelConfig,
        RBLNGPT2LMHeadModel,
        RBLNGPT2LMHeadModelConfig,
        RBLNGPT2Model,
        RBLNGPT2ModelConfig,
        RBLNIdefics3ForConditionalGeneration,
        RBLNIdefics3ForConditionalGenerationConfig,
        RBLNIdefics3VisionTransformer,
        RBLNIdefics3VisionTransformerConfig,
        RBLNLlamaForCausalLM,
        RBLNLlamaForCausalLMConfig,
        RBLNLlamaModel,
        RBLNLlamaModelConfig,
        RBLNLlavaForConditionalGeneration,
        RBLNLlavaForConditionalGenerationConfig,
        RBLNLlavaNextForConditionalGeneration,
        RBLNLlavaNextForConditionalGenerationConfig,
        RBLNMidmLMHeadModel,
        RBLNMidmLMHeadModelConfig,
        RBLNMistralForCausalLM,
        RBLNMistralForCausalLMConfig,
        RBLNMistralModel,
        RBLNMistralModelConfig,
        RBLNOPTForCausalLM,
        RBLNOPTForCausalLMConfig,
        RBLNOPTModel,
        RBLNOPTModelConfig,
        RBLNPegasusForConditionalGeneration,
        RBLNPegasusForConditionalGenerationConfig,
        RBLNPegasusModel,
        RBLNPegasusModelConfig,
        RBLNPhiForCausalLM,
        RBLNPhiForCausalLMConfig,
        RBLNPhiModel,
        RBLNPhiModelConfig,
        RBLNPixtralVisionModel,
        RBLNPixtralVisionModelConfig,
        RBLNQwen2_5_VisionTransformerPretrainedModel,
        RBLNQwen2_5_VisionTransformerPretrainedModelConfig,
        RBLNQwen2_5_VLForConditionalGeneration,
        RBLNQwen2_5_VLForConditionalGenerationConfig,
        RBLNQwen2ForCausalLM,
        RBLNQwen2ForCausalLMConfig,
        RBLNQwen2Model,
        RBLNQwen2ModelConfig,
        RBLNQwen3ForCausalLM,
        RBLNQwen3ForCausalLMConfig,
        RBLNQwen3Model,
        RBLNQwen3ModelConfig,
        RBLNResNetForImageClassification,
        RBLNResNetForImageClassificationConfig,
        RBLNRobertaForMaskedLM,
        RBLNRobertaForMaskedLMConfig,
        RBLNRobertaForSequenceClassification,
        RBLNRobertaForSequenceClassificationConfig,
        RBLNSiglipVisionModel,
        RBLNSiglipVisionModelConfig,
        RBLNT5EncoderModel,
        RBLNT5EncoderModelConfig,
        RBLNT5ForConditionalGeneration,
        RBLNT5ForConditionalGenerationConfig,
        RBLNTimeSeriesTransformerForPrediction,
        RBLNTimeSeriesTransformerForPredictionConfig,
        RBLNViTForImageClassification,
        RBLNViTForImageClassificationConfig,
        RBLNWav2Vec2ForCTC,
        RBLNWav2Vec2ForCTCConfig,
        RBLNWhisperForConditionalGeneration,
        RBLNWhisperForConditionalGenerationConfig,
        RBLNXLMRobertaForSequenceClassification,
        RBLNXLMRobertaForSequenceClassificationConfig,
        RBLNXLMRobertaModel,
        RBLNXLMRobertaModelConfig,
    )
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
