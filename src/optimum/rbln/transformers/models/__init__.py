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
    "audio_spectrogram_transformer": [
        "RBLNASTForAudioClassification",
        "RBLNASTForAudioClassificationConfig",
    ],
    "auto": [
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
        "RBLNAutoModelForImageTextToText",
        "RBLNAutoModelForTextEncoding",
    ],
    "bart": [
        "RBLNBartForConditionalGeneration",
        "RBLNBartModel",
        "RBLNBartForConditionalGenerationConfig",
        "RBLNBartModelConfig",
    ],
    "bert": [
        "RBLNBertModel",
        "RBLNBertModelConfig",
        "RBLNBertForQuestionAnswering",
        "RBLNBertForQuestionAnsweringConfig",
        "RBLNBertForMaskedLM",
        "RBLNBertForMaskedLMConfig",
    ],
    "blip_2": [
        "RBLNBlip2VisionModelConfig",
        "RBLNBlip2VisionModel",
        "RBLNBlip2ForConditionalGeneration",
        "RBLNBlip2ForConditionalGenerationConfig",
        "RBLNBlip2QFormerModel",
        "RBLNBlip2QFormerModelConfig",
    ],
    "clip": [
        "RBLNCLIPTextModel",
        "RBLNCLIPTextModelConfig",
        "RBLNCLIPTextModelWithProjection",
        "RBLNCLIPTextModelWithProjectionConfig",
        "RBLNCLIPVisionModel",
        "RBLNCLIPVisionModelConfig",
        "RBLNCLIPVisionModelWithProjection",
        "RBLNCLIPVisionModelWithProjectionConfig",
    ],
    "colpali": [
        "RBLNColPaliForRetrieval",
        "RBLNColPaliForRetrievalConfig",
    ],
    "distilbert": [
        "RBLNDistilBertForQuestionAnswering",
        "RBLNDistilBertForQuestionAnsweringConfig",
    ],
    "qwen2_5_vl": [
        "RBLNQwen2_5_VisionTransformerPretrainedModel",
        "RBLNQwen2_5_VisionTransformerPretrainedModelConfig",
        "RBLNQwen2_5_VLForConditionalGeneration",
        "RBLNQwen2_5_VLForConditionalGenerationConfig",
    ],
    "decoderonly": [
        "RBLNDecoderOnlyModelConfig",
        "RBLNDecoderOnlyModel",
        "RBLNDecoderOnlyModelForCausalLM",
        "RBLNDecoderOnlyModelForCausalLMConfig",
    ],
    "depth_anything": ["RBLNDepthAnythingForDepthEstimationConfig", "RBLNDepthAnythingForDepthEstimation"],
    "dpt": [
        "RBLNDPTForDepthEstimation",
        "RBLNDPTForDepthEstimationConfig",
    ],
    "exaone": ["RBLNExaoneForCausalLM", "RBLNExaoneForCausalLMConfig"],
    "gemma": ["RBLNGemmaForCausalLM", "RBLNGemmaForCausalLMConfig", "RBLNGemmaModel", "RBLNGemmaModelConfig"],
    "gemma3": [
        "RBLNGemma3ForCausalLM",
        "RBLNGemma3ForCausalLMConfig",
        "RBLNGemma3ForConditionalGeneration",
        "RBLNGemma3ForConditionalGenerationConfig",
    ],
    "gpt2": ["RBLNGPT2LMHeadModel", "RBLNGPT2LMHeadModelConfig", "RBLNGPT2Model", "RBLNGPT2ModelConfig"],
    "idefics3": [
        "RBLNIdefics3VisionTransformer",
        "RBLNIdefics3ForConditionalGeneration",
        "RBLNIdefics3ForConditionalGenerationConfig",
        "RBLNIdefics3VisionTransformerConfig",
    ],
    "llava": ["RBLNLlavaForConditionalGeneration", "RBLNLlavaForConditionalGenerationConfig"],
    "llama": ["RBLNLlamaForCausalLM", "RBLNLlamaForCausalLMConfig", "RBLNLlamaModel", "RBLNLlamaModelConfig"],
    "opt": ["RBLNOPTForCausalLM", "RBLNOPTForCausalLMConfig", "RBLNOPTModel", "RBLNOPTModelConfig"],
    "pegasus": [
        "RBLNPegasusForConditionalGeneration",
        "RBLNPegasusModel",
        "RBLNPegasusForConditionalGenerationConfig",
        "RBLNPegasusModelConfig",
    ],
    "llava_next": ["RBLNLlavaNextForConditionalGeneration", "RBLNLlavaNextForConditionalGenerationConfig"],
    "midm": ["RBLNMidmLMHeadModel", "RBLNMidmLMHeadModelConfig"],
    "pixtral": ["RBLNPixtralVisionModel", "RBLNPixtralVisionModelConfig"],
    "mistral": [
        "RBLNMistralForCausalLM",
        "RBLNMistralForCausalLMConfig",
        "RBLNMistralModel",
        "RBLNMistralModelConfig",
    ],
    "phi": ["RBLNPhiForCausalLM", "RBLNPhiForCausalLMConfig", "RBLNPhiModel", "RBLNPhiModelConfig"],
    "qwen2": ["RBLNQwen2ForCausalLM", "RBLNQwen2ForCausalLMConfig", "RBLNQwen2Model", "RBLNQwen2ModelConfig"],
    "qwen3": ["RBLNQwen3ForCausalLM", "RBLNQwen3ForCausalLMConfig", "RBLNQwen3Model", "RBLNQwen3ModelConfig"],
    "resnet": ["RBLNResNetForImageClassification", "RBLNResNetForImageClassificationConfig"],
    "roberta": [
        "RBLNRobertaForMaskedLM",
        "RBLNRobertaForMaskedLMConfig",
        "RBLNRobertaForSequenceClassification",
        "RBLNRobertaForSequenceClassificationConfig",
    ],
    "siglip": [
        "RBLNSiglipVisionModel",
        "RBLNSiglipVisionModelConfig",
    ],
    "time_series_transformer": [
        "RBLNTimeSeriesTransformerForPrediction",
        "RBLNTimeSeriesTransformerForPredictionConfig",
    ],
    "t5": [
        "RBLNT5EncoderModel",
        "RBLNT5ForConditionalGeneration",
        "RBLNT5EncoderModelConfig",
        "RBLNT5ForConditionalGenerationConfig",
    ],
    "vit": ["RBLNViTForImageClassification", "RBLNViTForImageClassificationConfig"],
    "wav2vec2": ["RBLNWav2Vec2ForCTC", "RBLNWav2Vec2ForCTCConfig"],
    "whisper": ["RBLNWhisperForConditionalGeneration", "RBLNWhisperForConditionalGenerationConfig"],
    "xlm_roberta": [
        "RBLNXLMRobertaModel",
        "RBLNXLMRobertaModelConfig",
        "RBLNXLMRobertaForSequenceClassification",
        "RBLNXLMRobertaForSequenceClassificationConfig",
    ],
}

if TYPE_CHECKING:
    from .audio_spectrogram_transformer import RBLNASTForAudioClassification, RBLNASTForAudioClassificationConfig
    from .auto import (
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
        RBLNAutoModelForTextEncoding,
        RBLNAutoModelForVision2Seq,
    )
    from .bart import (
        RBLNBartForConditionalGeneration,
        RBLNBartForConditionalGenerationConfig,
        RBLNBartModel,
        RBLNBartModelConfig,
    )
    from .bert import (
        RBLNBertForMaskedLM,
        RBLNBertForMaskedLMConfig,
        RBLNBertForQuestionAnswering,
        RBLNBertForQuestionAnsweringConfig,
        RBLNBertModel,
        RBLNBertModelConfig,
    )
    from .blip_2 import (
        RBLNBlip2ForConditionalGeneration,
        RBLNBlip2ForConditionalGenerationConfig,
        RBLNBlip2QFormerModel,
        RBLNBlip2QFormerModelConfig,
        RBLNBlip2VisionModel,
        RBLNBlip2VisionModelConfig,
    )
    from .clip import (
        RBLNCLIPTextModel,
        RBLNCLIPTextModelConfig,
        RBLNCLIPTextModelWithProjection,
        RBLNCLIPTextModelWithProjectionConfig,
        RBLNCLIPVisionModel,
        RBLNCLIPVisionModelConfig,
        RBLNCLIPVisionModelWithProjection,
        RBLNCLIPVisionModelWithProjectionConfig,
    )
    from .colpali import RBLNColPaliForRetrieval, RBLNColPaliForRetrievalConfig
    from .decoderonly import (
        RBLNDecoderOnlyModel,
        RBLNDecoderOnlyModelConfig,
        RBLNDecoderOnlyModelForCausalLM,
        RBLNDecoderOnlyModelForCausalLMConfig,
    )
    from .depth_anything import RBLNDepthAnythingForDepthEstimation, RBLNDepthAnythingForDepthEstimationConfig
    from .distilbert import RBLNDistilBertForQuestionAnswering, RBLNDistilBertForQuestionAnsweringConfig
    from .dpt import RBLNDPTForDepthEstimation, RBLNDPTForDepthEstimationConfig
    from .exaone import RBLNExaoneForCausalLM, RBLNExaoneForCausalLMConfig
    from .gemma import RBLNGemmaForCausalLM, RBLNGemmaForCausalLMConfig, RBLNGemmaModel, RBLNGemmaModelConfig
    from .gemma3 import (
        RBLNGemma3ForCausalLM,
        RBLNGemma3ForCausalLMConfig,
        RBLNGemma3ForConditionalGeneration,
        RBLNGemma3ForConditionalGenerationConfig,
    )
    from .gpt2 import RBLNGPT2LMHeadModel, RBLNGPT2LMHeadModelConfig, RBLNGPT2Model, RBLNGPT2ModelConfig
    from .idefics3 import (
        RBLNIdefics3ForConditionalGeneration,
        RBLNIdefics3ForConditionalGenerationConfig,
        RBLNIdefics3VisionTransformer,
        RBLNIdefics3VisionTransformerConfig,
    )
    from .llama import RBLNLlamaForCausalLM, RBLNLlamaForCausalLMConfig, RBLNLlamaModel, RBLNLlamaModelConfig
    from .llava import RBLNLlavaForConditionalGeneration, RBLNLlavaForConditionalGenerationConfig
    from .llava_next import RBLNLlavaNextForConditionalGeneration, RBLNLlavaNextForConditionalGenerationConfig
    from .midm import RBLNMidmLMHeadModel, RBLNMidmLMHeadModelConfig
    from .mistral import RBLNMistralForCausalLM, RBLNMistralForCausalLMConfig, RBLNMistralModel, RBLNMistralModelConfig
    from .opt import RBLNOPTForCausalLM, RBLNOPTForCausalLMConfig, RBLNOPTModel, RBLNOPTModelConfig
    from .pegasus import (
        RBLNPegasusForConditionalGeneration,
        RBLNPegasusForConditionalGenerationConfig,
        RBLNPegasusModel,
        RBLNPegasusModelConfig,
    )
    from .phi import RBLNPhiForCausalLM, RBLNPhiForCausalLMConfig, RBLNPhiModel, RBLNPhiModelConfig
    from .pixtral import RBLNPixtralVisionModel, RBLNPixtralVisionModelConfig
    from .qwen2 import RBLNQwen2ForCausalLM, RBLNQwen2ForCausalLMConfig, RBLNQwen2Model, RBLNQwen2ModelConfig
    from .qwen2_5_vl import (
        RBLNQwen2_5_VisionTransformerPretrainedModel,
        RBLNQwen2_5_VisionTransformerPretrainedModelConfig,
        RBLNQwen2_5_VLForConditionalGeneration,
        RBLNQwen2_5_VLForConditionalGenerationConfig,
    )
    from .qwen3 import RBLNQwen3ForCausalLM, RBLNQwen3ForCausalLMConfig, RBLNQwen3Model, RBLNQwen3ModelConfig
    from .resnet import RBLNResNetForImageClassification, RBLNResNetForImageClassificationConfig
    from .roberta import (
        RBLNRobertaForMaskedLM,
        RBLNRobertaForMaskedLMConfig,
        RBLNRobertaForSequenceClassification,
        RBLNRobertaForSequenceClassificationConfig,
    )
    from .siglip import RBLNSiglipVisionModel, RBLNSiglipVisionModelConfig
    from .t5 import (
        RBLNT5EncoderModel,
        RBLNT5EncoderModelConfig,
        RBLNT5ForConditionalGeneration,
        RBLNT5ForConditionalGenerationConfig,
    )
    from .time_series_transformer import (
        RBLNTimeSeriesTransformerForPrediction,
        RBLNTimeSeriesTransformerForPredictionConfig,
    )
    from .vit import RBLNViTForImageClassification, RBLNViTForImageClassificationConfig
    from .wav2vec2 import RBLNWav2Vec2ForCTC, RBLNWav2Vec2ForCTCConfig
    from .whisper import RBLNWhisperForConditionalGeneration, RBLNWhisperForConditionalGenerationConfig
    from .xlm_roberta import (
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
