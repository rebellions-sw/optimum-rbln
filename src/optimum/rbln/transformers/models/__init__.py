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
    "qwen2_5_vl": [
        "RBLNQwen2_5_VisionTransformerPretrainedModel",
        "RBLNQwen2_5_VisionTransformerPretrainedModelConfig",
        "RBLNQwen2_5_VLForConditionalGeneration",
        "RBLNQwen2_5_VLForConditionalGenerationConfig",
    ],
    "decoderonly": [
        "RBLNDecoderOnlyModelForCausalLM",
        "RBLNDecoderOnlyModelForCausalLMConfig",
    ],
    "dpt": [
        "RBLNDPTForDepthEstimation",
        "RBLNDPTForDepthEstimationConfig",
    ],
    "exaone": ["RBLNExaoneForCausalLM", "RBLNExaoneForCausalLMConfig"],
    "gemma": ["RBLNGemmaForCausalLM", "RBLNGemmaForCausalLMConfig"],
    "gpt2": ["RBLNGPT2LMHeadModel", "RBLNGPT2LMHeadModelConfig"],
    "idefics3": [
        "RBLNIdefics3VisionTransformer",
        "RBLNIdefics3ForConditionalGeneration",
        "RBLNIdefics3ForConditionalGenerationConfig",
        "RBLNIdefics3VisionTransformerConfig",
    ],
    "llama": ["RBLNLlamaForCausalLM", "RBLNLlamaForCausalLMConfig"],
    "opt": ["RBLNOPTForCausalLM", "RBLNOPTForCausalLMConfig"],
    "llava_next": ["RBLNLlavaNextForConditionalGeneration", "RBLNLlavaNextForConditionalGenerationConfig"],
    "midm": ["RBLNMidmLMHeadModel", "RBLNMidmLMHeadModelConfig"],
    "mistral": ["RBLNMistralForCausalLM", "RBLNMistralForCausalLMConfig"],
    "phi": ["RBLNPhiForCausalLM", "RBLNPhiForCausalLMConfig"],
    "qwen2": ["RBLNQwen2ForCausalLM", "RBLNQwen2ForCausalLMConfig"],
    "siglip": [
        "RBLNSiglipVisionModel",
        "RBLNSiglipVisionModelConfig",
    ],
    "time_series_transformers": [
        "RBLNTimeSeriesTransformerForPrediction",
        "RBLNTimeSeriesTransformerForPredictionConfig",
    ],
    "t5": [
        "RBLNT5EncoderModel",
        "RBLNT5ForConditionalGeneration",
        "RBLNT5EncoderModelConfig",
        "RBLNT5ForConditionalGenerationConfig",
    ],
    "wav2vec2": ["RBLNWav2Vec2ForCTC", "RBLNWav2Vec2ForCTCConfig"],
    "whisper": ["RBLNWhisperForConditionalGeneration", "RBLNWhisperForConditionalGenerationConfig"],
    "xlm_roberta": ["RBLNXLMRobertaModel", "RBLNXLMRobertaModelConfig"],
}

if TYPE_CHECKING:
    from .auto import (
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
    from .decoderonly import (
        RBLNDecoderOnlyModelForCausalLM,
        RBLNDecoderOnlyModelForCausalLMConfig,
    )
    from .dpt import (
        RBLNDPTForDepthEstimation,
        RBLNDPTForDepthEstimationConfig,
    )
    from .exaone import RBLNExaoneForCausalLM, RBLNExaoneForCausalLMConfig
    from .gemma import RBLNGemmaForCausalLM, RBLNGemmaForCausalLMConfig
    from .gpt2 import RBLNGPT2LMHeadModel, RBLNGPT2LMHeadModelConfig
    from .idefics3 import (
        RBLNIdefics3ForConditionalGeneration,
        RBLNIdefics3ForConditionalGenerationConfig,
        RBLNIdefics3VisionTransformer,
        RBLNIdefics3VisionTransformerConfig,
    )
    from .llama import RBLNLlamaForCausalLM, RBLNLlamaForCausalLMConfig
    from .llava_next import RBLNLlavaNextForConditionalGeneration, RBLNLlavaNextForConditionalGenerationConfig
    from .midm import RBLNMidmLMHeadModel, RBLNMidmLMHeadModelConfig
    from .mistral import RBLNMistralForCausalLM, RBLNMistralForCausalLMConfig
    from .opt import RBLNOPTForCausalLM, RBLNOPTForCausalLMConfig
    from .phi import RBLNPhiForCausalLM, RBLNPhiForCausalLMConfig
    from .qwen2 import RBLNQwen2ForCausalLM, RBLNQwen2ForCausalLMConfig
    from .qwen2_5_vl import (
        RBLNQwen2_5_VisionTransformerPretrainedModel,
        RBLNQwen2_5_VisionTransformerPretrainedModelConfig,
        RBLNQwen2_5_VLForConditionalGeneration,
        RBLNQwen2_5_VLForConditionalGenerationConfig,
    )
    from .siglip import RBLNSiglipVisionModel, RBLNSiglipVisionModelConfig
    from .t5 import (
        RBLNT5EncoderModel,
        RBLNT5EncoderModelConfig,
        RBLNT5ForConditionalGeneration,
        RBLNT5ForConditionalGenerationConfig,
    )
    from .time_series_transformers import (
        RBLNTimeSeriesTransformerForPrediction,
        RBLNTimeSeriesTransformerForPredictionConfig,
    )
    from .wav2vec2 import RBLNWav2Vec2ForCTC, RBLNWav2Vec2ForCTCConfig
    from .whisper import RBLNWhisperForConditionalGeneration, RBLNWhisperForConditionalGenerationConfig
    from .xlm_roberta import RBLNXLMRobertaModel, RBLNXLMRobertaModelConfig

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
