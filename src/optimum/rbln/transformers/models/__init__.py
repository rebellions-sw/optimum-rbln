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
    "decoderonly": [
        "RBLNDecoderOnlyModelForCausalLM",
        "RBLNDecoderOnlyModelForCausalLMConfig",
    ],
    "dpt": ["RBLNDPTForDepthEstimation"],
    "exaone": ["RBLNExaoneForCausalLM"],
    "gemma": ["RBLNGemmaForCausalLM"],
    "gpt2": ["RBLNGPT2LMHeadModel"],
    "llama": ["RBLNLlamaForCausalLM", "RBLNLlamaForCausalLMConfig"],
    "llava_next": ["RBLNLlavaNextForConditionalGeneration", "RBLNLlavaNextForConditionalGenerationConfig"],
    "midm": ["RBLNMidmLMHeadModel"],
    "mistral": ["RBLNMistralForCausalLM", "RBLNMistralForCausalLMConfig"],
    "phi": ["RBLNPhiForCausalLM"],
    "qwen2": ["RBLNQwen2ForCausalLM"],
    "time_series_transformers": ["RBLNTimeSeriesTransformerForPrediction"],
    "t5": ["RBLNT5EncoderModel", "RBLNT5ForConditionalGeneration"],
    "wav2vec2": ["RBLNWav2Vec2ForCTC"],
    "whisper": ["RBLNWhisperForConditionalGeneration"],
    "xlm_roberta": ["RBLNXLMRobertaModel"],
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
    from .dpt import RBLNDPTForDepthEstimation
    from .exaone import RBLNExaoneForCausalLM
    from .gemma import RBLNGemmaForCausalLM
    from .gpt2 import RBLNGPT2LMHeadModel
    from .llama import RBLNLlamaForCausalLM, RBLNLlamaForCausalLMConfig
    from .llava_next import RBLNLlavaNextForConditionalGeneration, RBLNLlavaNextForConditionalGenerationConfig
    from .midm import RBLNMidmLMHeadModel
    from .mistral import RBLNMistralForCausalLM, RBLNMistralForCausalLMConfig
    from .phi import RBLNPhiForCausalLM
    from .qwen2 import RBLNQwen2ForCausalLM
    from .t5 import RBLNT5EncoderModel, RBLNT5ForConditionalGeneration
    from .time_series_transformers import RBLNTimeSeriesTransformerForPrediction
    from .wav2vec2 import RBLNWav2Vec2ForCTC
    from .whisper import RBLNWhisperForConditionalGeneration
    from .xlm_roberta import RBLNXLMRobertaModel

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
