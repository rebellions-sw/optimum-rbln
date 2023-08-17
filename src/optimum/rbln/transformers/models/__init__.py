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
    "bart": ["RBLNBartForConditionalGeneration", "RBLNBartModel"],
    "bert": ["RBLNBertModel"],
    "clip": ["RBLNCLIPTextModel", "RBLNCLIPTextModelWithProjection", "RBLNCLIPVisionModel"],
    "dpt": ["RBLNDPTForDepthEstimation"],
    "exaone": ["RBLNExaoneForCausalLM"],
    "gemma": ["RBLNGemmaForCausalLM"],
    "gpt2": ["RBLNGPT2LMHeadModel"],
    "llama": ["RBLNLlamaForCausalLM"],
    "llava_next": ["RBLNLlavaNextForConditionalGeneration"],
    "midm": ["RBLNMidmLMHeadModel"],
    "mistral": ["RBLNMistralForCausalLM"],
    "phi": ["RBLNPhiForCausalLM"],
    "qwen2": ["RBLNQwen2ForCausalLM"],
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
    from .bart import RBLNBartForConditionalGeneration, RBLNBartModel
    from .bert import RBLNBertModel
    from .clip import RBLNCLIPTextModel, RBLNCLIPTextModelWithProjection, RBLNCLIPVisionModel
    from .dpt import RBLNDPTForDepthEstimation
    from .exaone import RBLNExaoneForCausalLM
    from .gemma import RBLNGemmaForCausalLM
    from .gpt2 import RBLNGPT2LMHeadModel
    from .llama import RBLNLlamaForCausalLM
    from .llava_next import RBLNLlavaNextForConditionalGeneration
    from .midm import RBLNMidmLMHeadModel
    from .mistral import RBLNMistralForCausalLM
    from .phi import RBLNPhiForCausalLM
    from .qwen2 import RBLNQwen2ForCausalLM
    from .t5 import RBLNT5EncoderModel, RBLNT5ForConditionalGeneration
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
