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

from .__version__ import __version__
from .utils import check_version_compats


_import_structure = {
    "modeling": [
        "RBLNBaseModel",
        "RBLNModel",
    ],
    "modeling_config": [
        "RBLNCompileConfig",
        "RBLNConfig",
    ],
    "transformers": [
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
        "RBLNLlamaForCausalLM",
        "RBLNT5EncoderModel",
        "RBLNT5ForConditionalGeneration",
        "RBLNPhiForCausalLM",
        "RBLNLlavaNextForConditionalGeneration",
        "RBLNMidmLMHeadModel",
        "RBLNMistralForCausalLM",
        "RBLNWhisperForConditionalGeneration",
        "RBLNXLMRobertaModel",
        "RBLNASTForAudioClassification",
        "RBLNBertForQuestionAnswering",
        "RBLNDistilBertForQuestionAnswering",
        "RBLNResNetForImageClassification",
        "RBLNXLMRobertaForSequenceClassification",
        "RBLNRobertaForSequenceClassification",
        "RBLNRobertaForMaskedLM",
        "RBLNViTForImageClassification",
    ],
    "diffusers": [
        "RBLNStableDiffusionPipeline",
        "RBLNStableDiffusionXLPipeline",
        "RBLNAutoencoderKL",
        "RBLNUNet2DConditionModel",
        "RBLNControlNetModel",
        "RBLNStableDiffusionImg2ImgPipeline",
        "RBLNStableDiffusionInpaintPipeline",
        "RBLNStableDiffusionControlNetImg2ImgPipeline",
        "RBLNMultiControlNetModel",
        "RBLNStableDiffusionXLImg2ImgPipeline",
        "RBLNStableDiffusionXLInpaintPipeline",
        "RBLNStableDiffusionControlNetPipeline",
        "RBLNStableDiffusionXLControlNetPipeline",
        "RBLNStableDiffusionXLControlNetImg2ImgPipeline",
        "RBLNSD3Transformer2DModel",
        "RBLNStableDiffusion3Img2ImgPipeline",
        "RBLNStableDiffusion3InpaintPipeline",
        "RBLNStableDiffusion3Pipeline",
        "RBLNDiffusionMixin",
    ],
}

if TYPE_CHECKING:
    from .diffusers import (
        RBLNAutoencoderKL,
        RBLNControlNetModel,
        RBLNDiffusionMixin,
        RBLNMultiControlNetModel,
        RBLNSD3Transformer2DModel,
        RBLNStableDiffusion3Img2ImgPipeline,
        RBLNStableDiffusion3InpaintPipeline,
        RBLNStableDiffusion3Pipeline,
        RBLNStableDiffusionControlNetImg2ImgPipeline,
        RBLNStableDiffusionControlNetPipeline,
        RBLNStableDiffusionImg2ImgPipeline,
        RBLNStableDiffusionInpaintPipeline,
        RBLNStableDiffusionPipeline,
        RBLNStableDiffusionXLControlNetImg2ImgPipeline,
        RBLNStableDiffusionXLControlNetPipeline,
        RBLNStableDiffusionXLImg2ImgPipeline,
        RBLNStableDiffusionXLInpaintPipeline,
        RBLNStableDiffusionXLPipeline,
        RBLNUNet2DConditionModel,
    )
    from .modeling import (
        RBLNBaseModel,
        RBLNModel,
    )
    from .modeling_config import (
        RBLNCompileConfig,
        RBLNConfig,
    )
    from .transformers import (
        RBLNASTForAudioClassification,
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
        RBLNBertForQuestionAnswering,
        RBLNBertModel,
        RBLNCLIPTextModel,
        RBLNCLIPTextModelWithProjection,
        RBLNCLIPVisionModel,
        RBLNDistilBertForQuestionAnswering,
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
        RBLNResNetForImageClassification,
        RBLNRobertaForMaskedLM,
        RBLNRobertaForSequenceClassification,
        RBLNT5EncoderModel,
        RBLNT5ForConditionalGeneration,
        RBLNViTForImageClassification,
        RBLNWav2Vec2ForCTC,
        RBLNWhisperForConditionalGeneration,
        RBLNXLMRobertaForSequenceClassification,
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


check_version_compats()
