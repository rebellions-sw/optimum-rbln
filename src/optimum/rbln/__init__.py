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

from .__version__ import __version__
from .utils import check_version_compats


_import_structure = {
    "modeling": [
        "RBLNBaseModel",
        "RBLNModel",
    ],
    "configuration_utils": [
        "RBLNCompileConfig",
        "RBLNModelConfig",
    ],
    "transformers": [
        "RBLNASTForAudioClassification",
        "RBLNASTForAudioClassificationConfig",
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
        "RBLNBartForConditionalGenerationConfig",
        "RBLNBartModel",
        "RBLNBartModelConfig",
        "RBLNBertForMaskedLM",
        "RBLNBertForMaskedLMConfig",
        "RBLNBertForQuestionAnswering",
        "RBLNBertForQuestionAnsweringConfig",
        "RBLNBertModel",
        "RBLNBertModelConfig",
        "RBLNCLIPTextModel",
        "RBLNCLIPTextModelConfig",
        "RBLNCLIPTextModelWithProjection",
        "RBLNCLIPTextModelWithProjectionConfig",
        "RBLNCLIPVisionModel",
        "RBLNCLIPVisionModelConfig",
        "RBLNCLIPVisionModelWithProjection",
        "RBLNCLIPVisionModelWithProjectionConfig",
        "RBLNDistilBertForQuestionAnswering",
        "RBLNDistilBertForQuestionAnsweringConfig",
        "RBLNDecoderOnlyModelForCausalLM",
        "RBLNDecoderOnlyModelForCausalLMConfig",
        "RBLNDPTForDepthEstimation",
        "RBLNDPTForDepthEstimationConfig",
        "RBLNExaoneForCausalLM",
        "RBLNExaoneForCausalLMConfig",
        "RBLNGemmaForCausalLM",
        "RBLNGemmaForCausalLMConfig",
        "RBLNGPT2LMHeadModel",
        "RBLNGPT2LMHeadModelConfig",
        "RBLNLlamaForCausalLM",
        "RBLNLlamaForCausalLMConfig",
        "RBLNLlavaNextForConditionalGeneration",
        "RBLNLlavaNextForConditionalGenerationConfig",
        "RBLNMidmLMHeadModel",
        "RBLNMidmForCausalLMConfig",
        "RBLNMistralForCausalLM",
        "RBLNMistralForCausalLMConfig",
        "RBLNPhiForCausalLM",
        "RBLNPhiForCausalLMConfig",
        "RBLNQwen2ForCausalLM",
        "RBLNQwen2ForCausalLMConfig",
        "RBLNResNetForImageClassification",
        "RBLNResNetForImageClassificationConfig",
        "RBLNRobertaForMaskedLM",
        "RBLNRobertaForMaskedLMConfig",
        "RBLNRobertaForSequenceClassification",
        "RBLNRobertaForSequenceClassificationConfig",
        "RBLNT5EncoderModel",
        "RBLNT5EncoderModelConfig",
        "RBLNT5ForConditionalGeneration",
        "RBLNT5ForConditionalGenerationConfig",
        "RBLNViTForImageClassification",
        "RBLNViTForImageClassificationConfig",
        "RBLNWav2Vec2ForCTC",
        "RBLNWav2Vec2ForCTCConfig",
        "RBLNWhisperForConditionalGeneration",
        "RBLNWhisperForConditionalGenerationConfig",
        "RBLNXLMRobertaForSequenceClassification",
        "RBLNXLMRobertaForSequenceClassificationConfig",
        "RBLNXLMRobertaModel",
        "RBLNBertForMaskedLM",
        "RBLNTimeSeriesTransformerForPrediction",
    ],
    "diffusers": [
        "RBLNAutoencoderKL",
        "RBLNAutoencoderKLConfig",
        "RBLNControlNetModel",
        "RBLNDiffusionMixin",
        "RBLNKandinskyV22CombinedPipeline",
        "RBLNKandinskyV22Img2ImgCombinedPipeline",
        "RBLNKandinskyV22Img2ImgPipeline",
        "RBLNKandinskyV22InpaintCombinedPipeline",
        "RBLNKandinskyV22InpaintPipeline",
        "RBLNKandinskyV22Pipeline",
        "RBLNKandinskyV22PriorPipeline",
        "RBLNMultiControlNetModel",
        "RBLNPriorTransformer",
        "RBLNSD3Transformer2DModel",
        "RBLNStableDiffusion3Img2ImgPipeline",
        "RBLNStableDiffusion3InpaintPipeline",
        "RBLNStableDiffusion3Pipeline",
        "RBLNStableDiffusionControlNetImg2ImgPipeline",
        "RBLNStableDiffusionControlNetPipeline",
        "RBLNStableDiffusionImg2ImgPipeline",
        "RBLNStableDiffusionImg2ImgPipelineConfig",
        "RBLNStableDiffusionInpaintPipeline",
        "RBLNStableDiffusionInpaintPipelineConfig",
        "RBLNStableDiffusionPipeline",
        "RBLNStableDiffusionPipelineConfig",
        "RBLNStableDiffusionXLControlNetImg2ImgPipeline",
        "RBLNStableDiffusionXLControlNetPipeline",
        "RBLNStableDiffusionXLImg2ImgPipeline",
        "RBLNStableDiffusionXLInpaintPipeline",
        "RBLNStableDiffusionXLPipeline",
        "RBLNUNet2DConditionModel",
        "RBLNUNet2DConditionModelConfig",
        "RBLNVQModel",
    ],
}

if TYPE_CHECKING:
    from .configuration_utils import (
        RBLNCompileConfig,
        RBLNModelConfig,
    )
    from .diffusers import (
        RBLNAutoencoderKL,
        RBLNAutoencoderKLConfig,
        RBLNControlNetModel,
        RBLNDiffusionMixin,
        RBLNKandinskyV22CombinedPipeline,
        RBLNKandinskyV22Img2ImgCombinedPipeline,
        RBLNKandinskyV22Img2ImgPipeline,
        RBLNKandinskyV22InpaintCombinedPipeline,
        RBLNKandinskyV22InpaintPipeline,
        RBLNKandinskyV22Pipeline,
        RBLNKandinskyV22PriorPipeline,
        RBLNMultiControlNetModel,
        RBLNPriorTransformer,
        RBLNSD3Transformer2DModel,
        RBLNStableDiffusion3Img2ImgPipeline,
        RBLNStableDiffusion3InpaintPipeline,
        RBLNStableDiffusion3Pipeline,
        RBLNStableDiffusionControlNetImg2ImgPipeline,
        RBLNStableDiffusionControlNetPipeline,
        RBLNStableDiffusionImg2ImgPipeline,
        RBLNStableDiffusionImg2ImgPipelineConfig,
        RBLNStableDiffusionInpaintPipeline,
        RBLNStableDiffusionInpaintPipelineConfig,
        RBLNStableDiffusionPipeline,
        RBLNStableDiffusionPipelineConfig,
        RBLNStableDiffusionXLControlNetImg2ImgPipeline,
        RBLNStableDiffusionXLControlNetPipeline,
        RBLNStableDiffusionXLImg2ImgPipeline,
        RBLNStableDiffusionXLInpaintPipeline,
        RBLNStableDiffusionXLPipeline,
        RBLNUNet2DConditionModel,
        RBLNUNet2DConditionModelConfig,
        RBLNVQModel,
    )
    from .modeling import (
        RBLNBaseModel,
        RBLNModel,
    )
    from .transformers import (
        RBLNASTForAudioClassification,
        RBLNASTForAudioClassificationConfig,
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
        RBLNBartForConditionalGenerationConfig,
        RBLNBartModel,
        RBLNBartModelConfig,
        RBLNBertForMaskedLM,
        RBLNBertForMaskedLMConfig,
        RBLNBertForQuestionAnswering,
        RBLNBertForQuestionAnsweringConfig,
        RBLNBertModel,
        RBLNBertModelConfig,
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
        RBLNDistilBertForQuestionAnswering,
        RBLNDistilBertForQuestionAnsweringConfig,
        RBLNDPTForDepthEstimation,
        RBLNDPTForDepthEstimationConfig,
        RBLNExaoneForCausalLM,
        RBLNExaoneForCausalLMConfig,
        RBLNGemmaForCausalLM,
        RBLNGemmaForCausalLMConfig,
        RBLNGPT2LMHeadModel,
        RBLNGPT2LMHeadModelConfig,
        RBLNLlamaForCausalLM,
        RBLNLlamaForCausalLMConfig,
        RBLNLlavaNextForConditionalGeneration,
        RBLNLlavaNextForConditionalGenerationConfig,
        RBLNMidmForCausalLMConfig,
        RBLNMidmLMHeadModel,
        RBLNMistralForCausalLM,
        RBLNMistralForCausalLMConfig,
        RBLNPhiForCausalLM,
        RBLNPhiForCausalLMConfig,
        RBLNQwen2ForCausalLM,
        RBLNQwen2ForCausalLMConfig,
        RBLNResNetForImageClassification,
        RBLNResNetForImageClassificationConfig,
        RBLNRobertaForMaskedLM,
        RBLNRobertaForMaskedLMConfig,
        RBLNRobertaForSequenceClassification,
        RBLNRobertaForSequenceClassificationConfig,
        RBLNT5EncoderModel,
        RBLNT5EncoderModelConfig,
        RBLNT5ForConditionalGeneration,
        RBLNT5ForConditionalGenerationConfig,
        RBLNTimeSeriesTransformerForPrediction,
        RBLNViTForImageClassification,
        RBLNViTForImageClassificationConfig,
        RBLNWav2Vec2ForCTC,
        RBLNWav2Vec2ForCTCConfig,
        RBLNWhisperForConditionalGeneration,
        RBLNWhisperForConditionalGenerationConfig,
        RBLNXLMRobertaForSequenceClassification,
        RBLNXLMRobertaForSequenceClassificationConfig,
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
