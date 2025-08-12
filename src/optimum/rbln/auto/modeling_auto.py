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

from diffusers.models.controlnets import ControlNetUnionModel
from diffusers.pipelines.auto_pipeline import (
    AUTO_IMAGE2IMAGE_PIPELINES_MAPPING,
    AUTO_INPAINT_PIPELINES_MAPPING,
    AUTO_TEXT2IMAGE_PIPELINES_MAPPING,
    AutoPipelineForImage2Image,
    AutoPipelineForInpainting,
    AutoPipelineForText2Image,
)
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING,
    MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_CTC_MAPPING,
    MODEL_FOR_CTC_MAPPING_NAMES,
    MODEL_FOR_DEPTH_ESTIMATION_MAPPING,
    MODEL_FOR_DEPTH_ESTIMATION_MAPPING_NAMES,
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING,
    MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES,
    MODEL_FOR_MASKED_LM_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING_NAMES,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
    MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES,
    MODEL_FOR_TEXT_ENCODING_MAPPING,
    MODEL_FOR_TEXT_ENCODING_MAPPING_NAMES,
    MODEL_FOR_VISION_2_SEQ_MAPPING,
    MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES,
    MODEL_MAPPING,
    MODEL_MAPPING_NAMES,
)

from .auto_factory import _BaseAutoModelClass, _BaseAutoPipelineClass


MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.update(
    {
        "midm": "MidmLMHeadModel",
        "exaone": "ExaoneForCausalLM",
    }
)


class RBLNAutoModel(_BaseAutoModelClass):
    _model_mapping = MODEL_MAPPING
    _model_mapping_names = MODEL_MAPPING_NAMES


class RBLNAutoModelForCTC(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_CTC_MAPPING
    _model_mapping_names = MODEL_FOR_CTC_MAPPING_NAMES


class RBLNAutoModelForCausalLM(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_CAUSAL_LM_MAPPING
    _model_mapping_names = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES


class RBLNAutoModelForSeq2SeqLM(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING
    _model_mapping_names = MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES


class RBLNAutoModelForSpeechSeq2Seq(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING
    _model_mapping_names = MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES


class RBLNAutoModelForDepthEstimation(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_DEPTH_ESTIMATION_MAPPING
    _model_mapping_names = MODEL_FOR_DEPTH_ESTIMATION_MAPPING_NAMES


class RBLNAutoModelForSequenceClassification(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING
    _model_mapping_names = MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES


class RBLNAutoModelForVision2Seq(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_VISION_2_SEQ_MAPPING
    _model_mapping_names = MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES


class RBLNAutoModelForImageTextToText(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING
    _model_mapping_names = MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES


class RBLNAutoModelForMaskedLM(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_MASKED_LM_MAPPING
    _model_mapping_names = MODEL_FOR_MASKED_LM_MAPPING_NAMES


class RBLNAutoModelForAudioClassification(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING
    _model_mapping_names = MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES


class RBLNAutoModelForImageClassification(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING
    _model_mapping_names = MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES


class RBLNAutoModelForQuestionAnswering(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_QUESTION_ANSWERING_MAPPING
    _model_mapping_names = MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES


class RBLNAutoModelForTextEncoding(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_TEXT_ENCODING_MAPPING
    _model_mapping_names = MODEL_FOR_TEXT_ENCODING_MAPPING_NAMES


class RBLNAutoPipelineForText2Image(_BaseAutoPipelineClass, AutoPipelineForText2Image):
    _model_mapping = AUTO_TEXT2IMAGE_PIPELINES_MAPPING
    _model_mapping_names = {x[0]: x[1].__name__ for x in AUTO_TEXT2IMAGE_PIPELINES_MAPPING.items()}


class RBLNAutoPipelineForImage2Image(_BaseAutoPipelineClass, AutoPipelineForImage2Image):
    _model_mapping = AUTO_IMAGE2IMAGE_PIPELINES_MAPPING
    _model_mapping_names = {x[0]: x[1].__name__ for x in AUTO_IMAGE2IMAGE_PIPELINES_MAPPING.items()}

    @classmethod
    def get_pipeline_key_name(cls, config, **kwargs):
        orig_class_name = config["_class_name"]
        # the `orig_class_name` can be:
        # `- *Pipeline` (for regular text-to-image checkpoint)
        #  - `*ControlPipeline` (for Flux tools specific checkpoint)
        # `- *Img2ImgPipeline` (for refiner checkpoint)
        if "Img2Img" in orig_class_name:
            to_replace = "Img2ImgPipeline"
        elif "ControlPipeline" in orig_class_name:
            to_replace = "ControlPipeline"
        else:
            to_replace = "Pipeline"

        if "controlnet" in kwargs:
            if isinstance(kwargs["controlnet"], ControlNetUnionModel):
                orig_class_name = orig_class_name.replace(to_replace, "ControlNetUnion" + to_replace)
            else:
                orig_class_name = orig_class_name.replace(to_replace, "ControlNet" + to_replace)
        if "enable_pag" in kwargs:
            enable_pag = kwargs.pop("enable_pag")
            if enable_pag:
                orig_class_name = orig_class_name.replace(to_replace, "PAG" + to_replace)

        if to_replace == "ControlPipeline":
            orig_class_name = orig_class_name.replace(to_replace, "ControlImg2ImgPipeline")

        return orig_class_name


class RBLNAutoPipelineForInpainting(_BaseAutoPipelineClass, AutoPipelineForInpainting):
    _model_mapping = AUTO_INPAINT_PIPELINES_MAPPING
    _model_mapping_names = {x[0]: x[1].__name__ for x in AUTO_INPAINT_PIPELINES_MAPPING.items()}

    @classmethod
    def get_pipeline_key_name(cls, config, **kwargs):
        orig_class_name = config["_class_name"]

        # The `orig_class_name`` can be:
        # `- *InpaintPipeline` (for inpaint-specific checkpoint)
        #  - `*ControlPipeline` (for Flux tools specific checkpoint)
        #  - or *Pipeline (for regular text-to-image checkpoint)
        if "Inpaint" in orig_class_name:
            to_replace = "InpaintPipeline"
        elif "ControlPipeline" in orig_class_name:
            to_replace = "ControlPipeline"
        else:
            to_replace = "Pipeline"

        if "controlnet" in kwargs:
            if isinstance(kwargs["controlnet"], ControlNetUnionModel):
                orig_class_name = orig_class_name.replace(to_replace, "ControlNetUnion" + to_replace)
            else:
                orig_class_name = orig_class_name.replace(to_replace, "ControlNet" + to_replace)
        if "enable_pag" in kwargs:
            enable_pag = kwargs.pop("enable_pag")
            if enable_pag:
                orig_class_name = orig_class_name.replace(to_replace, "PAG" + to_replace)
        if to_replace == "ControlPipeline":
            orig_class_name = orig_class_name.replace(to_replace, "ControlInpaintPipeline")

        return orig_class_name
