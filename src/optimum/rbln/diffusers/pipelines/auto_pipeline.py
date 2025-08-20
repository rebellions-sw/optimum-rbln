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


import importlib
from typing import Type

from diffusers.models.controlnets import ControlNetUnionModel
from diffusers.pipelines.auto_pipeline import (
    AUTO_IMAGE2IMAGE_PIPELINES_MAPPING,
    AUTO_INPAINT_PIPELINES_MAPPING,
    AUTO_TEXT2IMAGE_PIPELINES_MAPPING,
    AutoPipelineForImage2Image,
    AutoPipelineForInpainting,
    AutoPipelineForText2Image,
    _get_task_class,
)
from huggingface_hub.utils import validate_hf_hub_args

from optimum.rbln.modeling_base import RBLNBaseModel
from optimum.rbln.utils.model_utils import (
    MODEL_MAPPING,
    convert_hf_to_rbln_model_name,
    convert_rbln_to_hf_model_name,
    get_rbln_model_cls,
)


class RBLNAutoPipelineBase:
    _model_mapping = None
    _model_mapping_names = None

    @classmethod
    def get_rbln_cls(cls, pretrained_model_name_or_path, export=True, **kwargs):
        if export:
            hf_model_class = cls.infer_hf_model_class(pretrained_model_name_or_path, **kwargs)
            rbln_class_name = convert_hf_to_rbln_model_name(hf_model_class.__name__)
        else:
            rbln_class_name = cls.get_rbln_model_cls_name(pretrained_model_name_or_path, **kwargs)
            if convert_rbln_to_hf_model_name(rbln_class_name) not in cls._model_mapping_names:
                raise ValueError(
                    f"The architecture '{rbln_class_name}' is not supported by the `{cls.__name__}.from_pretrained()` method. "
                    "Please use the `from_pretrained()` method of the appropriate class to load this model, "
                    f"or directly use '{rbln_class_name}.from_pretrained()`."
                )

        try:
            rbln_cls = get_rbln_model_cls(rbln_class_name)
        except AttributeError as e:
            raise AttributeError(
                f"Class '{rbln_class_name}' not found in 'optimum.rbln' module for model ID '{pretrained_model_name_or_path}'. "
                "Ensure that the class name is correctly mapped and available in the 'optimum.rbln' module."
            ) from e

        return rbln_cls

    @classmethod
    def get_rbln_model_cls_name(cls, pretrained_model_name_or_path, **kwargs):
        """
        Retrieve the path to the compiled model directory for a given RBLN model.

        Args:
            pretrained_model_name_or_path (str): Identifier of the model.

        Returns:
            str: Path to the compiled model directory.
        """
        model_index_config = cls.load_config(pretrained_model_name_or_path)

        if "_class_name" not in model_index_config:
            raise ValueError(
                "The `_class_name` field is missing from model_index_config. This is unexpected and should be reported as an issue. "
                "Please use the `from_pretrained()` method of the appropriate class to load this model."
            )

        return model_index_config["_class_name"]

    @classmethod
    def infer_hf_model_class(
        cls,
        pretrained_model_or_path,
        cache_dir=None,
        force_download=False,
        proxies=None,
        token=None,
        local_files_only=False,
        revision=None,
        **kwargs,
    ):
        config = cls.load_config(
            pretrained_model_or_path,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            token=token,
            local_files_only=local_files_only,
            revision=revision,
        )
        pipeline_key_name = cls.get_pipeline_key_name(config, **kwargs)

        pipeline_cls = _get_task_class(cls._model_mapping, pipeline_key_name)

        return pipeline_cls

    @classmethod
    def get_pipeline_key_name(cls, config, **kwargs):
        orig_class_name = config["_class_name"]
        if "ControlPipeline" in orig_class_name:
            to_replace = "ControlPipeline"
        else:
            to_replace = "Pipeline"

        if "controlnet" in kwargs:
            if isinstance(kwargs["controlnet"], ControlNetUnionModel):
                orig_class_name = config["_class_name"].replace(to_replace, "ControlNetUnionPipeline")
            else:
                orig_class_name = config["_class_name"].replace(to_replace, "ControlNetPipeline")
        if "enable_pag" in kwargs:
            enable_pag = kwargs.pop("enable_pag")
            if enable_pag:
                orig_class_name = orig_class_name.replace(to_replace, "PAGPipeline")

        return orig_class_name

    @classmethod
    @validate_hf_hub_args
    def from_pretrained(cls, model_id, **kwargs):
        rbln_cls = cls.get_rbln_cls(model_id, **kwargs)
        return rbln_cls.from_pretrained(model_id, **kwargs)

    @classmethod
    def from_model(cls, model, **kwargs):
        rbln_cls = get_rbln_model_cls(f"RBLN{model.__class__.__name__}")
        return rbln_cls.from_model(model, **kwargs)

    @staticmethod
    def register(rbln_cls: Type[RBLNBaseModel], exist_ok=False):
        """
        Register a new RBLN model class.

        Args:
            rbln_cls (Type[RBLNBaseModel]): The RBLN model class to register.
            exist_ok (bool): Whether to allow registering an already registered model.
        """
        if not issubclass(rbln_cls, RBLNBaseModel):
            raise ValueError("`rbln_cls` must be a subclass of RBLNBaseModel.")

        native_cls = getattr(importlib.import_module("optimum.rbln"), rbln_cls.__name__, None)
        if rbln_cls.__name__ in MODEL_MAPPING or native_cls is not None:
            if not exist_ok:
                raise ValueError(f"Model for {rbln_cls.__name__} already registered.")

        MODEL_MAPPING[rbln_cls.__name__] = rbln_cls


class RBLNAutoPipelineForText2Image(RBLNAutoPipelineBase, AutoPipelineForText2Image):
    _model_mapping = AUTO_TEXT2IMAGE_PIPELINES_MAPPING
    _model_mapping_names = {x[0]: x[1].__name__ for x in AUTO_TEXT2IMAGE_PIPELINES_MAPPING.items()}


class RBLNAutoPipelineForImage2Image(RBLNAutoPipelineBase, AutoPipelineForImage2Image):
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


class RBLNAutoPipelineForInpainting(RBLNAutoPipelineBase, AutoPipelineForInpainting):
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
