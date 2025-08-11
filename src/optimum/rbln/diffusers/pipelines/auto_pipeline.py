from typing import Optional

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

from ...utils.model_utils import (
    convert_hf_to_rbln_model_name,
    convert_rbln_to_hf_model_name,
    get_rbln_model_cls,
)


class RBLNAutoPipelineBase:
    _model_mapping = None

    @classmethod
    @validate_hf_hub_args
    def from_pretrained(cls, model_id: str, export: Optional[bool] = None, **kwargs):
        if export:
            hf_model_class = cls.infer_hf_model_class(model_id, **kwargs)
            rbln_class_name = convert_hf_to_rbln_model_name(hf_model_class.__name__)
        else:
            rbln_class_name = cls.get_rbln_model_cls_name(model_id, **kwargs)
            if convert_rbln_to_hf_model_name(rbln_class_name) not in [x.__name__ for x in cls._model_mapping.values()]:
                raise ValueError(
                    f"The architecture '{rbln_class_name}' is not supported by the `{cls.__name__}.from_pretrained()` method. "
                    "Please use the `from_pretrained()` method of the appropriate class to load this model, "
                    f"or directly use '{rbln_class_name}.from_pretrained()`."
                )

        try:
            rbln_cls = get_rbln_model_cls(rbln_class_name)
        except AttributeError as e:
            raise AttributeError(
                f"Class '{rbln_class_name}' not found in 'optimum.rbln' module for model ID '{model_id}'. "
                "Ensure that the class name is correctly mapped and available in the 'optimum.rbln' module."
            ) from e

        return rbln_cls.from_pretrained(model_id, export=export, **kwargs)

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
        return model_index_config["_class_name"]

    @classmethod
    def infer_hf_model_class(
        cls,
        pretrained_model_or_path,
        *args,
        **kwargs,
    ):
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        token = kwargs.pop("token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)

        load_config_kwargs = {
            "cache_dir": cache_dir,
            "force_download": force_download,
            "proxies": proxies,
            "token": token,
            "local_files_only": local_files_only,
            "revision": revision,
        }

        config = cls.load_config(pretrained_model_or_path, **load_config_kwargs)
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


class RBLNAutoPipelineForText2Image(RBLNAutoPipelineBase, AutoPipelineForText2Image):
    _model_mapping = AUTO_TEXT2IMAGE_PIPELINES_MAPPING


class RBLNAutoPipelineForImage2Image(RBLNAutoPipelineBase, AutoPipelineForImage2Image):
    _model_mapping = AUTO_IMAGE2IMAGE_PIPELINES_MAPPING

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
