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

import copy
import importlib
from os import PathLike
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import torch

from ..modeling import RBLNModel
from ..modeling_config import RUNTIME_KEYWORDS, ContextRblnConfig, use_rbln_config
from ..utils.decorator_utils import remove_compile_time_kwargs
from ..utils.logging import get_logger


logger = get_logger(__name__)

if TYPE_CHECKING:
    from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel


class RBLNDiffusionMixin:
    """
    RBLNDiffusionMixin provides essential functionalities for compiling Stable Diffusion pipeline components to run on RBLN NPUs.
    This mixin class serves as a base for implementing RBLN-compatible Stable Diffusion pipelines. It contains shared logic for
    handling the core components of Stable Diffusion.

    To use this mixin:

    1. Create a new pipeline class that inherits from both this mixin and the original StableDiffusionPipeline.
    2. Define the required _submodules class variable listing the components to be compiled.
    3. If needed, implement get_default_rbln_config for custom configuration of submodules.

    Example:
        ```python
        class RBLNStableDiffusionPipeline(RBLNDiffusionMixin, StableDiffusionPipeline):
            _submodules = ["text_encoder", "unet", "vae"]

            @classmethod
            def get_default_rbln_config(cls, model, submodule_name, rbln_config):
                # Configuration for other submodules...
                pass
        ```

    Class Variables:
        _submodules: List of submodule names that should be compiled (typically ["text_encoder", "unet", "vae"])

    Methods:
        from_pretrained: Creates and optionally compiles a model from a pretrained checkpoint

    Notes:
        - When `export=True`, all compatible submodules will be compiled for NPU inference
        - The compilation config can be customized per submodule by including submodule names
          as keys in rbln_config
    """

    _submodules = []

    @classmethod
    @property
    def img2img_pipeline(cls):
        return "Img2Img" in cls.__name__

    @classmethod
    @property
    def inpaint_pipeline(cls):
        return "Inpaint" in cls.__name__

    @classmethod
    def get_submodule_rbln_config(
        cls, model: torch.nn.Module, submodule_name: str, rbln_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        submodule = getattr(model, submodule_name)
        submodule_class_name = submodule.__class__.__name__

        if submodule_class_name == "MultiControlNetModel":
            submodule_class_name = "ControlNetModel"

        submodule_cls: RBLNModel = getattr(importlib.import_module("optimum.rbln"), f"RBLN{submodule_class_name}")

        submodule_config = rbln_config.get(submodule_name, {})
        submodule_config = copy.deepcopy(submodule_config)

        pipe_global_config = {k: v for k, v in rbln_config.items() if k not in cls._submodules}

        submodule_config.update({k: v for k, v in pipe_global_config.items() if k not in submodule_config})
        submodule_config.update(
            {
                "img2img_pipeline": cls.img2img_pipeline,
                "inpaint_pipeline": cls.inpaint_pipeline,
            }
        )
        submodule_config = submodule_cls.update_rbln_config_using_pipe(model, submodule_config)
        return submodule_config

    @staticmethod
    def _maybe_apply_and_fuse_lora(
        model: torch.nn.Module,
        lora_ids: Optional[Union[str, List[str]]] = None,
        lora_weights_names: Optional[Union[str, List[str]]] = None,
        lora_scales: Optional[Union[float, List[float]]] = None,
    ) -> torch.nn.Module:
        lora_ids = [lora_ids] if isinstance(lora_ids, str) else lora_ids
        lora_weights_names = [lora_weights_names] if isinstance(lora_weights_names, str) else lora_weights_names
        lora_scales = [lora_scales] if isinstance(lora_scales, float) else lora_scales

        # adapt lora weight into pipeline before compilation
        if lora_ids and lora_weights_names:
            if len(lora_ids) == 1:
                if len(lora_ids) != len(lora_weights_names):
                    raise ValueError(
                        f"You must define the same number of lora ids ({len(lora_ids)} and lora weights ({len(lora_weights_names)}))"
                    )
                else:
                    model.load_lora_weights(lora_ids[0], weight_name=lora_weights_names[0])
                    model.fuse_lora(lora_scale=lora_scales[0] if lora_scales else 1.0)
            elif len(lora_ids) > 1:
                if not len(lora_ids) == len(lora_weights_names):
                    raise ValueError(
                        f"If you fuse {len(lora_ids)} lora models, but you must define the same number for lora weights and adapters."
                    )

                adapter_names = [f"adapter_{i}" for i in range(len(lora_ids))]

                for lora_id, lora_weight, adapter_name in zip(lora_ids, lora_weights_names, adapter_names):
                    model.load_lora_weights(lora_id, weight_name=lora_weight, adapter_name=adapter_name)

                if lora_scales:
                    model.set_adapters(adapter_names, adapter_weights=lora_scales)

                model.fuse_lora()
        return model

    @classmethod
    @use_rbln_config
    def from_pretrained(
        cls,
        model_id: str,
        *,
        export: bool = False,
        model_save_dir: Optional[PathLike] = None,
        rbln_config: Dict[str, Any] = {},
        lora_ids: Optional[Union[str, List[str]]] = None,
        lora_weights_names: Optional[Union[str, List[str]]] = None,
        lora_scales: Optional[Union[float, List[float]]] = None,
        **kwargs,
    ) -> RBLNModel:
        if export:
            # keep submodules if user passed any of them.
            passed_submodules = {
                name: kwargs.pop(name) for name in cls._submodules if isinstance(kwargs.get(name), RBLNModel)
            }

        else:
            # raise error if any of submodules are torch module.
            model_index_config = None
            for submodule_name in cls._submodules:
                if isinstance(kwargs.get(submodule_name), torch.nn.Module):
                    raise AssertionError(
                        f"{submodule_name} is not compiled torch module. If you want to compile, set `export=True`."
                    )

                submodule_config = rbln_config.get(submodule_name, {})

                for key, value in rbln_config.items():
                    if key in RUNTIME_KEYWORDS and key not in submodule_config:
                        submodule_config[key] = value

                if not any(kwd in submodule_config for kwd in RUNTIME_KEYWORDS):
                    continue

                if model_index_config is None:
                    model_index_config = cls.load_config(pretrained_model_name_or_path=model_id)

                module_name, class_name = model_index_config[submodule_name]
                if module_name != "optimum.rbln":
                    raise ValueError(
                        f"Invalid module_name '{module_name}' found in model_index.json for "
                        f"submodule '{submodule_name}'. "
                        "Expected 'optimum.rbln'. Please check the model_index.json configuration."
                    )

                submodule_cls: RBLNModel = getattr(importlib.import_module("optimum.rbln"), class_name)

                submodule = submodule_cls.from_pretrained(
                    model_id, export=False, subfolder=submodule_name, rbln_config=submodule_config
                )
                kwargs[submodule_name] = submodule

        with ContextRblnConfig(
            device=rbln_config.get("device"),
            device_map=rbln_config.get("device_map"),
            create_runtimes=rbln_config.get("create_runtimes"),
            optimize_host_mem=rbln_config.get("optimize_host_memory"),
            activate_profiler=rbln_config.get("activate_profiler"),
        ):
            model = super().from_pretrained(pretrained_model_name_or_path=model_id, **kwargs)

        if not export:
            return model

        model = cls._maybe_apply_and_fuse_lora(
            model,
            lora_ids=lora_ids,
            lora_weights_names=lora_weights_names,
            lora_scales=lora_scales,
        )

        compiled_submodules = cls._compile_submodules(model, passed_submodules, model_save_dir, rbln_config)
        return cls._construct_pipe(model, compiled_submodules, model_save_dir, rbln_config)

    @classmethod
    def _compile_submodules(
        cls,
        model: torch.nn.Module,
        passed_submodules: Dict[str, RBLNModel],
        model_save_dir: Optional[PathLike],
        rbln_config: Dict[str, Any],
    ) -> Dict[str, RBLNModel]:
        compiled_submodules = {}

        for submodule_name in cls._submodules:
            submodule = passed_submodules.get(submodule_name) or getattr(model, submodule_name, None)
            submodule_rbln_config = cls.get_submodule_rbln_config(model, submodule_name, rbln_config)

            if submodule is None:
                raise ValueError(f"submodule ({submodule_name}) cannot be accessed since it is not provided.")
            elif isinstance(submodule, RBLNModel):
                pass
            elif submodule_name == "controlnet" and hasattr(submodule, "nets"):
                # In case of multicontrolnet
                submodule = cls._compile_multicontrolnet(
                    controlnets=submodule,
                    model_save_dir=model_save_dir,
                    controlnet_rbln_config=submodule_rbln_config,
                )
            elif isinstance(submodule, torch.nn.Module):
                submodule_cls: RBLNModel = getattr(
                    importlib.import_module("optimum.rbln"), f"RBLN{submodule.__class__.__name__}"
                )
                submodule = submodule_cls.from_model(
                    model=submodule,
                    subfolder=submodule_name,
                    model_save_dir=model_save_dir,
                    rbln_config=submodule_rbln_config,
                )
            else:
                raise ValueError(f"Unknown class of submodule({submodule_name}) : {submodule.__class__.__name__} ")

            compiled_submodules[submodule_name] = submodule
        return compiled_submodules

    @classmethod
    def _compile_multicontrolnet(
        cls,
        controlnets: "MultiControlNetModel",
        model_save_dir: Optional[PathLike],
        controlnet_rbln_config: Dict[str, Any],
    ):
        # Compile multiple ControlNet models for a MultiControlNet setup
        from .models.controlnet import RBLNControlNetModel
        from .pipelines.controlnet import RBLNMultiControlNetModel

        compiled_controlnets = [
            RBLNControlNetModel.from_model(
                model=controlnet,
                subfolder="controlnet" if i == 0 else f"controlnet_{i}",
                model_save_dir=model_save_dir,
                rbln_config=controlnet_rbln_config,
            )
            for i, controlnet in enumerate(controlnets.nets)
        ]
        return RBLNMultiControlNetModel(compiled_controlnets)

    @classmethod
    def _construct_pipe(cls, model, submodules, model_save_dir, rbln_config):
        # Construct finalize pipe setup with compiled submodules and configurations

        if model_save_dir is not None:
            # To skip saving original pytorch modules
            for submodule_name in cls._submodules:
                delattr(model, submodule_name)

            # Direct calling of `save_pretrained` causes config.unet = (None, None).
            # So config must be saved again, later.
            model.save_pretrained(model_save_dir)
            # FIXME: Here, model touches its submodules such as model.unet,
            # Causing warning messeages.

        update_dict = {}
        for submodule_name in cls._submodules:
            # replace submodule
            setattr(model, submodule_name, submodules[submodule_name])
            update_dict[submodule_name] = ("optimum.rbln", submodules[submodule_name].__class__.__name__)

        # Update config to be able to load from model directory.
        #
        # e.g)
        # update_dict = {
        #     "vae": ("optimum.rbln", "RBLNAutoencoderKL"),
        #     "text_encoder": ("optimum.rbln", "RBLNCLIPTextModel"),
        #     "unet": ("optimum.rbln", "RBLNUNet2DConditionModel"),
        # }
        model.register_to_config(**update_dict)

        if model_save_dir:
            # overwrite to replace incorrect config
            model.save_config(model_save_dir)

        if rbln_config.get("optimize_host_memory") is False:
            # Keep compiled_model objs to further analysis. -> TODO: remove soon...
            model.compiled_models = []
            for name in cls._submodules:
                submodule = getattr(model, name)
                model.compiled_models.extend(submodule.compiled_models)

        return model

    def get_compiled_image_size(self):
        if hasattr(self, "vae"):
            compiled_image_size = self.vae.image_size
        else:
            compiled_image_size = None
        return compiled_image_size

    def handle_additional_kwargs(self, **kwargs):
        """
        Function to handle additional compile-time parameters during inference.

        If the additional variable is determined by another module, this method should be overrided.

        Example:
            ```python
            if hasattr(self, "movq"):
                compiled_image_size = self.movq.image_size
                kwargs["height"] = compiled_image_size[0]
                kwargs["width"] = compiled_image_size[1]

            compiled_num_frames = self.unet.rbln_config.model_cfg.get("num_frames", None)
            if compiled_num_frames is not None:
                kwargs["num_frames"] = self.unet.rbln_config.model_cfg.get("num_frames")
            return kwargs
            ```
        """
        return kwargs

    @remove_compile_time_kwargs
    def __call__(self, *args, **kwargs):
        kwargs = self.handle_additional_kwargs(**kwargs)
        return super().__call__(*args, **kwargs)
