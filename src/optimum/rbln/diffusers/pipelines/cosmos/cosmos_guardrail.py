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
import os
from abc import abstractmethod
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import rebel
import torch  # noqa: I001
from diffusers.pipelines.cosmos.cosmos_guardrail import (
    CosmosSafetyChecker,
)

from optimum.rbln import RBLNLlamaForCausalLM

from ....modeling_config import DEFAULT_COMPILED_MODEL_NAME, RBLNCompileConfig, RBLNConfig, use_rbln_config
from ....utils.hub import validate_files
from ....utils.logging import get_logger
from ....utils.runtime_utils import RBLNPytorchRuntime, UnavailableRuntime


if TYPE_CHECKING:
    import torch

logger = get_logger(__name__)

COSMOS_GUARDRAIL_CHECKPOINT = "nvidia/Cosmos-1.0-Guardrail"


class RBLNsimpleModel:
    def __init__(
        self,
        models: List[rebel.Runtime],
        rbln_config: RBLNConfig,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        rbln_compiled_models: Optional[rebel.RBLNCompiledModel] = None,
        subfolder: str = "",
    ):
        self.model = models
        self.rbln_config = rbln_config
        self.compiled_models = rbln_compiled_models
        self.model_save_dir = model_save_dir
        self.subfolder = subfolder

    @classmethod
    def load_compiled_model(cls, model_id, rbln_config, subfolder=""):
        return cls._load_compiled_model(model_id=model_id, subfolder=subfolder, rbln_config=rbln_config)

    @classmethod
    def _load_compiled_model(cls, model_id: str, subfolder: str, rbln_config, rbln_compiled_models=None):
        model_path = Path(model_id)

        model_save_dir = model_path / subfolder
        rbln_files = list(model_save_dir.glob("*.rbln"))
        rbln_config_filenames = list(model_save_dir.glob("rbln_config.json"))
        validate_files(rbln_files, rbln_config_filenames, f"directory {model_save_dir}")

        from_export_method = isinstance(rbln_config, RBLNConfig) and rbln_compiled_models is not None
        if not from_export_method:
            rbln_kwargs = rbln_config or {}
            rbln_config = RBLNConfig.load(model_save_dir)
            rbln_config.update_runtime_cfg(rbln_kwargs)

        compiled_models = Path(model_save_dir).glob("*.rbln")
        rbln_compiled_models = {cm.stem: rebel.RBLNCompiledModel(cm) for cm in compiled_models}
        return cls._from_compiled_models(rbln_compiled_models, rbln_config, model_path=model_path, subfolder=subfolder)

    @classmethod
    def _from_compiled_models(
        cls,
        rbln_compiled_models: Dict[str, rebel.RBLNCompiledModel],
        rbln_config: RBLNConfig,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        subfolder="",
        **kwargs,
    ):
        compiled_model_names = [cfg.compiled_model_name for cfg in rbln_config.compile_cfgs]
        rbln_compiled_models = [rbln_compiled_models[cm_name] for cm_name in compiled_model_names]

        # create runtimes only if `rbln_create_runtimes` is enabled
        try:
            models = (
                cls._create_runtimes(rbln_compiled_models, rbln_config.device_map, rbln_config.activate_profiler)
                if rbln_config.create_runtimes
                else UnavailableRuntime()
            )

        except rebel.core.exception.RBLNRuntimeError as e:
            logger.warning(
                f"Failed to create the runtime for the model due to a runtime error: {e.__class__.__name__} - {e}"
            )
            models = UnavailableRuntime()

        models = [cls.wrap_runtime_if_needed(model) for model in models]

        return cls(
            models,
            rbln_config=rbln_config,
            model_save_dir=model_save_dir,
            rbln_compiled_models=rbln_compiled_models,
            subfolder=subfolder,
        )

    @classmethod
    def wrap_runtime_if_needed(cls, model):
        return model

    @classmethod
    def wrap_model_if_needed(cls, model):
        return model

    @classmethod
    def compile_model(cls, model, rbln_kwargs, model_save_dir, subfolder=""):
        rbln_config = cls._get_rbln_config(rbln_kwargs)
        compiled_model = cls._get_compiled_model(model, rbln_config)

        # Save compiled models (.rbln)
        (model_save_dir / subfolder).mkdir(exist_ok=True)
        if not isinstance(compiled_model, dict):
            compiled_models = {DEFAULT_COMPILED_MODEL_NAME: compiled_model}
        else:
            compiled_models = compiled_model
        for compiled_model_name, cm in compiled_models.items():
            cm.save(model_save_dir / subfolder / f"{compiled_model_name}.rbln")

        rbln_config.save(model_save_dir / subfolder)
        rbln_compiled_models = cls._load_compiled_model(
            model_id=model_save_dir, subfolder=subfolder, rbln_config=rbln_config, rbln_compiled_models=compiled_models
        )
        return rbln_compiled_models

    @classmethod
    @abstractmethod
    def _get_rbln_config(cls, rbln_kwargs):
        pass

    @classmethod
    def _get_compiled_model(cls, model, rbln_config):
        return cls._compile(cls.wrap_model_if_needed(model), rbln_compile_config=rbln_config.compile_cfgs[0])

    @classmethod
    def _compile(cls, model, rbln_compile_config: Optional[RBLNCompileConfig] = None, **kwargs):
        compiled_model = rebel.compile_from_torch(
            model,
            input_info=rbln_compile_config.input_info,
            fusion=rbln_compile_config.fusion,
            npu=rbln_compile_config.npu,
            tensor_parallel_size=rbln_compile_config.tensor_parallel_size,
            **kwargs,
        )
        return compiled_model

    @classmethod
    def _create_runtimes(
        cls,
        compiled_models: List[rebel.RBLNCompiledModel],
        rbln_device_map: Dict[str, int],
        activate_profiler: Optional[bool] = None,
    ) -> List[rebel.Runtime]:
        if DEFAULT_COMPILED_MODEL_NAME not in rbln_device_map:
            cls._raise_missing_compiled_file_error([DEFAULT_COMPILED_MODEL_NAME])

        device = rbln_device_map[DEFAULT_COMPILED_MODEL_NAME]
        return [
            compiled_model.create_runtime(tensor_type="pt", device=device, activate_profiler=activate_profiler)
            for compiled_model in compiled_models
        ]

    @staticmethod
    def _raise_missing_compiled_file_error(missing_files: List[str]):
        """Raises a KeyError with a message indicating missing compiled model files."""

        if len(missing_files) == 1:
            message = f"The rbln model folder is missing the required '{missing_files[0]}.rbln' file. "
        else:
            files_str = ", ".join([f"'{f}.rbln'" for f in missing_files])
            message = (
                "The rbln model folder is missing required files. "
                f"Ensure that {files_str} files are present in the folder. "
            )
        message += (
            "These files are necessary for loading the rbln model. "
            "If these files are missing, please recompile the model using the latest optimum-rbln "
            "and ensure the compilation completes successfully."
        )
        raise KeyError(message)

    def parameters(self):
        for param in [torch.tensor([1], dtype=torch.float32, device=torch.device("cpu"))]:
            yield param

    def to(self, device: Union[str, torch.device] = None, dtype: torch.dtype = None):
        pass

    def __call__(self, *args: List[torch.Tensor], **kwargs: Dict[str, torch.Tensor]):
        output = self.model[0](*args, **kwargs)
        return output


class RBLNRetinaFace(RBLNsimpleModel):
    @classmethod
    def _get_rbln_config(cls, rbln_kwargs):
        height = rbln_kwargs.get("height", 704)
        width = rbln_kwargs.get("width", 1280)
        input_info = [("frames", [1, 3, height, width], "float32")]  # hard coded

        postprocessor_config = RBLNCompileConfig(input_info=input_info)

        rbln_config = RBLNConfig(
            rbln_cls=cls.__name__,
            compile_cfgs=[postprocessor_config],
            rbln_kwargs=rbln_kwargs,
        )
        return rbln_config


class RBLNVideoSafetyModel(RBLNsimpleModel):
    @classmethod
    def _get_rbln_config(cls, rbln_kwargs):
        input_info_cls = [("data", [1, 1152], "float32")]  # hard coded
        cls_config = RBLNCompileConfig(input_info=input_info_cls)

        rbln_config = RBLNConfig(
            rbln_cls=cls.__name__,
            compile_cfgs=[cls_config],
            rbln_kwargs=rbln_kwargs,
        )
        return rbln_config

    @classmethod
    def wrap_model_if_needed(cls, model):
        return model.network

    def network(self, x):
        return self(x)


class RBLNRuntimeSiglipVisionModel(RBLNPytorchRuntime):
    def __init__(self, runtime, **kwargs):
        super().__init__(runtime=runtime, **kwargs)
        self.model = runtime

    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
    ) -> torch.FloatTensor:
        # Use SiglipModel's config for some fields (if specified) instead of those of vision & text components.
        vision_outputs = self.model(
            pixel_values=pixel_values,
        )
        pooled_output = vision_outputs[1]
        return pooled_output


class _SiglipVisionModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model(
            pixel_values=pixel_values,
            return_dict=False,
        )


class RBLNSiglipVisionModel(RBLNsimpleModel):
    @classmethod
    def _get_rbln_config(cls, rbln_kwargs):
        input_info_enc = [("pixel_values", [1, 3, 384, 384], "float32")]  # hard coded
        enc_config = RBLNCompileConfig(input_info=input_info_enc)

        rbln_config = RBLNConfig(
            rbln_cls=cls.__name__,
            compile_cfgs=[enc_config],
            rbln_kwargs=rbln_kwargs,
        )
        return rbln_config

    @classmethod
    def wrap_model_if_needed(cls, model):
        return _SiglipVisionModel(model)

    @classmethod
    def wrap_runtime_if_needed(cls, model):
        return RBLNRuntimeSiglipVisionModel(model)


class RBLNLlamaGuard(RBLNLlamaForCausalLM):
    @classmethod
    def load_compiled_model(cls, model_id, rbln_config, subfolder=""):
        model = RBLNLlamaForCausalLM.from_pretrained(
            model_id=model_id,
            export=False,
            rbln_config=rbln_config,
        )
        return model

    @classmethod
    def compile_model(cls, model, rbln_kwargs, model_save_dir, subfolder=""):
        batch_size = rbln_kwargs.get("batch_size", 1)

        max_seq_len = rbln_kwargs.get("max_seq_len", 4096)
        tensor_parallel_size = rbln_kwargs.get("tensor_parallel_size", 4)
        model = model.merge_and_unload()
        compiled_model = RBLNLlamaForCausalLM.from_model(
            model,
            export=True,
            rbln_batch_size=batch_size,
            rbln_max_seq_len=max_seq_len,
            rbln_tensor_parallel_size=tensor_parallel_size,
            model_save_dir=model_save_dir,
            subfolder=subfolder,
        )
        return compiled_model


class RBLNCosmosSafetyChecker:
    original_class = CosmosSafetyChecker
    _guardrails = ["video_guardrail", "video_guardrail", "video_guardrail", "text_guardrail"]
    _submodules = ["postprocessors", "safety_models", "safety_models", "safety_models"]
    _module_ids = [0, 0, 0, 1]
    _additional_modules = [None, "encoder.model", None, None]
    _target_model_names = ["net", "vision_model", "model", "model"]
    _subfolders = ["", "encoder", "model", ""]
    _rbln_modules = [RBLNRetinaFace, RBLNSiglipVisionModel, RBLNVideoSafetyModel, RBLNLlamaGuard]

    @classmethod
    @use_rbln_config
    def compile_submodules(cls, model, rbln_config, model_save_dir: str = "safety_checker"):
        save_dir_path = Path(model_save_dir)
        save_dir_path.mkdir(exist_ok=True)
        for i, target_model_name in enumerate(cls._target_model_names):
            guardrail = cls._guardrails[i]
            submodule = cls._submodules[i]
            additional_module = cls._additional_modules[i]

            save_dir_path = Path(model_save_dir + f"/{guardrail}/{submodule}")
            os.makedirs(save_dir_path, exist_ok=True)
            target_model = getattr(getattr(model, guardrail), submodule)[cls._module_ids[i]]
            if additional_module is not None:
                for m in additional_module.split("."):
                    target_model = getattr(target_model, m)
            compile_target = getattr(target_model, cls._target_model_names[i])
            compiled_model = cls._rbln_modules[i].compile_model(
                compile_target,
                rbln_kwargs=rbln_config.get(guardrail, {}),
                model_save_dir=save_dir_path,
                subfolder=cls._subfolders[i],
            )
            delattr(target_model, target_model_name)
            setattr(target_model, target_model_name, compiled_model)
        return model

    @classmethod
    @use_rbln_config
    def load_submodules(cls, model, rbln_config, model_save_dir=""):
        for i, target_model_name in enumerate(cls._target_model_names):
            guardrail = cls._guardrails[i]
            submodule = cls._submodules[i]
            additional_module = cls._additional_modules[i]
            rbln_module = cls._rbln_modules[i]

            save_dir_path = Path(model_save_dir + f"/{guardrail}/{submodule}")

            target_model = getattr(getattr(model, guardrail), submodule)[cls._module_ids[i]]
            if additional_module is not None:
                for m in additional_module.split("."):
                    target_model = getattr(target_model, m)

            compiled_model = rbln_module.load_compiled_model(
                save_dir_path,
                rbln_config=rbln_config.get(guardrail, {}),
                subfolder=cls._subfolders[i],
            )
            delattr(target_model, target_model_name)
            setattr(target_model, target_model_name, compiled_model)
        return model
