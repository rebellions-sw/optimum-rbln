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

import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import rebel
import torch
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from transformers import AutoConfig, PretrainedConfig

from .modeling_base import RBLNBaseModel
from .modeling_config import DEFAULT_COMPILED_MODEL_NAME, RBLNConfig, use_rbln_config


if TYPE_CHECKING:
    from transformers import PreTrainedModel


logger = logging.getLogger(__name__)


class RBLNModel(RBLNBaseModel):
    """
    A class that inherits from RBLNBaseModel for models consisting of a single `torch.nn.Module`.

    This class supports all the functionality of RBLNBaseModel, including loading and saving models using
    the `from_pretrained` and `save_pretrained` methods, compiling PyTorch models for execution on RBLN NPU
    devices.

    Example:
        ```python
        model = RBLNModel.from_pretrained("model_id", export=True, rbln_npu="npu_name")
        outputs = model(**inputs)
        ```
    """

    @classmethod
    def update_kwargs(cls, kwargs):
        """
        Update user-given kwargs to get proper pytorch model.

        For example, `torchscript`=True should be set because torch.jit
        does not support `transformers` output instances as module output;
        """
        kwargs.update(
            {
                "torchscript": True,
                "return_dict": False,
            }
        )
        return kwargs

    @classmethod
    def save_torch_artifacts(
        cls,
        model: "PreTrainedModel",
        save_dir_path: Path,
        subfolder: str,
        rbln_config: RBLNConfig,
    ):
        """
        If you are unavoidably running on a CPU rather than an RBLN device,
        store the torch tensor, weight, etc. in this function.
        """

    @classmethod
    def wrap_model_if_needed(cls, model: torch.nn.Module, rbln_config: RBLNConfig) -> torch.nn.Module:
        # Wrap the model if needed.
        return model

    @classmethod
    def get_compiled_model(cls, model: "PreTrainedModel", rbln_config: RBLNConfig):
        model = cls.wrap_model_if_needed(model, rbln_config)
        rbln_compile_config = rbln_config.compile_cfgs[0]
        compiled_model = cls.compile(model, rbln_compile_config=rbln_compile_config)
        return compiled_model

    @classmethod
    @use_rbln_config
    def from_model(
        cls,
        model: "PreTrainedModel",
        config: Optional[PretrainedConfig] = None,
        rbln_config: Dict[str, Any] = {},
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        subfolder: str = "",
        **kwargs,
    ):
        preprocessors = kwargs.pop("preprocessors", [])
        rbln_kwargs = rbln_config

        # Directory to save compile artifacts(.rbln) and original configs
        if model_save_dir is None:
            save_dir = TemporaryDirectory()
            save_dir_path = Path(save_dir.name)
        else:
            save_dir = model_save_dir
            if isinstance(save_dir, TemporaryDirectory):
                save_dir_path = Path(model_save_dir.name)
            else:
                save_dir_path = Path(model_save_dir)
                save_dir_path.mkdir(exist_ok=True)

        # Save configs
        if config is None:
            config = model.config
            # remote_config
            if hasattr(config, "auto_map") and "AutoConfig" in config.auto_map:
                config = AutoConfig.from_pretrained(config._name_or_path, **kwargs)

        if hasattr(model, "can_generate") and model.can_generate():
            generation_config = model.generation_config
            generation_config.save_pretrained(save_dir_path / subfolder)

        if not isinstance(config, PretrainedConfig):  # diffusers config
            config = PretrainedConfig(**config)
        config.save_pretrained(save_dir_path / subfolder)

        # Save preprocessor
        for preprocessor in preprocessors:
            preprocessor.save_pretrained(save_dir_path / subfolder)

        # Get compilation arguments (e.g. input_info)
        rbln_config: RBLNConfig = cls.get_rbln_config(
            preprocessors=preprocessors, model_config=config, rbln_kwargs=rbln_kwargs
        )
        # rbln_config.update_runtime_cfg(rbln_kwargs) # This is done in get_rbln_config

        compiled_model: Union[rebel.RBLNCompiledModel, Dict[str, rebel.RBLNCompiledModel]] = cls.get_compiled_model(
            model, rbln_config=rbln_config
        )

        # Save compiled models (.rbln)
        (save_dir_path / subfolder).mkdir(exist_ok=True)
        if not isinstance(compiled_model, dict):
            compiled_models = {DEFAULT_COMPILED_MODEL_NAME: compiled_model}
        else:
            compiled_models = compiled_model
        for compiled_model_name, cm in compiled_models.items():
            cm.save(save_dir_path / subfolder / f"{compiled_model_name}.rbln")
        rbln_config.save(save_dir_path / subfolder)

        # Save torch artifacts (e.g. embedding matrix if needed.)
        cls.save_torch_artifacts(model, save_dir_path=save_dir_path, subfolder=subfolder, rbln_config=rbln_config)

        # Load submodules
        if len(cls._rbln_submodules) > 0:
            rbln_submodules = cls._load_submodules(
                model=model,
                model_save_dir=save_dir,
                rbln_kwargs=rbln_kwargs,
                **kwargs,
            )
        else:
            rbln_submodules = []

        # Instantiate
        return cls._from_pretrained(
            model_id=save_dir_path,
            config=config,
            model_save_dir=save_dir,
            subfolder=subfolder,
            rbln_config=rbln_config,
            rbln_compiled_models=compiled_models,
            rbln_submodules=rbln_submodules,
            **kwargs,
        )

    @classmethod
    def get_pytorch_model(
        cls,
        model_id: str,
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = HUGGINGFACE_HUB_CACHE,
        subfolder: str = "",
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        # Some rbln-kwargs should be applied before loading torch module (i.e. quantized llm)
        rbln_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> "PreTrainedModel":
        kwargs = cls.update_kwargs(kwargs)
        return cls.hf_class.from_pretrained(
            model_id,
            subfolder=subfolder,
            revision=revision,
            cache_dir=cache_dir,
            use_auth_token=use_auth_token,
            local_files_only=local_files_only,
            force_download=force_download,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

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

    def forward(self, *args: List[torch.Tensor], **kwargs: Dict[str, torch.Tensor]):
        output = self.model[0](*args, **kwargs)
        return output
