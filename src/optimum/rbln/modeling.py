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

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import rebel
import torch
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from transformers import AutoConfig, PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput

from .configuration_utils import DEFAULT_COMPILED_MODEL_NAME, RBLNModelConfig
from .modeling_base import RBLNBaseModel
from .utils.logging import get_logger


if TYPE_CHECKING:
    from transformers import PreTrainedModel


logger = get_logger(__name__)


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

    output_class = None
    output_key = "last_hidden_state"

    @classmethod
    def update_kwargs(cls, kwargs):
        """
        Update user-given kwargs to get proper pytorch model.
        """
        return kwargs

    @classmethod
    def save_torch_artifacts(
        cls,
        model: "PreTrainedModel",
        save_dir_path: Path,
        subfolder: str,
        rbln_config: RBLNModelConfig,
    ):
        """
        If you are unavoidably running on a CPU rather than an RBLN device,
        store the torch tensor, weight, etc. in this function.
        """

    @classmethod
    def wrap_model_if_needed(cls, model: torch.nn.Module, rbln_config: RBLNModelConfig) -> torch.nn.Module:
        # Wrap the model if needed.
        return model

    @classmethod
    def get_compiled_model(cls, model: "PreTrainedModel", rbln_config: RBLNModelConfig):
        model = cls.wrap_model_if_needed(model, rbln_config)
        rbln_compile_config = rbln_config.compile_cfgs[0]
        compiled_model = cls.compile(model, rbln_compile_config=rbln_compile_config)
        return compiled_model

    @classmethod
    def from_model(
        cls,
        model: "PreTrainedModel",
        config: Optional[PretrainedConfig] = None,
        rbln_config: Optional[RBLNModelConfig] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        subfolder: str = "",
        **kwargs,
    ):
        preprocessors = kwargs.pop("preprocessors", [])
        rbln_config, kwargs = cls.prepare_rbln_config(rbln_config=rbln_config, **kwargs)

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
            import json

            generation_config = model.generation_config
            generation_config_path = save_dir_path / subfolder / "generation_config.json"

            generation_config.save_pretrained(generation_config_path.parent)
            local_config = json.loads(generation_config_path.read_text(encoding="utf-8"))
            local_config["transformers_version"] = generation_config.transformers_version
            generation_config_path.write_text(json.dumps(local_config, indent=2) + "\n", encoding="utf-8")

        if not isinstance(config, PretrainedConfig):  # diffusers config
            config = PretrainedConfig(**config)

        # Save preprocessor
        for preprocessor in preprocessors:
            preprocessor.save_pretrained(save_dir_path / subfolder)

        # Load submodules
        if len(cls._rbln_submodules) > 0:
            rbln_submodules = cls._load_submodules(
                model=model,
                model_save_dir=save_dir,
                rbln_config=rbln_config,
                **kwargs,
            )
        else:
            rbln_submodules = []

        # Get compilation arguments (e.g. input_info)
        rbln_config: RBLNModelConfig = cls.update_rbln_config(
            preprocessors=preprocessors, model=model, model_config=config, rbln_config=rbln_config
        )

        # torchscript should be True for jit to work
        torchscript_backup = config.torchscript
        config.torchscript = True

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

        config.torchscript = torchscript_backup
        config.save_pretrained(save_dir_path / subfolder)

        # Save torch artifacts (e.g. embedding matrix if needed.)
        cls.save_torch_artifacts(model, save_dir_path=save_dir_path, subfolder=subfolder, rbln_config=rbln_config)

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
        # Some rbln-config should be applied before loading torch module (i.e. quantized llm)
        rbln_config: Optional[RBLNModelConfig] = None,
        **kwargs,
    ) -> "PreTrainedModel":
        kwargs = cls.update_kwargs(kwargs)
        return cls.get_hf_class().from_pretrained(
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
        rbln_config: RBLNModelConfig,
    ) -> List[rebel.Runtime]:
        if DEFAULT_COMPILED_MODEL_NAME not in rbln_config.device_map:
            cls._raise_missing_compiled_file_error([DEFAULT_COMPILED_MODEL_NAME])

        return [
            rebel.Runtime(
                compiled_model,
                tensor_type="pt",
                device=rbln_config.device_map[DEFAULT_COMPILED_MODEL_NAME],
                activate_profiler=rbln_config.activate_profiler,
            )
            for compiled_model in compiled_models
        ]

    def forward(self, *args, return_dict: Optional[bool] = None, **kwargs):
        if self.hf_library_name == "transformers":
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        else:
            return_dict = True if return_dict is None else return_dict

        # Get output from the model
        output = self.model[0](*args, **kwargs)

        # Format output according to task requirements
        return self._prepare_output(output, return_dict)

    def _prepare_output(self, output, return_dict):
        """
        Prepare model output based on return_dict flag.
        This method can be overridden by subclasses to provide task-specific output handling.
        """
        if not return_dict:
            return (output,) if not isinstance(output, (tuple, list)) else output
        else:
            if self.output_class is None:
                return BaseModelOutput(last_hidden_state=output)

            # Create output with the appropriate class and key
            return self.output_class(**{self.output_key: output})
