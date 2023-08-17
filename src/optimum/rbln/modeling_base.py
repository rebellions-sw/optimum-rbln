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

import importlib
import logging
import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import rebel
import torch
from transformers import (
    AutoConfig,
    AutoModel,
    GenerationConfig,
    PretrainedConfig,
)

from .modeling_config import RBLNCompileConfig, RBLNConfig, use_rbln_config
from .utils.hub import PushToHubMixin, pull_compiled_model_from_hub, validate_files
from .utils.runtime_utils import UnavailableRuntime
from .utils.save_utils import maybe_load_preprocessors
from .utils.submodule import SubModulesMixin


if TYPE_CHECKING:
    from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


class PreTrainedModel(ABC):  # noqa: F811
    pass


class RBLNBaseModel(SubModulesMixin, PushToHubMixin, PreTrainedModel):
    """
    An abstract base class for compiling, loading, and saving neural network models from the huggingface
    transformers and diffusers libraries to run on RBLN NPU devices.

    This class supports loading and saving models using the `from_pretrained` and `save_pretrained` methods,
    similar to the huggingface libraries.

    The `from_pretrained` method loads a model corresponding to the given `model_id` from a local repository
    or the huggingface hub onto the NPU. If the model is a PyTorch model and `export=True` is passed as a
    kwarg, it compiles the PyTorch model corresponding to the given `model_id` before loading. If `model_id`
    is an already rbln-compiled model, it can be directly loaded onto the NPU with `export=False`.

    `rbln_npu` is a kwarg required for compilation, specifying the name of the NPU to be used. If this
    keyword is not specified, the NPU installed on the host machine is used. If no NPU is installed on the
    host machine, an error occurs.

    `rbln_device` specifies the device to be used at runtime. If not specified, device 0 is used.

    `rbln_create_runtimes` indicates whether to create runtime objects. If False, the runtime does not load
    the model onto the NPU. This option is particularly useful when you want to perform compilation only on a
    host machine without an NPU.

    `RBLNModel`, `RBLNModelFor*`, etc. are all child classes of RBLNBaseModel.

    Models compiled in this way can be saved to a local repository using `save_pretrained` or uploaded to
    the huggingface hub.

    It also supports generation through `generate` (for transformers models that support generation).

    RBLNBaseModel is a class for models consisting of an arbitrary number of `torch.nn.Module`s, and
    therefore is an abstract class without explicit implementations of `forward` or `export` functions.
    To inherit from this class, `forward`, `export`, etc. must be implemented.
    """

    model_type = "rbln_model"
    auto_model_class = AutoModel
    config_class = AutoConfig
    config_name = "config.json"
    hf_library_name = "transformers"
    _hf_class = None

    def __init__(
        self,
        models: List[rebel.Runtime],
        config: "PretrainedConfig",
        rbln_config: RBLNConfig,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        subfolder: str = "",
        rbln_compiled_models: Optional[rebel.RBLNCompiledModel] = None,
        rbln_submodules: List["RBLNBaseModel"] = [],
        **kwargs,
    ):
        self.model = models
        self.config = config
        self.rbln_config = rbln_config
        self.compiled_models = rbln_compiled_models

        # Registers the RBLN classes into the transformers AutoModel classes to avoid warnings when creating
        # a pipeline https://github.com/huggingface/transformers/blob/3d3204c025b6b5de013e07dd364208e28b4d9589/src/transformers/pipelines/base.py#L940
        AutoConfig.register(self.model_type, AutoConfig)
        if hasattr(self.auto_model_class, "register"):
            self.auto_model_class.register(AutoConfig, self.__class__)

        # copied from tranformers PreTrainedModel __init__
        if self.can_generate():
            gen_config_dir = model_save_dir.name if isinstance(model_save_dir, TemporaryDirectory) else model_save_dir
            self.generation_config = GenerationConfig.from_pretrained(gen_config_dir, trust_remote_code=True)
        else:
            self.generation_config = None

        # self.generation_config = GenerationConfig.from_model_config(config) if self.can_generate() else None
        if self.generation_config is not None:
            self.generation_config.use_cache = True

        self.device = torch.device("cpu")
        self.training = False
        self.dtype = torch.float32

        # FIXME :: model_save_dir is not used after initialized. (This can be used when save/load)
        # This attribute is needed to keep one reference on the temporary directory, since garbage collecting it
        # would end-up removing the directory containing the underlying RBLN model.
        self._model_save_dir_tempdirectory_instance = None
        if isinstance(model_save_dir, TemporaryDirectory):
            self._model_save_dir_tempdirectory_instance = model_save_dir
            self.model_save_dir = Path(model_save_dir.name)
        elif isinstance(model_save_dir, str):
            self.model_save_dir = Path(model_save_dir)
        else:
            self.model_save_dir = model_save_dir
        self.subfolder = subfolder

        self.rbln_submodules = rbln_submodules
        self.__post_init__(**kwargs)

    @classmethod
    def _load_compiled_model_dir(
        cls,
        model_id: Union[str, Path],
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        subfolder: str = "",
        local_files_only: bool = False,
    ) -> str:
        """Load the directory containing the compiled model files."""
        model_path = Path(model_id)

        if model_path.is_dir():
            model_path = model_path / subfolder
            rbln_files = list(model_path.glob("*.rbln"))
            rbln_config_filenames = list(model_path.glob("rbln_config.json"))
            validate_files(rbln_files, rbln_config_filenames, f"directory {model_path}")
        else:
            model_path = pull_compiled_model_from_hub(
                model_id=model_id,
                subfolder=subfolder,
                use_auth_token=use_auth_token,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
            )

        return str(model_path)

    @classmethod
    def _load_compiled_models(cls, model_path: str):
        compiled_models = Path(model_path).glob("*.rbln")
        rbln_compiled_models = {cm.stem: rebel.RBLNCompiledModel(cm) for cm in compiled_models}
        return rbln_compiled_models

    @classmethod
    @use_rbln_config
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: "PretrainedConfig" = None,
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        subfolder: str = "",
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        # passed from compile function
        rbln_config: Optional[RBLNConfig] = None,
        rbln_compiled_models: Optional[Dict[str, rebel.RBLNCompiledModel]] = None,
        rbln_submodules: List["RBLNBaseModel"] = [],
        **kwargs,
    ) -> "RBLNBaseModel":
        from_export_method = isinstance(rbln_config, RBLNConfig) and rbln_compiled_models is not None

        if not from_export_method:
            # from compiled dir
            rbln_kwargs = rbln_config or {}

            model_path_subfolder = cls._load_compiled_model_dir(
                model_id=model_id,
                use_auth_token=use_auth_token,
                revision=revision,
                force_download=force_download,
                cache_dir=cache_dir,
                subfolder=subfolder,
                local_files_only=local_files_only,
            )

            rbln_config = RBLNConfig.load(model_path_subfolder)
            rbln_config.update_runtime_cfg(rbln_kwargs)

            if rbln_config.meta["cls"] != cls.__name__:
                raise NameError(
                    f"Cannot load the model. The model was originally compiled using "
                    f"{rbln_config.meta['cls']}, but you are trying to load it with {cls.__name__}."
                    "Please use the same model class that was used during compilation."
                )

            if config is None:
                if cls.hf_library_name == "transformers":
                    config = AutoConfig.from_pretrained(
                        model_path_subfolder,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        revision=revision,
                        token=use_auth_token,
                        trust_remote_code=trust_remote_code,
                    )
                elif cls.hf_library_name == "diffusers":
                    # import here to prevent diffusers dependency
                    # TODO(jongho): Remove diffusers dependency if use transformers only.
                    from diffusers.configuration_utils import ConfigMixin

                    class DummyConfigMixin(ConfigMixin):
                        # Just to load config, We need to specify `config_name`
                        config_name = "config.json"

                    config = DummyConfigMixin.load_config(
                        model_id,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        local_files_only=local_files_only,
                        revision=revision,
                        token=use_auth_token,
                        subfolder=subfolder,
                    )
                    config = PretrainedConfig(**config)

            rbln_compiled_models = cls._load_compiled_models(model_path_subfolder)

            if len(cls._rbln_submodules) > 0:
                rbln_submodules = cls._load_submodules(
                    model_save_dir=model_id,
                    rbln_kwargs=rbln_kwargs,
                    **kwargs,
                )
            else:
                rbln_submodules = []

            if subfolder != "":
                model_save_dir = Path(model_path_subfolder).absolute().parent
            else:
                model_save_dir = Path(model_path_subfolder).absolute()

        return cls._from_compiled_models(
            rbln_compiled_models=rbln_compiled_models,
            rbln_config=rbln_config,
            config=config,
            model_save_dir=model_save_dir,
            subfolder=subfolder,
            rbln_submodules=rbln_submodules,
            **kwargs,
        )

    @classmethod
    def _from_compiled_models(
        cls,
        rbln_compiled_models: Dict[str, rebel.RBLNCompiledModel],
        rbln_config: RBLNConfig,
        config: "PretrainedConfig",
        model_save_dir: Union[Path, str],
        subfolder: Union[Path, str],
        rbln_submodules: List["RBLNBaseModel"] = [],
        **kwargs,
    ):
        if isinstance(model_save_dir, str):
            model_save_dir = Path(model_save_dir)
        # FIXME:: Should we convert it?
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

        return cls(
            models,
            config,
            rbln_config,
            model_save_dir=model_save_dir,
            subfolder=subfolder,
            rbln_compiled_models=(None if rbln_config.optimize_host_memory else rbln_compiled_models),
            rbln_submodules=rbln_submodules,
            **kwargs,
        )

    @classmethod
    @use_rbln_config
    def _export(
        cls,
        model_id: Union[str, Path],
        rbln_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> "RBLNBaseModel":
        subfolder = kwargs.get("subfolder", "")
        model_save_dir = kwargs.pop("model_save_dir", None)

        rbln_kwargs = rbln_config
        model: "PreTrainedModel" = cls.get_pytorch_model(
            model_id=model_id,
            rbln_kwargs=rbln_kwargs,
            **kwargs,
        )
        preprocessors = maybe_load_preprocessors(model_id, subfolder=subfolder)
        return cls.from_model(
            model,
            rbln_config=rbln_config,
            preprocessors=preprocessors,
            model_save_dir=model_save_dir,
            **kwargs,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_id: Union[str, Path],
        export: bool = False,
        **kwargs,
    ) -> "RBLNBaseModel":
        if isinstance(model_id, Path):
            model_id = model_id.as_posix()
        from_pretrained_method = cls._export if export else cls._from_pretrained
        return from_pretrained_method(model_id=model_id, **kwargs)

    @classmethod
    def compile(cls, model, rbln_compile_config: Optional[RBLNCompileConfig] = None, **kwargs):
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
    def get_rbln_config(
        cls,
        rbln_kwargs: Dict[str, Any],
        **others,
    ) -> RBLNConfig:
        """
        Make default rbln-config for the model.
        kwargs for overriding model's config can be accepted.
        Note that batch_size should be specified with proper input_info.
        """
        rbln_config = cls._get_rbln_config(**others, rbln_kwargs=rbln_kwargs)
        return rbln_config

    @classmethod
    @property
    def hf_class(cls):
        """
        Lazily loads and caches the corresponding Hugging Face model class.
        Removes 'RBLN' prefix from the class name to get the original class name
        (e.g., RBLNLlamaForCausalLM -> LlamaForCausalLM) and imports it from
        the transformers/diffusers module.

        Returns:
            type: The original Hugging Face model class
        """
        if cls._hf_class is None:
            hf_cls_name = cls.__name__[4:]
            library = importlib.import_module(cls.hf_library_name)
            cls._hf_class = getattr(library, hf_cls_name, None)
        return cls._hf_class

    def can_generate(self):
        return False

    def to(self, *args, **kwargs):
        return self

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __repr__(self):
        return repr(self.model) + repr(self.rbln_submodules)

    def __post_init__(self, **kwargs):
        pass

    def save_pretrained(
        self,
        save_directory: Union[str, Path],
        push_to_hub: bool = False,
        **kwargs,
    ):
        """
        Saves a model and its configuration file to a directory, so that it can be re-loaded using the
        [`~optimum.rbln.modeling_base.RBLNBaseModel.from_pretrained`] class method.

        Args:
            save_directory (`Union[str, Path]`):
                Directory where to save the model file.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it.

        """
        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        os.makedirs(save_directory, exist_ok=True)

        real_save_dir = self.model_save_dir / self.subfolder
        save_directory_path = Path(save_directory)
        if os.path.exists(real_save_dir) and os.path.isdir(real_save_dir):
            if save_directory_path.absolute() == real_save_dir.absolute():
                raise FileExistsError(
                    f"Cannot save model to '{save_directory}'. "
                    f"This directory already exists and contains the model files."
                )
            shutil.copytree(real_save_dir, save_directory, dirs_exist_ok=True)
            self.config.save_pretrained(save_directory)
            if self.generation_config is not None:
                self.generation_config.save_pretrained(save_directory)
        else:
            raise FileNotFoundError(
                f"Unable to save the model. The model directory '{real_save_dir}' does not exist or is not accessible. "
                f"Cannot save to the specified destination '{save_directory}'. "
                f"Please ensure the model directory exists and you have the necessary permissions to access it."
            )

        if push_to_hub:
            return super().push_to_hub(save_directory, **kwargs)

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

    @classmethod
    @abstractmethod
    def _get_rbln_config(cls, **rbln_config_kwargs) -> RBLNConfig:
        pass

    @classmethod
    @abstractmethod
    def _create_runtimes(
        cls,
        compiled_models: List[rebel.RBLNCompiledModel],
        rbln_device_map: Dict[str, int],
        activate_profiler: Optional[bool] = None,
    ) -> List[rebel.Runtime]:
        # compiled_models -> runtimes
        pass

    @classmethod
    @abstractmethod
    def get_pytorch_model(cls, *args, **kwargs):
        pass

    @classmethod
    @abstractmethod
    @use_rbln_config
    def from_model(
        cls,
        model: "PreTrainedModel",
        rbln_config: Dict[str, Any] = {},
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        subfolder: str = "",
        **kwargs,
    ):
        pass

    @abstractmethod
    def forward(self, *args: List[torch.Tensor], **kwargs: Dict[str, torch.Tensor]):
        pass
