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
import inspect
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import rebel
import torch

from .__version__ import __version__
from .utils.logging import get_logger
from .utils.runtime_utils import ContextRblnConfig


logger = get_logger(__name__)


DEFAULT_COMPILED_MODEL_NAME = "compiled_model"
DEFAULT_MOD_NAME = "default"


@dataclass
class RBLNCompileConfig:
    """
    Configuration for RBLN compilation.

    Attributes:
        compiled_model_name (str): Name of the compiled model.
        mod_name (str): Name of the RBLN module.
        input_info (List[Tuple[str, Tuple[int], Optional[str]]]): Information about input tensors.
        fusion (Optional[bool]): Whether to use fusion optimization.
        npu (Optional[str]): NPU configuration.
        tensor_parallel_size (Optional[int]): Size for tensor parallelism.
    """

    compiled_model_name: str = DEFAULT_COMPILED_MODEL_NAME
    mod_name: str = DEFAULT_MOD_NAME
    input_info: List[Tuple[str, Tuple[int], Optional[str]]] = None
    fusion: Optional[bool] = None
    npu: Optional[str] = None
    tensor_parallel_size: Optional[int] = None

    @staticmethod
    def normalize_dtype(dtype):
        """
        Convert framework-specific dtype to string representation.
        i.e. torch.float32 -> "float32"

        Args:
            dtype: The input dtype (can be string, torch dtype, or numpy dtype).

        Returns:
            str: The normalized string representation of the dtype.
        """
        if isinstance(dtype, str):
            return dtype
        else:
            dtype: str = repr(dtype).split(".")[-1]
            if dtype.endswith("'>"):  # numpy
                dtype = dtype[:-2]
            return dtype

    def __post_init__(self):
        self.input_info = [(i[0], i[1], RBLNCompileConfig.normalize_dtype(i[2]) or "float32") for i in self.input_info]

    def update(self, kwargs: Dict[str, Any]):
        self.compiled_model_name = kwargs.get("compiled_model_name", self.compiled_model_name)
        self.mod_name = kwargs.get("mod_name", self.mod_name)
        self.input_info = kwargs.get("input_info", self.input_info)
        self.fusion = kwargs.get("fusion", self.fusion)
        self.npu = kwargs.get("npu", self.npu)
        self.tensor_parallel_size = kwargs.get("tensor_parallel_size", self.tensor_parallel_size)
        return self

    def get_dummy_inputs(
        self, fill=0, static_tensors: Dict[str, torch.Tensor] = {}, meta_tensor_names: List[str] = []
    ):
        dummy = []
        for name, shape, dtype in self.input_info:
            if name in static_tensors:
                tensor = static_tensors[name]
                if shape != list(tensor.shape):
                    raise RuntimeError(f"Different shape for dummy inputs. ({shape} != {list(tensor.shape)})")
                if getattr(torch, dtype) != tensor.dtype:
                    raise RuntimeError(f"Different dtype for dummy inputs ({dtype} != {tensor.dtype})")
                dummy.append(tensor)
            else:
                if name in meta_tensor_names:
                    device = "meta"
                else:
                    device = "cpu"

                dummy.append(
                    torch.fill(torch.empty(*shape, dtype=getattr(torch, dtype), device=torch.device(device)), fill)
                    if len(shape) > 0
                    else torch.tensor(fill, dtype=getattr(torch, dtype), device=torch.device(device))
                )
        return tuple(dummy)

    def asdict(self):
        return asdict(self)


RUNTIME_KEYWORDS = ["create_runtimes", "optimize_host_memory", "device", "device_map", "activate_profiler"]


def load_config(path: str) -> Tuple[Type["RBLNModelConfig"], Dict[str, Any]]:
    path = Path(path)
    if path.is_dir():
        path = path / "rbln_config.json"

    with open(path, "r") as jsonf:
        config_file = json.load(jsonf)

    if "_meta" in config_file:
        is_legacy_rbln_config = True

        if is_legacy_rbln_config:
            raise RuntimeError(
                f"`{path}` is an old version. Please recompile the model to get the latest config file."
            )

    cls_name = config_file["cls_name"]
    cls = getattr(importlib.import_module("optimum.rbln"), cls_name)
    return cls, config_file


class RBLNAutoConfig:
    def __new__(cls, **kwargs):
        cls_name = kwargs.get("cls_name")
        if cls_name is None:
            raise ValueError("`cls_name` is required.")
        cls = getattr(importlib.import_module("optimum.rbln"), cls_name)
        return cls(**kwargs)

    @staticmethod
    def load(path: str, passed_rbln_config: Optional["RBLNModelConfig"] = None, **kwargs) -> "RBLNModelConfig":
        """
        Load RBLNModelConfig from a path.
        Class name is automatically inferred from the `rbln_config.json` file.

        Args:
            path (str): Path to the RBLNModelConfig.
            passed_rbln_config (Optional["RBLNModelConfig"]): RBLNModelConfig to be passed runtime options.

        Returns:
            RBLNModelConfig: The loaded RBLNModelConfig.
        """
        cls, config_file = load_config(path)

        rbln_keys = [key for key in kwargs.keys() if key.startswith("rbln_")]

        rbln_runtime_kwargs = {key[5:]: kwargs.pop(key) for key in rbln_keys if key[5:] in RUNTIME_KEYWORDS}
        rbln_kwargs = {
            key[5:]: kwargs.pop(key)
            for key in rbln_keys
            if key[5:] not in RUNTIME_KEYWORDS and key[5:] not in cls.submodules
        }

        if len(rbln_kwargs) > 0:
            raise ValueError(f"Cannot set the following arguments: {list(rbln_kwargs.keys())}")

        config_file.update(rbln_runtime_kwargs)

        if passed_rbln_config is not None:
            for key, value in passed_rbln_config._runtime_options.items():
                if key in config_file:
                    raise ValueError(f"Already set runtime option: {key}")
                config_file[key] = value

        return cls(**config_file)


class RBLNModelConfig:
    non_save_attributes = [
        "_frozen",
        "_runtime_options",
        "npu",
        "tensor_parallel_size",
        "create_runtimes",
        "optimize_host_memory",
        "device",
        "device_map",
        "activate_profiler",
    ]
    submodules: List[str] = []

    def init_submodule_config(
        self,
        submodule_config_cls: Type["RBLNModelConfig"],
        submodule_config: Optional[Union[Dict[str, Any], "RBLNModelConfig"]] = None,
        **kwargs,
    ) -> "RBLNModelConfig":
        """
        Initialize a submodule config from a dict or a RBLNModelConfig.

        kwargs is specified from the predecessor config.
        """
        if submodule_config is None:
            submodule_config = {}

        if isinstance(submodule_config, dict):
            from_predecessor = self._runtime_options.copy()
            from_predecessor.update(kwargs)
            init_kwargs = from_predecessor
            init_kwargs.update(submodule_config)
            submodule_config = submodule_config_cls(**init_kwargs)

        if not isinstance(submodule_config, submodule_config_cls):
            raise TypeError(f"Invalid submodule config type: {type(submodule_config)}")

        return submodule_config

    def __setattr__(self, key, value):
        if key != "_attributes_map" and key not in self.non_save_attributes:
            self._attributes_map[key] = value

        if hasattr(self, "_frozen") and self._frozen:
            if not hasattr(self, key) or getattr(self, key) != value:
                raise RuntimeError(
                    f"`{self.__class__.__name__}` is frozen. Cannot update or set attribute after freezing."
                )

        # If the submodule is a dict, Instantiate the submodule config class
        if key in self.submodules and isinstance(value, dict) and (cls_name := value.get("cls_name")):
            rbln_config_cls = getattr(importlib.import_module("optimum.rbln"), cls_name)
            value = rbln_config_cls(**value)

        # Forbid setting keyword-only arguments
        # keyword-only arguments should be translated to other attributes, not set directly
        _keyword_only_args = set()
        init_signature = inspect.signature(self.__class__.__init__)
        for param_name, param in init_signature.parameters.items():
            if param.kind == inspect.Parameter.KEYWORD_ONLY:
                _keyword_only_args.add(param_name)

        if key in _keyword_only_args:
            raise AttributeError(
                f"Cannot set attribute '{key}'. This is an internal error. Please report it to the developers."
            )

        super().__setattr__(key, value)

    def __init__(
        self,
        cls_name: Optional[str] = None,
        create_runtimes: Optional[bool] = None,
        optimize_host_memory: Optional[bool] = None,
        device: Optional[Union[int, List[int]]] = None,
        device_map: Optional[Dict[str, Union[int, List[int]]]] = None,
        activate_profiler: Optional[bool] = None,
        npu: Optional[str] = None,
        tensor_parallel_size: Optional[int] = None,
        optimum_rbln_version: Optional[str] = None,
        _compile_cfgs: List[RBLNCompileConfig] = [],
        **kwargs,
    ):
        self._attributes_map = {}
        self._frozen = False

        self.cls_name = cls_name
        if self.cls_name is None:
            self.cls_name = self.__class__.__name__

        self._runtime_options = {}
        self._runtime_options["create_runtimes"] = create_runtimes
        self._runtime_options["optimize_host_memory"] = optimize_host_memory
        self._runtime_options["device"] = device
        self._runtime_options["device_map"] = device_map
        self._runtime_options["activate_profiler"] = activate_profiler

        # Automatically pass npu, tensor_parallel_size to compile_cfgs
        self.npu = npu
        self.tensor_parallel_size = tensor_parallel_size

        self.optimum_rbln_version = optimum_rbln_version
        if self.optimum_rbln_version is None:
            self.optimum_rbln_version = __version__

        self._compile_cfgs: List[RBLNCompileConfig] = _compile_cfgs

        if not isinstance(self._compile_cfgs, list):
            raise ValueError("`compile_cfgs` must be a list of `RBLNCompileConfig`.")
        if len(self._compile_cfgs) > 0 and not isinstance(self._compile_cfgs[0], RBLNCompileConfig):
            self.set_compile_cfgs([RBLNCompileConfig(**cfg) for cfg in self._compile_cfgs])

        if len(kwargs) > 0:
            raise ValueError(f"Unexpected arguments: {kwargs.keys()}")

    @property
    def rbln_model_cls_name(self) -> str:
        return self.__class__.__name__[:-6]

    @property
    def rbln_model_cls(self) -> Type:
        rbln_model_cls = getattr(importlib.import_module("optimum.rbln"), self.rbln_model_cls_name, None)
        if rbln_model_cls is None:
            raise ValueError(
                f"RBLN model class {self.rbln_model_cls_name} not found. This is an internal error. "
                "Please report it to the developers."
            )
        return rbln_model_cls

    def _prepare_for_serialization(self):
        """
        Prepare the attributes map for serialization by converting nested RBLNModelConfig
        objects to their serializable form.
        """
        serializable_map = {}
        for key, value in self._attributes_map.items():
            if isinstance(value, RBLNModelConfig):
                # Convert nested RBLNModelConfig to its serializable form
                serializable_map[key] = value._prepare_for_serialization()
            elif key == "_compile_cfgs":
                serializable_map[key] = [cfg.asdict() for cfg in value]
            else:
                serializable_map[key] = value
        return serializable_map

    def __repr__(self):
        repr_dict = self._prepare_for_serialization()
        return json.dumps(repr_dict, indent=2)

    @property
    def compile_cfgs(self):
        return self._compile_cfgs

    @compile_cfgs.setter
    def compile_cfgs(self, compile_cfgs: List[RBLNCompileConfig]):
        raise RuntimeError("`compile_cfgs` cannot be set directly. Please use `set_compile_cfgs` instead.")

    def set_compile_cfgs(self, compile_cfgs: List[RBLNCompileConfig]):
        if not isinstance(compile_cfgs, list):
            raise ValueError("`compile_cfgs` must be a list of `RBLNCompileConfig`.")
        if len(compile_cfgs) == 0:
            raise ValueError("`compile_cfgs` must contain at least one `RBLNCompileConfig`.")
        if not isinstance(compile_cfgs[0], RBLNCompileConfig):
            raise ValueError("`compile_cfgs` must contain only `RBLNCompileConfig`.")

        self._compile_cfgs = compile_cfgs
        for compile_cfg in self._compile_cfgs:
            compile_cfg.npu = self.npu
            compile_cfg.tensor_parallel_size = self.tensor_parallel_size

    def freeze(self):
        if self._frozen:
            raise RuntimeError(f"`{self.__class__.__name__}` is already frozen.")

        if (
            not isinstance(self._compile_cfgs, list)
            or len(self._compile_cfgs) == 0
            or not all(isinstance(cfg, RBLNCompileConfig) for cfg in self._compile_cfgs)
        ):
            raise RuntimeError("`compile_cfgs` must be set before freezing.")

        for submodule_name in self.submodules:
            submodule_config = getattr(self, submodule_name, None)
            if not isinstance(submodule_config, RBLNModelConfig):
                raise ValueError(f"`{submodule_name}` must be an instance of `RBLNModelConfig` before freezing.")

            if not submodule_config.is_frozen():
                raise ValueError(f"`{submodule_name}` config must be frozen before freezing super config.")

        self._frozen = True

    def is_frozen(self):
        return self._frozen

    def save(self, path: str):
        if not self._frozen:
            raise RuntimeError("`RBLNModelConfig` is not frozen. Please call `set_compile_cfgs` first.")

        # save as json file without runtime attributes
        path = Path(path)
        if path.is_dir():
            path = path / "rbln_config.json"

        with open(path, "w") as jsonf:
            serializable_data = self._prepare_for_serialization()
            json.dump(serializable_data, jsonf, indent=2)

    @classmethod
    def load(cls, path: str, **kwargs) -> "RBLNModelConfig":
        cls_reserved, config_file = load_config(path)

        if cls_reserved != cls:
            logger.warning(f"Expected {cls.__name__}, but got {cls_reserved.__name__}.")

        rbln_keys = [key for key in kwargs.keys() if key.startswith("rbln_")]
        rbln_kwargs = {key[5:]: kwargs.pop(key) for key in rbln_keys}
        config_file.update(rbln_kwargs)

        return cls(**config_file)

    @classmethod
    def initialize_from_kwargs(
        cls: Type["RBLNModelConfig"],
        rbln_config: Optional[Union[Dict[str, Any], "RBLNModelConfig"]] = None,
        **kwargs,
    ) -> Tuple["RBLNModelConfig", Dict[str, Any]]:
        """
        Initialize RBLNModelConfig from kwargs.
        """
        kwargs_keys = list(kwargs.keys())
        rbln_kwargs = {key[5:]: kwargs.pop(key) for key in kwargs_keys if key.startswith("rbln_")}

        if isinstance(rbln_config, dict):
            rbln_config.update(rbln_kwargs)
            rbln_config = cls(**rbln_config)

        elif rbln_config is None:
            rbln_config = cls(**rbln_kwargs)

        elif isinstance(rbln_config, RBLNModelConfig):
            for key, value in rbln_kwargs.items():
                setattr(rbln_config, key, value)

        return rbln_config, kwargs

    @property
    def create_runtimes(self):
        context = ContextRblnConfig.get_current_context()["create_runtimes"]
        if context is not None:
            return context
        elif self._runtime_options["create_runtimes"] is None:
            return rebel.npu_is_available()
        return self._runtime_options["create_runtimes"]

    @create_runtimes.setter
    def create_runtimes(self, create_runtimes: bool):
        self._runtime_options["create_runtimes"] = create_runtimes

    @property
    def optimize_host_memory(self):
        context = ContextRblnConfig.get_current_context()["optimize_host_memory"]
        if context is not None:
            return context
        elif self._runtime_options["optimize_host_memory"] is None:
            return True
        return self._runtime_options["optimize_host_memory"]

    @optimize_host_memory.setter
    def optimize_host_memory(self, optimize_host_memory: bool):
        self._runtime_options["optimize_host_memory"] = optimize_host_memory

    @property
    def device(self):
        context = ContextRblnConfig.get_current_context()["device"]
        if context is not None:
            return context
        return self._runtime_options["device"]

    @device.setter
    def device(self, device: Union[int, List[int]]):
        self._runtime_options["device"] = device

    @property
    def device_map(self):
        context = ContextRblnConfig.get_current_context()["device_map"]
        if context:
            return context
        elif self._runtime_options["device_map"] is None:
            rbln_device_map = {}
            device_val = self.device
            for cfg in self.compile_cfgs:
                rbln_device_map[cfg.compiled_model_name] = device_val
            return rbln_device_map
        return self._runtime_options["device_map"]

    @device_map.setter
    def device_map(self, device_map: Dict[str, Union[int, List[int]]]):
        self._runtime_options["device_map"] = device_map

    @property
    def activate_profiler(self):
        context = ContextRblnConfig.get_current_context()["activate_profiler"]
        if context is not None:
            return context
        return self._runtime_options["activate_profiler"]

    @activate_profiler.setter
    def activate_profiler(self, activate_profiler: bool):
        self._runtime_options["activate_profiler"] = activate_profiler
