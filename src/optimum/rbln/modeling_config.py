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
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import rebel
import torch

from .__version__ import __version__
from .utils.runtime_utils import ContextRblnConfig


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
COMPILE_KEYWORDS = ["compiled_model_name", "mod_name", "input_info", "fusion", "npu", "tensor_parallel_size"]


class RBLNConfig:
    """
    Configuration for single RBLN OptimizedModel, representing multiple compiled models.

    Attributes:
        compile_cfgs (List[RBLNCompileConfig]): Compilation configurations.
        meta (dict): Metadata including version and class information.
        runtime_cfg (dict): Runtime-specific configuration.
    """

    # It represents multiple compiled model, one of each can have multiple runtimes.
    def __init__(
        self,
        rbln_cls,
        compile_cfgs: List[RBLNCompileConfig],
        rbln_kwargs=None,
        meta=None,
    ) -> None:
        if rbln_kwargs is None:
            rbln_kwargs = {}
        else:
            rbln_kwargs = copy.deepcopy(rbln_kwargs)

        # meta : class, version and other informations.
        if meta is None:
            self.meta = {"version": __version__, "cls": rbln_cls}
        else:
            self.meta = meta

        # compile_cfgs : compile args for each runtime
        self.compile_cfgs = compile_cfgs
        for compile_cfg in self.compile_cfgs:
            compile_cfg.update(rbln_kwargs)
        for K in COMPILE_KEYWORDS:
            rbln_kwargs.pop(K, None)

        # runtime_cfg : Values that don't be saved / loaded.
        self.runtime_cfg = {}
        for runtime_key in RUNTIME_KEYWORDS:
            if runtime_key in rbln_kwargs:
                self.runtime_cfg[runtime_key] = rbln_kwargs.pop(runtime_key)

        # model_cfg : All user-provided values such as "max_seq_len".
        self.model_cfg: Dict[str, Any] = rbln_kwargs

    def save(self, dir_path: str):
        dir_path = Path(dir_path)

        s_json = {}
        compile_cfgs = [asdict(cfg) for cfg in self.compile_cfgs]
        s_json["_compile_cfgs"] = compile_cfgs
        s_json["_meta"] = self.meta
        s_json.update(self.model_cfg)

        with open(dir_path / "rbln_config.json", "w") as jsonf:
            json.dump(s_json, jsonf, indent=2)

    @classmethod
    def load(cls, dir_path: str) -> "RBLNConfig":
        dir_path = Path(dir_path)
        with open(dir_path / "rbln_config.json", "r") as jsonf:
            config_file = json.load(jsonf)

        return cls.fromdict(config_file)

    @classmethod
    def fromdict(cls, dic: dict):
        compile_cfgs = dic.pop("_compile_cfgs")
        compile_cfgs = [RBLNCompileConfig(**cfg) for cfg in compile_cfgs]

        meta = dic.pop("_meta")
        rbln_cls = meta["cls"]

        rbln_kwargs = dic
        return cls(rbln_cls=rbln_cls, compile_cfgs=compile_cfgs, rbln_kwargs=rbln_kwargs, meta=meta)

    def update_runtime_cfg(self, rbln_kwargs: Dict[str, Any]):
        keys = list(rbln_kwargs.keys())
        for key in keys:
            if key in RUNTIME_KEYWORDS:
                self.runtime_cfg[key] = rbln_kwargs[key]

    def __repr__(self):
        compile_cfgs_repr = [f"\n    {cfg!r}" for cfg in self.compile_cfgs]
        return (
            f"RBLNConfig(\n"
            f"  rbln_cls={self.meta['cls']},\n"
            f"  version='{self.meta['version']}',\n"
            f"  compile_cfgs=[{''.join(compile_cfgs_repr)}\n  ],\n"
            f"  model_cfg={self.model_cfg},\n"
            f"  runtime_cfg={self.runtime_cfg}\n"
            f")"
        )

    @property
    def create_runtimes(self):
        context = ContextRblnConfig.get_current_context()["create_runtimes"]
        if context is not None:
            return context
        elif self.runtime_cfg.get("create_runtimes", None) is None:
            return rebel.npu_is_available()
        return self.runtime_cfg["create_runtimes"]

    @property
    def optimize_host_memory(self):
        context = ContextRblnConfig.get_current_context()["optimize_host_memory"]
        if context is not None:
            return context
        elif self.runtime_cfg.get("optimize_host_memory", None) is None:
            return True
        return self.runtime_cfg["optimize_host_memory"]

    @property
    def device(self):
        context = ContextRblnConfig.get_current_context()["device"]
        if context:
            return context
        elif self.runtime_cfg.get("device", None) is None:
            return 0
        return self.runtime_cfg["device"]

    @property
    def device_map(self):
        context = ContextRblnConfig.get_current_context()["device_map"]
        if context:
            return context
        elif self.runtime_cfg.get("device_map", None) is None:
            rbln_device_map = {}
            device_val = self.device
            for cfg in self.compile_cfgs:
                rbln_device_map[cfg.compiled_model_name] = device_val
            return rbln_device_map
        return self.runtime_cfg["device_map"]

    @property
    def activate_profiler(self):
        context = ContextRblnConfig.get_current_context()["activate_profiler"]
        if context:
            return context
        elif self.runtime_cfg.get("activate_profiler", None) is None:
            return False
        return self.runtime_cfg["activate_profiler"]


def use_rbln_config(fn):
    """
    If the function uses rbln_config and kwargs,
        then extract `rbln_` prefix from kwargs.

    If rbln_config is already an instance of RBLNConfig, then pass.
    """

    def merged_rbln_config_fn(*args, **kwargs):
        rbln_kwargs = kwargs.pop("rbln_kwargs", None)
        if rbln_kwargs is not None:
            raise KeyError("`rbln_kwargs` cannot be specified when using `rbln_config`!")

        rbln_config = kwargs.pop("rbln_config", None)

        keys = list(kwargs.keys())
        rbln_kwargs = {key[5:]: kwargs.pop(key) for key in keys if key.startswith("rbln_")}

        if isinstance(rbln_config, RBLNConfig):
            # merge runtime kwargs if exists.
            runtime_rbln_kwargs = {k: rbln_kwargs.pop(k) for k in RUNTIME_KEYWORDS if k in rbln_kwargs}

            # ignore internal keys and recover "rbln_" prefix
            RBLN_INTERNAL_KEYS = {"compiled_models", "submodules"}
            internal_kwargs = {"rbln_" + k: rbln_kwargs.pop(k) for k in RBLN_INTERNAL_KEYS if k in rbln_kwargs}

            if len(rbln_kwargs) > 0:
                raise KeyError(
                    f"Failed to merging function argument : {rbln_kwargs.keys()}. "
                    "If you passed `rbln_config` an instance of `RBLNConfig`, "
                    "then none `rbln_` prefixes are allowed to be passed."
                )
            rbln_config.update_runtime_cfg(runtime_rbln_kwargs)
            return fn(*args, **kwargs, **internal_kwargs, rbln_config=rbln_config)

        elif rbln_config is None:
            rbln_config_dict = {}

        else:
            rbln_config_dict = rbln_config

        for key in rbln_config_dict:
            if key in rbln_kwargs:
                raise KeyError(f"Duplicated key in both `rbln_config` and rbln_{key}.")

        rbln_kwargs.update(rbln_config_dict)
        return fn(*args, **kwargs, rbln_config=rbln_kwargs)

    return merged_rbln_config_fn
