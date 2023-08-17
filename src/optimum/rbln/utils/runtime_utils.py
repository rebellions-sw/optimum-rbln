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

import threading
from typing import Any, Dict, List

import rebel
import torch


class RBLNPytorchRuntime:
    mandatory_members = []

    def __init__(self, runtime: rebel.Runtime, **kwargs) -> None:
        self.runtime = runtime
        for key, value in kwargs.items():
            setattr(self, key, value)
        for mandatory_member in self.mandatory_members:
            if mandatory_member not in kwargs:
                raise AttributeError(f"`{mandatory_member}` should be assigned to {self.__class__.__name__} objects.")

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

    def forward(self, *args: List["torch.Tensor"], **kwargs: Dict[str, "torch.Tensor"]):
        # filtering useless args or kwarg such as None.
        args = list(filter(lambda arg: isinstance(arg, torch.Tensor), args))
        kwargs = dict(filter(lambda kwarg: isinstance(kwarg[1], torch.Tensor) or kwarg[0] == "out", kwargs.items()))
        output = self.runtime(*args, **kwargs)
        return output

    def __repr__(self) -> str:
        return repr(self.runtime)


class UnavailableRuntime:
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise self.forward(*args, **kwargs)

    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx: int) -> Any:
        return self

    def __iter__(self):
        return iter([self])

    def forward(self, *args: List["torch.Tensor"], **kwargs: Dict[str, "torch.Tensor"]):
        raise RuntimeError("The model can't run because the runtime hasn't been created.")

    def __repr__(self) -> str:
        return "UnavailableRuntime"


class ContextRblnConfig:
    _local = threading.local()

    def __init__(
        self, device=None, device_map=None, create_runtimes=None, optimize_host_mem=None, activate_profiler=None
    ):
        self.device = device
        self.device_map = device_map
        self.create_runtimes = create_runtimes
        self.optimize_host_mem = optimize_host_mem
        self.activate_profiler = activate_profiler

    def __enter__(self):
        self._local.device = self.device
        self._local.device_map = self.device_map
        self._local.create_runtimes = self.create_runtimes
        self._local.optimize_host_memory = self.optimize_host_mem
        self._local.activate_profiler = self.activate_profiler
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._local.device = None
        self._local.device_map = None
        self._local.create_runtimes = None
        self._local.optimize_host_memory = None
        self._local.activate_profiler = None

    @classmethod
    def get_current_context(cls):
        return {
            "device": getattr(cls._local, "device", None),
            "device_map": getattr(cls._local, "device_map", None),
            "create_runtimes": getattr(cls._local, "create_runtimes", None),
            "optimize_host_memory": getattr(cls._local, "optimize_host_memory", None),
            "activate_profiler": getattr(cls._local, "activate_profiler", None),
        }
