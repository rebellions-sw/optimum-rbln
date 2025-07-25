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

import re
import threading
from typing import Any, List, Optional, Union

import rebel
import torch


def normalize_npu(npu: str) -> str:
    """Normalize the NPU string by removing the form factor."""
    match = re.match(r"(RBLN-CA|RBLN-CR)(\d+)", npu)
    if match:
        prefix, num = match.groups()
        if len(num) == 1:
            # Convert "RBLN-CAx" → "RBLN-CA0"
            # (e.g., "RBLN-CA2" -> "RBLN-CA0")
            npu = f"{prefix}0"
        elif len(num) == 2:
            # Strip form factor (e.g., "RBLN-CA15" → "RBLN-CA1")
            npu = f"{prefix}{num[:-1]}"
    return npu


def tp_and_devices_are_ok(
    tensor_parallel_size: Optional[int] = None,
    device: Optional[Union[int, List[int]]] = None,
    npu: Optional[str] = None,
) -> Optional[str]:
    if tensor_parallel_size is None:
        tensor_parallel_size = 1

    if rebel.device_count() < tensor_parallel_size:
        return (
            f"Tensor parallel size {tensor_parallel_size} is greater than "
            f"the number of available devices {rebel.device_count()}."
        )

    if device is None:
        device = list(range(tensor_parallel_size))
    elif isinstance(device, int):
        device = [device]
    elif isinstance(device, list):
        if any(not isinstance(d, int) for d in device):
            return "Device must be a(n) (list of) integer(s)."
        if len(device) != tensor_parallel_size:
            return (
                f"The number of devices ({len(device)}) does not match tensor parallel size ({tensor_parallel_size})."
            )
    else:
        return f"Invalid device: {device}"

    for device_id in device:
        if device_id < 0:  # if any device is dummy device, skip it
            return None
        if rebel.get_npu_name(device_id) is None:
            return (
                f"Device {device_id} is not a valid NPU device. Please check your NPU status with 'rbln-stat' command."
            )

    if npu is not None:
        for device_id in device:
            npu_name = rebel.get_npu_name(device_id)
            if normalize_npu(npu_name) != normalize_npu(npu):
                return f"Device {device_id} ({npu_name}) is not on the same NPU as {npu}."

    return None


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

    def forward(self, *args: List["torch.Tensor"], **kwargs: "torch.Tensor"):
        # filtering useless args or kwarg such as None.
        args = list(filter(lambda arg: isinstance(arg, torch.Tensor), args))
        kwargs = dict(filter(lambda kwarg: isinstance(kwarg[1], torch.Tensor) or kwarg[0] == "out", kwargs.items()))
        output = self.runtime(*args, **kwargs)
        return output

    def __repr__(self) -> str:
        return repr(self.runtime)

    def parameters(self):
        yield torch.tensor([1.0], dtype=torch.float32, device=torch.device("cpu"))


class UnavailableRuntime:
    """
    A placeholder class used when model runtimes are not created.

    This class is returned by RBLNBaseModel._from_compiled_models when rbln_config.create_runtimes=False.
    It provides proper error messages when users attempt to use a model that was loaded without
    runtime creation.

    Usage:
        1. When compiling models on machines without NPU hardware
        2. When preparing models for later deployment
        3. When only model compilation is needed, not inference

    To use a model with runtimes, either:
        - Load the model with from_pretrained(..., rbln_create_runtimes=True)
        - Or set rbln_config={"create_runtimes": True} during loading
    """

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Raises a RuntimeError when the model is called without runtimes."""
        raise self.forward(*args, **kwargs)

    def __len__(self) -> int:
        """Returns 0 since no runtimes are available."""
        return 0

    def __getitem__(self, idx: int) -> Any:
        """Returns self for any index, allowing iteration to work with appropriate errors."""
        return self

    def __iter__(self):
        """Returns an iterator with self as the only item."""
        return iter([self])

    def forward(self, *args: List["torch.Tensor"], **kwargs: "torch.Tensor"):
        """Raises a detailed RuntimeError explaining why inference cannot be performed."""
        raise RuntimeError(
            "Cannot perform inference: RBLN runtime is not available.\n\n"
            "This model was loaded with create_runtimes=False. To use this model for inference:\n"
            "1. Load the model with runtime creation enabled:\n"
            "   model = RBLNModel.from_pretrained(..., rbln_create_runtimes=True)\n"
            "2. Ensure your NPU hardware is properly configured (check with 'rbln-stat' command)\n"
            "3. If you're on a machine without NPU hardware, you need to transfer the model files\n"
            "   to a compatible system with NPU support."
        )

    def __repr__(self) -> str:
        """Returns a detailed string representation of the UnavailableRuntime."""
        return "<UnavailableRuntime: Model loaded without runtime creation (create_runtimes=False)>"


class ContextRblnConfig:
    _local = threading.local()

    def __init__(
        self,
        device=None,
        device_map=None,
        create_runtimes=None,
        optimize_host_mem=None,
        activate_profiler=None,
        timeout=None,
    ):
        self.device = device
        self.device_map = device_map
        self.create_runtimes = create_runtimes
        self.optimize_host_mem = optimize_host_mem
        self.activate_profiler = activate_profiler
        self.timeout = timeout

    def __enter__(self):
        self._local.device = self.device
        self._local.device_map = self.device_map
        self._local.create_runtimes = self.create_runtimes
        self._local.optimize_host_memory = self.optimize_host_mem
        self._local.activate_profiler = self.activate_profiler
        self._local.timeout = self.timeout
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._local.device = None
        self._local.device_map = None
        self._local.create_runtimes = None
        self._local.optimize_host_memory = None
        self._local.activate_profiler = None
        self._local.timeout = None

    @classmethod
    def get_current_context(cls):
        return {
            "device": getattr(cls._local, "device", None),
            "device_map": getattr(cls._local, "device_map", None),
            "create_runtimes": getattr(cls._local, "create_runtimes", None),
            "optimize_host_memory": getattr(cls._local, "optimize_host_memory", None),
            "activate_profiler": getattr(cls._local, "activate_profiler", None),
            "timeout": getattr(cls._local, "timeout", None),
        }
