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

import functools
import glob
import os
from typing import Any, Callable, Dict, Optional

import torch
from safetensors.torch import load_file
from torch.nn import Linear, Parameter
from torch.nn import functional as F

from ...utils.logging import get_logger


logger = get_logger()

SUPPORTED_QUANTIZATIONS: Dict[str, list[str]] = {
    "rbln": ["w4a16"],
}


class QuantizationManager:
    # The RBLN_QUANT_BITS environment variable defines the precision of each layer during the graph compilation process.
    # It specifies the quantization bit depth. For instance, setting RBLN_QUANT_BITS=4 will apply 4-bit precision for quantization.
    RBLN_QUANT_BITS_ENV = "RBLN_QUANT_BITS"

    @staticmethod
    def _raise_invalid_config_error(
        key: str, value: str, valid_values: list[str], context: Optional[str] = None
    ) -> None:
        context_info = f" for {context}" if context else ""
        valid_values_str = ", ".join(valid_values)
        raise ValueError(f"Invalid {key}: {value}{context_info}. " f"Supported values are: {valid_values_str}")

    @staticmethod
    def validate_quantization_config(quantize_config: Optional[dict]) -> Optional[dict]:
        if not quantize_config:
            return None

        q_format = quantize_config.get("format")
        q_precision = quantize_config.get("precision")

        if q_format not in SUPPORTED_QUANTIZATIONS:
            QuantizationManager._raise_invalid_config_error(
                "quantization format", q_format, list(SUPPORTED_QUANTIZATIONS.keys())
            )

        if q_precision not in SUPPORTED_QUANTIZATIONS[q_format]:
            QuantizationManager._raise_invalid_config_error(
                "precision", q_precision, SUPPORTED_QUANTIZATIONS[q_format], q_format
            )

        return quantize_config

    @classmethod
    def _set_env_var(cls, name: str, value: str) -> None:
        os.environ[name] = value

    @classmethod
    def _unset_env_var(cls, name: str) -> None:
        os.environ.pop(name, None)

    @classmethod
    def set_quantization_env(cls, quantize_config: Optional[dict]) -> Optional[str]:
        quantize_config = cls.validate_quantization_config(quantize_config)
        if quantize_config:
            q_precision: str = quantize_config["precision"]
            quant_bits = q_precision.split("w")[1].split("a")[0]
            cls._set_env_var(cls.RBLN_QUANT_BITS_ENV, quant_bits)
            return cls.RBLN_QUANT_BITS_ENV
        return None

    @classmethod
    def reset_quantization_env(cls, env_var_name: Optional[str]) -> None:
        if env_var_name:
            cls._unset_env_var(env_var_name)

    @classmethod
    def with_quantization_env(cls, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            quantize_config = kwargs.get("quantize_config")
            quantize_env_var = cls.set_quantization_env(quantize_config)
            try:
                return func(*args, **kwargs)
            finally:
                cls.reset_quantization_env(quantize_env_var)

        return wrapper


# Constants
QUANTIZED_WEIGHTS = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}


def prepare_model_for_quantization(model: torch.nn.Module, model_id: str, n_layer: Optional[int] = None) -> None:
    """
    Prepare the model for quantization by updating specified linear layers to quantized (qlinear) layers.
    """
    update_layers_to_quantize(model)
    load_weights(model, model_id, n_layer)


def update_layers_to_quantize(module: torch.nn.Module) -> None:
    """
    Updates specified linear layers to quantized (qlinear) layers in the given module.
    """
    processed_layers = []

    for name, layer in module.named_modules():
        if is_target_for_qlinear_replacement(name, layer):
            parent_module, layer_name = get_parent_and_child(module, name)
            setattr(parent_module, layer_name, create_qlinear(layer))
            processed_layers.append(name)

    if processed_layers:
        logger.debug(f"Updated the following linear layers to quantized layers:\n {{{', '.join(processed_layers)}}}")


def load_weights(model, model_id, n_layer=None):
    """
    Load safetensor file data directly into the model, filtering by layer if n_layer is provided.
    """

    model_params = dict(model.named_parameters(recurse=True))
    model_buffers = dict(model.named_buffers(recurse=True))
    safetensor_files = glob.glob(f"{model_id}/*.safetensors")

    target_layers = list(range(n_layer)) if n_layer is not None else None

    for safetensor_file in safetensor_files:
        file_data = load_file(safetensor_file)
        for key, value in file_data.items():
            if target_layers is not None:
                parts = key.split(".")

                if len(parts) > 2 and parts[2].isdigit() and (int(parts[2]) not in target_layers):
                    continue

            if key in model_params:
                model_params[key].data.copy_(value)
            elif key in model_buffers:
                model_buffers[key].data.copy_(value)


def is_target_for_qlinear_replacement(layer_name: str, layer: torch.nn.Module) -> bool:
    """
    Checks if a layer is a target for qlinear replacement.
    """
    return layer_name.split(".")[-1] in QUANTIZED_WEIGHTS and isinstance(layer, torch.nn.Linear)


def get_parent_and_child(module: torch.nn.Module, full_name: str) -> tuple:
    """
    Splits the full layer name to retrieve the parent module and the child layer.
    """
    *parent_address, child_name = full_name.split(".")
    parent_module = access_attribute(module, parent_address)
    return parent_module, child_name


def access_attribute(obj: Any, attributes: list[str]) -> Any:
    """
    Recursively accesses a nested attribute from an object using a list of attribute names.
    """
    for attr in attributes:
        obj = getattr(obj, attr)
    return obj


def create_qlinear(layer: Linear) -> Linear:
    """
    Converts a standard linear layer to a quantized linear (qlinear) layer with a custom forward pass.
    """

    def qlinear_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.dtype != self.scales.dtype:
            raise TypeError(f"Expected input dtype {self.scales.dtype}, but got {inputs.dtype}")

        w_fp = self.weight.type(inputs.dtype)
        w_fp *= self.scales.view(-1, 1)
        return F.linear(inputs, w_fp, self.bias)

    # Convert weight to int8 and add scale parameter
    layer.weight = Parameter(layer.weight.to(torch.int8), requires_grad=False)
    layer.scales = Parameter(torch.ones(layer.out_features, dtype=torch.float32), requires_grad=False)
    layer.forward = lambda inputs: qlinear_forward(layer, inputs)

    return layer
