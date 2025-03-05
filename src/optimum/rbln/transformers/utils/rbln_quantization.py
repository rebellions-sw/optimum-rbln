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
    "rbln": ["w4a16", "fp8_exp"],
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
        raise ValueError(f"Invalid {key}: {value}{context_info}. Supported values are: {valid_values_str}")

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
            if q_precision != "fp8_exp":
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


def prepare_model_for_quantization(
    model: torch.nn.Module,
    model_id: str,
    n_layer: Optional[int] = None,
    rbln_quantization: Optional[Dict[str, str]] = {},
) -> None:
    """
    Prepare the model for quantization by updating specified linear layers to quantized (qlinear) layers.
    """
    if rbln_quantization["precision"] == "w4a16":
        replace_method = create_qlinear
    elif rbln_quantization["precision"] == "fp8_exp":
        replace_method = create_fp8linear
    else:
        replace_method = None

    # TODO(jongho): organized quantization config
    kvcache_scale = False
    if "kvcache" in rbln_quantization and rbln_quantization["kvcache"] == "fp8":
        kvcache_scale = True

    update_layers_to_quantize(model, replace_method=replace_method, kvcache_scale=kvcache_scale)
    load_weights(model, model_id, n_layer)


def update_layers_to_quantize(module: torch.nn.Module, replace_method: Callable, kvcache_scale: bool = False) -> None:
    """
    Updates specified linear layers to quantized (qlinear) layers in the given module.
    """

    logger.debug("Updating layers to be quantized")  # TODO(jongho): remove.
    if replace_method not in [create_fp8linear, create_qlinear]:
        raise NotImplementedError

    processed_layers = []

    for name, layer in module.named_modules():
        if is_target_for_qlinear_replacement(name, layer):
            parent_module, layer_name = get_parent_and_child(module, name)
            setattr(parent_module, layer_name, replace_method(layer))
            processed_layers.append(name)
        elif kvcache_scale and is_target_for_adding_kv_scales(name):
            layer.k_scale = Parameter(torch.tensor(1, dtype=torch.float32), requires_grad=False)
            layer.v_scale = Parameter(torch.tensor(1, dtype=torch.float32), requires_grad=False)

    if processed_layers:
        logger.debug(f"Updated the following linear layers to quantized layers:\n {{{', '.join(processed_layers)}}}")


def load_weights(model, model_id, n_layer=None):
    """
    Load safetensor file data directly into the model, filtering by layer if n_layer is provided.
    """
    logger.debug("Loading the quantized weights into the CPU.")  # TODO(jongho): remove.

    model_params = dict(model.named_parameters(recurse=True))
    model_buffers = dict(model.named_buffers(recurse=True))
    safetensor_files = glob.glob(f"{model_id}/*.safetensors")

    target_layers = list(range(n_layer)) if n_layer is not None else None

    unloaded_keys = []
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
            elif "kv_scale" in key:
                model_params[key.replace("kv_scale", "k_scale")].data.copy_(value)
                model_params[key.replace("kv_scale", "v_scale")].data.copy_(value)
            else:
                unloaded_keys.append(key)

    if len(unloaded_keys) > 0:
        logger.warning(f"There are unexpected parameters/buffers on the checkpoint: {unloaded_keys}")

    logger.debug("Loaded the quantized weights into the CPU.")


def is_target_for_qlinear_replacement(layer_name: str, layer: torch.nn.Module) -> bool:
    """
    Checks if a layer is a target for qlinear replacement.
    """
    return layer_name.split(".")[-1] in QUANTIZED_WEIGHTS and isinstance(layer, torch.nn.Linear)


def is_target_for_adding_kv_scales(layer_name: str) -> bool:
    # FIXME
    return layer_name.split(".")[-1] in ["self_attn"]


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


def create_fp8linear(layer: Linear) -> Linear:
    """
    Converts a standard linear layer to a fp8 linear layer with a custom forward pass.
    """

    def static_per_tensor_quantize(tensor: torch.Tensor, inv_scale: float) -> torch.Tensor:
        finfo = torch.finfo(torch.float8_e4m3fn)
        qweight = (tensor / inv_scale).clamp(min=finfo.min, max=finfo.max)
        return qweight  # .to(torch.float8_e4m3fn)

    def fp8_gemm(A: torch.Tensor, A_scale, B: torch.Tensor, B_scale, bias, out_dtype: torch.dtype):
        A = A.type(out_dtype)
        B = B.type(out_dtype)

        A *= A_scale
        B *= B_scale.to(out_dtype)

        output = torch.nn.functional.linear(A, B, bias=bias)
        return output

    def fp8linear_forward(self, x: torch.Tensor) -> torch.Tensor:
        qinput = static_per_tensor_quantize(x, self.input_scale)
        output = fp8_gemm(
            A=qinput,
            A_scale=self.input_scale,
            B=self.weight,
            B_scale=self.weight_scale,
            bias=self.bias,
            out_dtype=x.dtype,
        )

        return output

    layer.weight = Parameter(layer.weight.to(torch.float8_e4m3fn), requires_grad=False)
    layer.weight_scale = Parameter(torch.tensor(1, dtype=torch.float32), requires_grad=False)
    layer.input_scale = Parameter(torch.tensor(1, dtype=torch.float32), requires_grad=False)
    layer.forward = lambda inputs: fp8linear_forward(layer, inputs)

    return layer
