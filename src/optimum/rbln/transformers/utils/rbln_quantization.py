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

import glob
import os
from typing import Any, Dict, Optional, Union

import torch
from safetensors.torch import load_file
from torch.nn import Linear, Parameter
from torch.nn import functional as F

from ...configuration_utils import RBLNSerializableConfigProtocol
from ...utils.logging import get_logger


logger = get_logger()


class RBLNQuantizationConfig(RBLNSerializableConfigProtocol):
    SUPPORTED_FORMATS = ["rbln"]
    SUPPORTED_WEIGHTS = ["int4", "fp8", "fp16"]
    SUPPORTED_ACTIVATIONS = ["fp8", "fp16"]
    SUPPORTED_KVCACHES = ["fp8", "fp16"]
    RBLN_QUANT_BITS_ENV = "RBLN_QUANT_BITS"

    def __init__(
        self,
        format: Optional[str] = None,
        weights: Optional[str] = None,
        activations: Optional[str] = None,
        kv_caches: Optional[str] = None,
        *,
        precision: Optional[str] = None,
    ):
        self.format = format or "rbln"
        if self.format not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Invalid format: {self.format}, supported formats are: {self.SUPPORTED_FORMATS}")

        if precision is not None:
            logger.warning("The `precision` argument is deprecated. Use `weights` and `activations` instead.")
            if any(precision_arg is not None for precision_arg in (weights, activations)):
                raise ValueError("`precision` and `weights` or `activations` cannot be set at the same time.")

            if precision == "w4a16":
                weights = "int4"
                activations = "fp16"
            else:
                raise ValueError(f"Invalid precision: {precision}")

        self.weights = weights or "fp16"
        self.activations = activations or "fp16"
        self.kv_caches = kv_caches or "fp16"

        self._validate()

    def _validate(self):
        if self.format not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Invalid format: {self.format}, supported formats are: {self.SUPPORTED_FORMATS}")
        if self.weights not in self.SUPPORTED_WEIGHTS:
            raise ValueError(f"Invalid weights: {self.weights}, supported weights are: {self.SUPPORTED_WEIGHTS}")
        if self.activations not in self.SUPPORTED_ACTIVATIONS:
            raise ValueError(
                f"Invalid activations: {self.activations}, supported activations are: {self.SUPPORTED_ACTIVATIONS}"
            )
        if self.kv_caches not in self.SUPPORTED_KVCACHES:
            raise ValueError(
                f"Invalid kv_caches: {self.kv_caches}, supported kv_caches are: {self.SUPPORTED_KVCACHES}"
            )
        if self.weights == "fp16" and self.activations == "fp16":
            raise ValueError("weights and activations of QuantizationConfig cannot be both fp16. It is meaningless.")

    def _prepare_for_serialization(self) -> Dict[str, Any]:
        return {
            "format": self.format,
            "weights": self.weights,
            "activations": self.activations,
            "kv_caches": self.kv_caches,
        }

    def maybe_set_quantization_env(self):
        quant_bits = None
        if self.weights == "int4":
            quant_bits = "4"
            os.environ[self.RBLN_QUANT_BITS_ENV] = quant_bits

    def maybe_reset_quantization_env(self):
        if self.RBLN_QUANT_BITS_ENV in os.environ:
            os.environ.pop(self.RBLN_QUANT_BITS_ENV)


class QuantizedLayerFactory:
    def __init__(self, quantization_config: RBLNQuantizationConfig):
        self.quantization_config = quantization_config

    def create_linear(self, layer: Linear) -> Linear:
        if self.quantization_config.weights == "int4":
            return self.create_qlinear(layer)
        elif self.quantization_config.weights == "fp8":
            return self.create_fp8linear(layer)
        else:
            raise ValueError(f"Invalid quantization weights: {self.quantization_config.weights}")

    def create_qlinear(self, layer: Linear) -> Linear:
        return create_qlinear(layer, self.quantization_config)

    def create_fp8linear(self, layer: Linear) -> Linear:
        return create_fp8linear(layer, self.quantization_config)


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
    use_auth_token: Optional[Union[bool, str]] = None,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    force_download: bool = False,
    local_files_only: bool = False,
    rbln_quantization: Optional[RBLNQuantizationConfig] = None,
) -> torch.nn.Module:
    """
    Prepare the model for quantization by updating specified linear layers to quantized (qlinear) layers.
    """

    update_layers_to_quantize(model, rbln_quantization)
    load_weights(
        model,
        model_id,
        n_layer,
        use_auth_token=use_auth_token,
        revision=revision,
        cache_dir=cache_dir,
        force_download=force_download,
        local_files_only=local_files_only,
        rbln_quantization=rbln_quantization,
    )
    return model


def update_layers_to_quantize(
    module: torch.nn.Module, rbln_quantization: Optional[RBLNQuantizationConfig] = None
) -> None:
    """
    Updates specified linear layers to quantized (qlinear) layers in the given module.
    """

    processed_layers = []
    quantized_layer_factory = QuantizedLayerFactory(rbln_quantization)

    for name, layer in module.named_modules():
        if is_target_for_qlinear_replacement(name, layer):
            parent_module, layer_name = get_parent_and_child(module, name)
            setattr(parent_module, layer_name, quantized_layer_factory.create_linear(layer))
            processed_layers.append(name)
        elif rbln_quantization.kv_caches == "fp8" and is_target_for_adding_kv_scales(name):
            layer.k_scale = Parameter(torch.tensor(1, dtype=torch.float32), requires_grad=False)
            layer.v_scale = Parameter(torch.tensor(1, dtype=torch.float32), requires_grad=False)

    if processed_layers:
        logger.debug(f"Updated the following linear layers to quantized layers:\n {{{', '.join(processed_layers)}}}")


def load_weights(
    model,
    model_id,
    n_layer=None,
    use_auth_token=None,
    revision=None,
    cache_dir=None,
    force_download=False,
    local_files_only=False,
    rbln_quantization: Optional[RBLNQuantizationConfig] = None,
):
    """
    Load safetensor file data directly into the model, filtering by layer if n_layer is provided.
    """

    model_params = dict(model.named_parameters(recurse=True))
    model_buffers = dict(model.named_buffers(recurse=True))

    if os.path.isdir(model_id):
        safetensor_files = glob.glob(f"{model_id}/*.safetensors")
    else:
        from huggingface_hub import hf_hub_download, list_repo_files

        try:
            # List all files in the repository
            repo_files = list_repo_files(model_id, revision=revision, token=use_auth_token)
            # Filter for safetensors files
            safetensor_files = []

            for file in repo_files:
                if file.endswith(".safetensors"):
                    # Download the safetensors file
                    downloaded_file = hf_hub_download(
                        repo_id=model_id,
                        filename=file,
                        revision=revision,
                        token=use_auth_token,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        local_files_only=local_files_only,
                    )
                    safetensor_files.append(downloaded_file)
        except Exception as e:
            logger.error(f"Failed to download safetensors files from Hugging Face Hub: {e}")
            raise e

    if not safetensor_files:
        raise FileNotFoundError(f"No safetensors files found for model_id: {model_id}")

    target_layers = list(range(n_layer)) if n_layer is not None else None

    unloaded_keys = []
    loaded_input_scale = False
    loaded_kv_scale = False
    loaded_weight_scale = False

    for safetensor_file in safetensor_files:
        file_data = load_file(safetensor_file)

        for key, value in file_data.items():
            loaded_input_scale = loaded_input_scale or "input_scale" in key
            loaded_weight_scale = loaded_weight_scale or "weight_scale" in key
            loaded_kv_scale = loaded_kv_scale or "kv_scale" in key

            if target_layers is not None:
                parts = key.split(".")

                if len(parts) > 2 and parts[2].isdigit() and (int(parts[2]) not in target_layers):
                    continue

            if key in model_params:
                model_params[key].data.copy_(value)
            elif key in model_buffers:
                model_buffers[key].data.copy_(value)
            elif "kv_scale" in key:
                if rbln_quantization.kv_caches == "fp8":
                    model_params[key.replace("kv_scale", "k_scale")].data.copy_(value)
                    model_params[key.replace("kv_scale", "v_scale")].data.copy_(value)
                else:
                    unloaded_keys.append(key)
            else:
                unloaded_keys.append(key)

    if len(unloaded_keys) > 0:
        logger.warning(f"There are unexpected parameters/buffers on the checkpoint: {unloaded_keys}")

    if not loaded_input_scale and rbln_quantization.activations == "fp8":
        raise ValueError(
            "No input_scale found in the checkpoint. Did you use the correct quantization config? "
            "If you are using fp8 quantization, you need to use the correct quantization config."
        )
    if not loaded_weight_scale and rbln_quantization.weights == "fp8":
        raise ValueError(
            "No weight_scale found in the checkpoint. Did you use the correct quantization config? "
            "If you are using fp8 quantization, you need to use the correct quantization config."
        )
    if not loaded_kv_scale and rbln_quantization.kv_caches == "fp8":
        raise ValueError(
            "No kv_scale found in the checkpoint. Did you use the correct quantization config? "
            "If you are using fp8 quantization, you need to use the correct quantization config."
        )
    if loaded_kv_scale and rbln_quantization.kv_caches != "fp8":
        logger.warning(
            "kv_scale found in the checkpoint, but kv_caches of quantization config is not fp8. Ignoring kv_scale."
        )


def is_target_for_qlinear_replacement(layer_name: str, layer: torch.nn.Module) -> bool:
    """
    Checks if a layer is a target for qlinear replacement.
    """
    return layer_name.split(".")[-1] in QUANTIZED_WEIGHTS and isinstance(layer, torch.nn.Linear)


def is_target_for_adding_kv_scales(layer_name: str) -> bool:
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


def create_qlinear(layer: Linear, rbln_quantization: RBLNQuantizationConfig) -> Linear:
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


def create_fp8linear(layer: Linear, rbln_quantization: RBLNQuantizationConfig) -> Linear:
    """
    Converts a standard linear layer to a fp8 linear layer with a custom forward pass.
    """

    def static_per_tensor_quantize(tensor: torch.Tensor, inv_scale: float) -> torch.Tensor:
        finfo = torch.finfo(torch.float8_e4m3fn)
        qweight = (tensor / inv_scale).clamp(min=finfo.min, max=finfo.max)
        return qweight

    def fp8_gemm(A: torch.Tensor, A_scale, B: torch.Tensor, B_scale, bias, out_dtype: torch.dtype):
        A = A.type(out_dtype)
        B = B.type(out_dtype)

        if A_scale is not None:
            A *= A_scale
        if B_scale is not None:
            B *= B_scale.to(out_dtype)

        output = torch.nn.functional.linear(A, B, bias=bias)
        return output

    def fp8linear_forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_scale:
            input = static_per_tensor_quantize(x, self.input_scale)
        else:
            input = x
        output = fp8_gemm(
            A=input,
            A_scale=self.input_scale,
            B=self.weight,
            B_scale=self.weight_scale,
            bias=self.bias,
            out_dtype=x.dtype,
        )

        return output

    layer.weight = Parameter(layer.weight.to(torch.float8_e4m3fn), requires_grad=False)
    layer.weight_scale = Parameter(torch.tensor(1, dtype=torch.float32), requires_grad=False)

    if rbln_quantization.activations == "fp8":
        layer.input_scale = Parameter(torch.tensor(1, dtype=torch.float32), requires_grad=False)
    else:
        layer.input_scale = None

    layer.forward = lambda inputs: fp8linear_forward(layer, inputs)

    return layer
