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
from typing import TYPE_CHECKING, Type


if TYPE_CHECKING:
    from ..modeling import RBLNModel

# Prefix used for RBLN model class names
RBLN_PREFIX = "RBLN"


MODEL_MAPPING = {}


def convert_hf_to_rbln_model_name(hf_model_name: str):
    """
    Convert HuggingFace model name to RBLN model name.

    Args:
        hf_model_name (str): The HuggingFace model name.

    Returns:
        str: The corresponding RBLN model name.
    """
    return RBLN_PREFIX + hf_model_name


def convert_rbln_to_hf_model_name(rbln_model_name: str):
    """
    Convert RBLN model name to HuggingFace model name.

    Args:
        rbln_model_name (str): The RBLN model name.

    Returns:
        str: The corresponding HuggingFace model name.
    """

    return rbln_model_name.removeprefix(RBLN_PREFIX)


def get_rbln_model_cls(cls_name: str) -> Type["RBLNModel"]:
    cls = getattr(importlib.import_module("optimum.rbln"), cls_name, None)
    if cls is None:
        if cls_name in MODEL_MAPPING:
            cls = MODEL_MAPPING[cls_name]
        else:
            raise AttributeError(f"RBLNModel for {cls_name} not found.")
    return cls
