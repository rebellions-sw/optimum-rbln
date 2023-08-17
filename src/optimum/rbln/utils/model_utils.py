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

# Prefix used for RBLN model class names
RBLN_PREFIX = "RBLN"


def convert_hf_to_rbln_model_name(hf_model_name: str):
    """
    Convert Hugging Face model name to RBLN model name.

    Args:
        hf_model_name (str): The Hugging Face model name.

    Returns:
        str: The corresponding RBLN model name.
    """
    return RBLN_PREFIX + hf_model_name


def convert_rbln_to_hf_model_name(rbln_model_name: str):
    """
    Convert RBLN model name to Hugging Face model name.

    Args:
        rbln_model_name (str): The RBLN model name.

    Returns:
        str: The corresponding Hugging Face model name.
    """

    return rbln_model_name.removeprefix(RBLN_PREFIX)
