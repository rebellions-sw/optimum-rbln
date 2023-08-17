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
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

from ..modeling_config import RBLNConfig


if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from ..modeling_base import RBLNBaseModel


class SubModulesMixin:
    """
    _rbln_submodules = [
        {"name": "vision_tower"},
        {"name": "language_model"},
    ]
    """

    _rbln_submodules: List[Dict[str, Any]] = []

    def __init__(
        self,
        *,
        rbln_submodules: List["RBLNBaseModel"] = [],
        **kwargs,
    ) -> None:
        for submodule_meta, submodule in zip(self._rbln_submodules, rbln_submodules):
            setattr(self, submodule_meta["name"], submodule)

    @classmethod
    def _export_submodules_from_model(
        cls,
        model: "PreTrainedModel",
        model_save_dir: str,
        rbln_kwargs: Dict[str, Any],
        **kwargs,
    ) -> List["RBLNBaseModel"]:
        rbln_submodules = []
        for submodule in cls._rbln_submodules:
            submodule_name = submodule["name"]
            torch_submodule: "PreTrainedModel" = getattr(model, submodule["name"])
            cls_name = torch_submodule.__class__.__name__
            submodule_cls: "RBLNBaseModel" = getattr(importlib.import_module("optimum.rbln"), f"RBLN{cls_name}")

            if submodule_name in rbln_kwargs:
                kwargs["rbln_config"] = rbln_kwargs[submodule_name]

            rbln_submodule = submodule_cls.from_model(
                model=torch_submodule,
                subfolder=submodule_name,
                model_save_dir=model_save_dir,
                **kwargs,
            )

            rbln_submodules.append(rbln_submodule)

        return rbln_submodules

    @classmethod
    def _load_submodules_from_compiled_models(
        cls,
        model_save_dir: str,
        rbln_kwargs: Dict[str, Any],
        **kwargs,
    ):
        rbln_submodules = []
        for submodule in cls._rbln_submodules:
            submodule_name = submodule["name"]

            if submodule_name in rbln_kwargs:
                kwargs["rbln_config"] = rbln_kwargs[submodule_name]

            # Get cls name for call the constructor of the rbln class
            submodule_rbln_config = RBLNConfig.load(Path(model_save_dir) / submodule_name)
            submodule_cls_name = submodule_rbln_config.meta["cls"]
            submodule_cls: "RBLNBaseModel" = getattr(importlib.import_module("optimum.rbln"), submodule_cls_name)

            rbln_submodule = submodule_cls._from_pretrained(
                model_id=model_save_dir,
                config=None,
                subfolder=submodule_name,
                **kwargs,
            )
            rbln_submodules.append(rbln_submodule)
        return rbln_submodules

    @classmethod
    def _load_submodules(
        cls,
        model_save_dir,
        rbln_kwargs,
        model=None,
        **kwargs,
    ):
        # Two ways :
        # 1. Compile from pytorch object
        # 2. Load from compiled file
        if model is not None:
            return cls._export_submodules_from_model(
                model=model,
                model_save_dir=model_save_dir,
                rbln_kwargs=rbln_kwargs,
                **kwargs,
            )

        else:
            return cls._load_submodules_from_compiled_models(
                model_save_dir=model_save_dir,
                rbln_kwargs=rbln_kwargs,
                **kwargs,
            )
