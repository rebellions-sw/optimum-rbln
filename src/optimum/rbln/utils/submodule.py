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
from typing import TYPE_CHECKING, Any, Dict, List, Type

from ..configuration_utils import RBLNModelConfig


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

    def __init__(self, *, rbln_submodules: List["RBLNBaseModel"] = [], **kwargs) -> None:
        for submodule_meta, submodule in zip(self._rbln_submodules, rbln_submodules):
            setattr(self, submodule_meta["name"], submodule)

    @classmethod
    def _export_submodules_from_model(
        cls, model: "PreTrainedModel", model_save_dir: str, rbln_config: RBLNModelConfig, **kwargs
    ) -> List["RBLNBaseModel"]:
        rbln_submodules = []
        submodule_prefix = getattr(cls, "_rbln_submodule_prefix", None)

        for submodule in cls._rbln_submodules:
            submodule_name = submodule["name"]
            if submodule_prefix is not None:
                torch_submodule: PreTrainedModel = getattr(model, submodule_prefix)
                torch_submodule = getattr(torch_submodule, submodule_name)
            else:
                torch_submodule: PreTrainedModel = getattr(model, submodule_name)

            cls_name = torch_submodule.__class__.__name__
            submodule_cls: Type["RBLNBaseModel"] = getattr(importlib.import_module("optimum.rbln"), f"RBLN{cls_name}")
            submodule_rbln_config = getattr(rbln_config, submodule_name) or {}

            if isinstance(submodule_rbln_config, dict):
                submodule_rbln_config_class = submodule_cls.get_rbln_config_class()
                submodule_rbln_config = submodule_rbln_config_class(**submodule_rbln_config)
                setattr(rbln_config, submodule_name, submodule_rbln_config)

            rbln_submodule = submodule_cls.from_model(
                model=torch_submodule,
                config=torch_submodule.config,
                subfolder=submodule_name,
                model_save_dir=model_save_dir,
                rbln_config=submodule_rbln_config,
                **kwargs,
            )

            rbln_submodules.append(rbln_submodule)

        return rbln_submodules

    @classmethod
    def _load_submodules_from_compiled_models(cls, model_save_dir: str, rbln_config: RBLNModelConfig, **kwargs):
        rbln_submodules = []

        for submodule in cls._rbln_submodules:
            submodule_name = submodule["name"]

            # Get cls name for call the constructor of the rbln class
            submodule_rbln_config = getattr(rbln_config, submodule_name)

            # RBLNModelConfig -> RBLNModel
            submodule_cls: "RBLNBaseModel" = getattr(
                importlib.import_module("optimum.rbln"), submodule_rbln_config.rbln_model_cls_name
            )

            rbln_submodule = submodule_cls._from_pretrained(
                model_id=model_save_dir,
                config=None,
                subfolder=submodule_name,
                rbln_config=submodule_rbln_config,
                **kwargs,
            )

            # update submodule's rbln_config since it is updated in the from_pretrained method
            setattr(rbln_config, submodule_name, rbln_submodule.rbln_config)
            rbln_submodules.append(rbln_submodule)

        return rbln_submodules

    @classmethod
    def _load_submodules(cls, model_save_dir, rbln_config: RBLNModelConfig, model=None, **kwargs):
        # Two ways :
        # 1. Compile from pytorch object
        # 2. Load from compiled file
        if model is not None:
            return cls._export_submodules_from_model(
                model=model, model_save_dir=model_save_dir, rbln_config=rbln_config, **kwargs
            )

        else:
            return cls._load_submodules_from_compiled_models(
                model_save_dir=model_save_dir, rbln_config=rbln_config, **kwargs
            )
