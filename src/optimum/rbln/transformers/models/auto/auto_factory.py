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
import inspect
import warnings

from transformers import AutoConfig, PretrainedConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.models.auto.auto_factory import _get_model_class

from optimum.rbln.modeling_base import RBLNBaseModel
from optimum.rbln.modeling_config import RBLNConfig
from optimum.rbln.utils.model_utils import convert_hf_to_rbln_model_name, convert_rbln_to_hf_model_name


class _BaseAutoModelClass:
    # Base class for auto models.
    _model_mapping = None

    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path)`"
        )

    @classmethod
    def get_rbln_cls(
        cls,
        pretrained_model_name_or_path,
        *args,
        export=True,
        **kwargs,
    ):
        """
        Determine the appropriate RBLN model class based on the given model ID and configuration.

        Args:
            pretrained_model_name_or_path (str): Identifier or path to the pretrained model.
            export (bool): Whether to infer the class based on Hugging Face (HF) architecture.
            kwargs: Additional arguments for configuration and loading.

        Returns:
            RBLNBaseModel: The corresponding RBLN model class.
        """
        if export:
            hf_model_class = cls.infer_hf_model_class(pretrained_model_name_or_path, **kwargs)
            rbln_class_name = convert_hf_to_rbln_model_name(hf_model_class.__name__)
        else:
            rbln_class_name = cls.get_rbln_model_class_name(pretrained_model_name_or_path, **kwargs)

            if convert_rbln_to_hf_model_name(rbln_class_name) not in cls._model_mapping_names.values():
                raise ValueError(
                    f"The architecture '{rbln_class_name}' is not supported by the `{cls.__name__}.from_pretrained()` method. "
                    "Please use the `from_pretrained()` method of the appropriate class to load this model, "
                    f"or directly use '{rbln_class_name}.from_pretrained()`."
                )

        try:
            module = importlib.import_module("optimum.rbln")
            rbln_cls = getattr(module, rbln_class_name)
        except AttributeError as e:
            raise AttributeError(
                f"Class '{rbln_class_name}' not found in 'optimum.rbln' module for model ID '{pretrained_model_name_or_path}'. "
                "Ensure that the class name is correctly mapped and available in the 'optimum.rbln' module."
            ) from e

        return rbln_cls

    @classmethod
    def infer_hf_model_class(
        cls,
        pretrained_model_name_or_path,
        *args,
        **kwargs,
    ):
        """
        Infer the Hugging Face model class based on the configuration or model name.

        Args:
            pretrained_model_name_or_path (str): Identifier or path to the pretrained model.
            kwargs: Additional arguments for configuration and loading.

        Returns:
            PretrainedModel: The inferred Hugging Face model class.
        """

        # Try to load configuration if provided or retrieve it from the model ID
        config = kwargs.pop("config", None)
        kwargs.update({"trust_remote_code": True})
        kwargs["_from_auto"] = True

        # Load configuration if not already provided
        if not isinstance(config, PretrainedConfig):
            config, kwargs = AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                return_unused_kwargs=True,
                **kwargs,
            )

        # Get hf_model_class from Config
        has_remote_code = (
            hasattr(config, "auto_map") and convert_rbln_to_hf_model_name(cls.__name__) in config.auto_map
        )
        if has_remote_code:
            class_ref = config.auto_map[convert_rbln_to_hf_model_name(cls.__name__)]
            model_class = get_class_from_dynamic_module(class_ref, pretrained_model_name_or_path, **kwargs)
        elif type(config) in cls._model_mapping.keys():
            model_class = _get_model_class(config, cls._model_mapping)
        else:
            raise ValueError(
                f"Unrecognized configuration class {config.__class__} for this kind of AutoModel: {cls.__name__}.\n"
                f"Model type should be one of {', '.join(c.__name__ for c in cls._model_mapping.keys())}."
            )

        if model_class.__name__ != config.architectures[0]:
            warnings.warn(
                f"`{cls.__name__}.from_pretrained()` is invoking `{convert_hf_to_rbln_model_name(model_class.__name__)}.from_pretrained()`, which does not match the "
                f"expected architecture `RBLN{config.architectures[0]}` from config. This mismatch could cause some operations to not be properly loaded "
                f"from the checkpoint, leading to potential unintended behavior. If this is not intentional, consider calling the "
                f"`from_pretrained()` method directly from the `RBLN{config.architectures[0]}` class instead.",
                UserWarning,
            )

        return model_class

    @classmethod
    def get_rbln_model_class_name(cls, pretrained_model_name_or_path, **kwargs):
        """
        Retrieve the path to the compiled model directory for a given RBLN model.

        Args:
            pretrained_model_name_or_path (str): Identifier of the model.
            kwargs: Additional arguments that match the parameters of `_load_compiled_model_dir`.

        Returns:
            str: Path to the compiled model directory.
        """
        sig = inspect.signature(RBLNBaseModel._load_compiled_model_dir)
        valid_params = sig.parameters.keys()
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

        model_path_subfolder = RBLNBaseModel._load_compiled_model_dir(
            model_id=pretrained_model_name_or_path, **filtered_kwargs
        )
        rbln_config = RBLNConfig.load(model_path_subfolder)

        return rbln_config.meta["cls"]

    @classmethod
    def from_pretrained(
        cls,
        model_id,
        *args,
        **kwargs,
    ):
        rbln_cls = cls.get_rbln_cls(model_id, *args, **kwargs)
        return rbln_cls.from_pretrained(model_id, *args, **kwargs)
