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

import inspect
import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from transformers import PretrainedConfig

from ....modeling import RBLNModel
from ....modeling_config import RBLNCompileConfig, RBLNConfig


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer


class RBLNBertModel(RBLNModel):
    @classmethod
    def _get_rbln_config(
        cls,
        preprocessors: Optional[Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"]],
        model_config: Optional["PretrainedConfig"] = None,
        rbln_kwargs: Dict[str, Any] = {},
    ) -> RBLNConfig:
        rbln_max_seq_len = rbln_kwargs.get("max_seq_len", None)
        rbln_batch_size = rbln_kwargs.get("batch_size", None)
        rbln_model_input_names = rbln_kwargs.get("model_input_names", None)

        max_position_embeddings = getattr(model_config, "max_position_embeddings", None)

        if rbln_max_seq_len is None:
            rbln_max_seq_len = max_position_embeddings
            if rbln_max_seq_len is None:
                for tokenizer in preprocessors:
                    if hasattr(tokenizer, "model_max_length"):
                        rbln_max_seq_len = tokenizer.model_max_length
                        break
                if rbln_max_seq_len is None:
                    raise ValueError("`rbln_max_seq_len` should be specified!")

        if max_position_embeddings is not None and rbln_max_seq_len > max_position_embeddings:
            raise ValueError("`rbln_max_seq_len` should be less or equal than max_position_embeddings!")

        if rbln_model_input_names is None:
            for tokenizer in preprocessors:
                if hasattr(tokenizer, "model_input_names"):
                    rbln_model_input_names = tokenizer.model_input_names
                    break
            if rbln_model_input_names is None and hasattr(cls, "rbln_model_input_names"):
                rbln_model_input_names = cls.rbln_model_input_names
            elif rbln_model_input_names is None and hasattr(cls, "rbln_model_input_names") is False:
                input_names_order = inspect.signature(cls.hf_class.forward).parameters.keys()
                raise ValueError(
                    "Specify the model input names obtained by the tokenizer via `rbln_model_input_names`, "
                    f"and be sure to make the order of the inputs same as BertModel forward() arguments like ({list(input_names_order)})"
                )

        if rbln_batch_size is None:
            rbln_batch_size = 1

        input_info = [
            (model_input_name, [rbln_batch_size, rbln_max_seq_len], "int64")
            for model_input_name in rbln_model_input_names
        ]

        rbln_compile_config = RBLNCompileConfig(input_info=input_info)

        rbln_config = RBLNConfig(
            rbln_cls=cls.__name__,
            compile_cfgs=[rbln_compile_config],
            rbln_kwargs=rbln_kwargs,
        )

        rbln_config.model_cfg.update({"max_seq_len": rbln_max_seq_len})
        return rbln_config
