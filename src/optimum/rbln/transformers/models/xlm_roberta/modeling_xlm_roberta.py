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

import logging
from typing import TYPE_CHECKING, Optional, Union

import torch
from transformers import PretrainedConfig

from ....modeling import RBLNModel
from ....modeling_config import RBLNCompileConfig, RBLNConfig


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer


class RBLNXLMRobertaModel(RBLNModel):
    @classmethod
    def _get_rbln_config(
        cls,
        preprocessors: Optional[Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"]],
        model_config: Optional["PretrainedConfig"] = None,
        rbln_kwargs={},
    ) -> RBLNConfig:
        rbln_max_seq_len = rbln_kwargs.get("max_seq_len", None)
        rbln_model_input_names = rbln_kwargs.get("model_input_names", None)
        rbln_batch_size = rbln_kwargs.get("batch_size", None)

        max_position_embeddings = getattr(model_config, "n_positions", None) or getattr(
            model_config, "max_position_embeddings", None
        )

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
            raise ValueError("`rbln_enc_max_seq_len` should be less or equal than max_position_embeddings!")

        if rbln_model_input_names is None:
            # These are BERT's inputs
            rbln_model_input_names = ["input_ids", "attention_mask", "token_type_ids"]

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

    def forward(
        self,
        input_ids: "torch.Tensor",
        attention_mask: "torch.Tensor",
        token_type_ids: "torch.Tensor" = None,
        **kwargs,
    ):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input=input_ids, dtype=torch.int64)
        output = super().forward(input_ids, attention_mask, token_type_ids)
        return output
