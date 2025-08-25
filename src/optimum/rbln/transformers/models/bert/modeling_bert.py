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

import torch

from ...modeling_generic import (
    RBLNModelForMaskedLM,
    RBLNModelForQuestionAnswering,
    RBLNTransformerEncoderForFeatureExtraction,
)
from .bert_architecture import BertModelWrapper
from .configuration_bert import RBLNBertModelConfig


class RBLNBertModel(RBLNTransformerEncoderForFeatureExtraction):
    """
    RBLN optimized BERT model for feature extraction tasks.

    This class provides hardware-accelerated inference for BERT models
    on RBLN devices, optimized for extracting contextualized embeddings
    and features from text sequences.
    """

    rbln_model_input_names = ["input_ids", "attention_mask"]

    @classmethod
    def wrap_model_if_needed(cls, model: torch.nn.Module, rbln_config: RBLNBertModelConfig) -> torch.nn.Module:
        return BertModelWrapper(model, rbln_config)


class RBLNBertForMaskedLM(RBLNModelForMaskedLM):
    """
    RBLN optimized BERT model for masked language modeling tasks.

    This class provides hardware-accelerated inference for BERT models
    on RBLN devices, supporting masked language modeling tasks such as
    token prediction and text completion.
    """

    rbln_model_input_names = ["input_ids", "attention_mask", "token_type_ids"]


class RBLNBertForQuestionAnswering(RBLNModelForQuestionAnswering):
    """
    RBLN optimized BERT model for question answering tasks.

    This class provides hardware-accelerated inference for BERT models
    on RBLN devices, supporting extractive question answering tasks where
    the model predicts start and end positions of answers in text.
    """

    rbln_model_input_names = ["input_ids", "attention_mask", "token_type_ids"]
