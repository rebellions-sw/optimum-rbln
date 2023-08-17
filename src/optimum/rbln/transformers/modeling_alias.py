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

from ..utils.logging import get_logger
from .modeling_generic import (
    RBLNModelForAudioClassification,
    RBLNModelForImageClassification,
    RBLNModelForMaskedLM,
    RBLNModelForQuestionAnswering,
    RBLNModelForSequenceClassification,
)


logger = get_logger()


class RBLNASTForAudioClassification(RBLNModelForAudioClassification):
    pass


class RBLNBertForQuestionAnswering(RBLNModelForQuestionAnswering):
    rbln_model_input_names = ["input_ids", "attention_mask", "token_type_ids"]


class RBLNDistilBertForQuestionAnswering(RBLNModelForQuestionAnswering):
    rbln_model_input_names = ["input_ids", "attention_mask"]


class RBLNResNetForImageClassification(RBLNModelForImageClassification):
    pass


class RBLNXLMRobertaForSequenceClassification(RBLNModelForSequenceClassification):
    rbln_model_input_names = ["input_ids", "attention_mask"]


class RBLNRobertaForSequenceClassification(RBLNModelForSequenceClassification):
    rbln_model_input_names = ["input_ids", "attention_mask"]


class RBLNRobertaForMaskedLM(RBLNModelForMaskedLM):
    rbln_model_input_names = ["input_ids", "attention_mask"]


class RBLNViTForImageClassification(RBLNModelForImageClassification):
    pass
