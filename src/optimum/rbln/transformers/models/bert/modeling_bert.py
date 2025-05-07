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

from ....utils.logging import get_logger
from ...modeling_generic import (
    RBLNModelForMaskedLM,
    RBLNModelForQuestionAnswering,
    RBLNTransformerEncoderForFeatureExtraction,
)


logger = get_logger(__name__)


class RBLNBertModel(RBLNTransformerEncoderForFeatureExtraction):
    rbln_model_input_names = ["input_ids", "attention_mask"]


class RBLNBertForMaskedLM(RBLNModelForMaskedLM):
    rbln_model_input_names = ["input_ids", "attention_mask", "token_type_ids"]


class RBLNBertForQuestionAnswering(RBLNModelForQuestionAnswering):
    rbln_model_input_names = ["input_ids", "attention_mask", "token_type_ids"]
