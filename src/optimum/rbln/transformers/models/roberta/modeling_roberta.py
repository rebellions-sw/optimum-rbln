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

from ...modeling_generic import RBLNModelForMaskedLM, RBLNModelForSequenceClassification


class RBLNRobertaForMaskedLM(RBLNModelForMaskedLM):
    """
    RBLN optimized RoBERTa model for masked language modeling tasks.

    This class provides hardware-accelerated inference for RoBERTa models
    on RBLN devices, supporting masked language modeling tasks such as
    token prediction and text completion.
    """

    rbln_model_input_names = ["input_ids", "attention_mask"]


class RBLNRobertaForSequenceClassification(RBLNModelForSequenceClassification):
    """
    RBLN optimized RoBERTa model for sequence classification tasks.

    This class provides hardware-accelerated inference for RoBERTa models
    on RBLN devices, supporting text classification tasks such as sentiment analysis,
    topic classification, and other sequence-level prediction tasks.
    """

    rbln_model_input_names = ["input_ids", "attention_mask"]
