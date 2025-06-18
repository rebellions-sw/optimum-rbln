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

from ...configuration_generic import (
    RBLNModelForMaskedLMConfig,
    RBLNModelForQuestionAnsweringConfig,
    RBLNTransformerEncoderForFeatureExtractionConfig,
)


class RBLNBertModelConfig(RBLNTransformerEncoderForFeatureExtractionConfig):
    """
    Configuration class for RBLNBertModel.

    This configuration class stores the configuration parameters specific to
    RBLN-optimized BERT models for feature extraction tasks.
    """


class RBLNBertForMaskedLMConfig(RBLNModelForMaskedLMConfig):
    """
    Configuration class for RBLNBertForMaskedLM.

    This configuration class stores the configuration parameters specific to
    RBLN-optimized BERT models for masked language modeling tasks.
    """


class RBLNBertForQuestionAnsweringConfig(RBLNModelForQuestionAnsweringConfig):
    """
    Configuration class for RBLNBertForQuestionAnswering.

    This configuration class stores the configuration parameters specific to
    RBLN-optimized BERT models for question answering tasks.
    """
