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

from ...configuration_generic import RBLNTransformerEncoderForFeatureExtractionConfig
from ..seq2seq import RBLNModelForSeq2SeqLMConfig


class RBLNT5EncoderModelConfig(RBLNTransformerEncoderForFeatureExtractionConfig):
    """
    Configuration class for RBLNT5EncoderModel.

    This configuration class stores the configuration parameters specific to
    RBLN-optimized T5 encoder models for feature extraction tasks.
    """


class RBLNT5ForConditionalGenerationConfig(RBLNModelForSeq2SeqLMConfig):
    """
    Configuration class for RBLNT5ForConditionalGeneration.

    This configuration class stores the configuration parameters specific to
    RBLN-optimized T5 models for conditional text generation tasks.
    """
