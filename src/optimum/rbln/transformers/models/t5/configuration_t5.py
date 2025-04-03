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

from typing import Optional

from ...configuration_generic import RBLNTransformerEncoderForFeatureExtractionConfig
from ..seq2seq import RBLNModelForSeq2SeqLMConfig


class RBLNT5EncoderModelConfig(RBLNTransformerEncoderForFeatureExtractionConfig):
    def __init__(self, max_sequence_length: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)

        # FIXME: why need max_sequence_length ??
        self.max_seq_len = self.max_seq_len or max_sequence_length


class RBLNT5ForConditionalGenerationConfig(RBLNModelForSeq2SeqLMConfig):
    pass
