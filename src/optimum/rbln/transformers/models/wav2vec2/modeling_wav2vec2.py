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
from transformers import AutoModelForMaskedLM, Wav2Vec2ForCTC

from ...modeling_generic import RBLNModelForMaskedLM
from .configuration_wav2vec2 import RBLNWav2Vec2ForCTCConfig


class _Wav2Vec2(torch.nn.Module):
    def __init__(self, model: "Wav2Vec2ForCTC"):
        super().__init__()
        self.model = model

    def forward(self, input_values):
        output = self.model.wav2vec2(input_values=input_values)
        return self.model.lm_head(output[0])


class RBLNWav2Vec2ForCTC(RBLNModelForMaskedLM):
    """
    Wav2Vec2 Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC).

    This model inherits from [`RBLNModelForMaskedLM`]. Check the superclass documentation for the generic methods the
    library implements for all its model.

    It implements the methods to convert a pre-trained Wav2Vec2 model into a RBLN Wav2Vec2 model by:
    - transferring the checkpoint weights of the original into an optimized RBLN graph,
    - compiling the resulting graph using the RBLN compiler.
    """

    main_input_name = "input_values"
    auto_model_class = AutoModelForMaskedLM
    rbln_dtype = "float32"

    @classmethod
    def wrap_model_if_needed(cls, model: torch.nn.Module, rbln_config: RBLNWav2Vec2ForCTCConfig) -> torch.nn.Module:
        return _Wav2Vec2(model).eval()
