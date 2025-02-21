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

from ....utils import logging
from ...models.decoderonly import RBLNDecoderOnlyModelForCausalLM
from .mistral_architecture import MistralForCausalLMWrapper


logger = logging.get_logger(__name__)


class RBLNMistralForCausalLM(RBLNDecoderOnlyModelForCausalLM):
    """
    The Llama Model transformer with a language modeling head (linear layer) on top.
    This model inherits from [`RBLNDecoderOnlyModelForCausalLM`]. Check the superclass documentation for the generic methods the library implements for all its models.

    A class to convert and run pre-trained transformers based LlamaForCausalLM model on RBLN devices.
    It implements the methods to convert a pre-trained transformers LlamaForCausalLM model into a RBLN transformer model by:
    - transferring the checkpoint weights of the original into an optimized RBLN graph,
    - compiling the resulting graph using the RBLN compiler.
    """

    _decoder_wrapper_cls = MistralForCausalLMWrapper
