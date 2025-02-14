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


from transformers import AutoModelForCausalLM

from ....utils import logging
from ..decoderonly import RBLNDecoderOnlyModelForCausalLM
from .exaone_architecture import ExaoneForCausalLMWrapper


logger = logging.get_logger(__name__)


class RBLNExaoneForCausalLM(RBLNDecoderOnlyModelForCausalLM):
    """
    The Exaone Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).

    This model inherits from [`RBLNDecoderOnlyModelForCausalLM`]. Check the superclass documentation for the generic methods the
    library implements for all its model.

    It implements the methods to convert a pre-trained transformers Exaone model into a RBLN transformer model by:
    - transferring the checkpoint weights of the original into an optimized RBLN graph,
    - compiling the resulting graph using the RBLN compiler.

    """

    _decoder_wrapper_cls = ExaoneForCausalLMWrapper
    _hf_class = AutoModelForCausalLM

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        kwargs.setdefault("trust_remote_code", True)
        return super().from_pretrained(*args, **kwargs)
