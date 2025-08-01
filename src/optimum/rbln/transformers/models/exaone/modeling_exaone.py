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


import inspect
from typing import Any, Callable

from transformers import AutoModelForCausalLM
from transformers.generation.utils import GenerationMixin

from ....utils import logging
from ..decoderonly import RBLNDecoderOnlyModelForCausalLM
from .exaone_architecture import ExaoneForCausalLMWrapper


logger = logging.get_logger(__name__)


class RBLNExaoneForCausalLM(RBLNDecoderOnlyModelForCausalLM):
    """
    The Exaone Model transformer with a language modeling head (linear layer) on top.
    This model inherits from [`RBLNDecoderOnlyModelForCausalLM`]. Check the superclass documentation for the generic methods the library implements for all its models.

    A class to convert and run pre-trained transformers based ExaoneForCausalLM model on RBLN devices.
    It implements the methods to convert a pre-trained transformers ExaoneForCausalLM model into a RBLN transformer model by:

    - transferring the checkpoint weights of the original into an optimized RBLN graph,
    - compiling the resulting graph using the RBLN compiler.

    **Configuration:**
    This model uses [`RBLNExaoneForCausalLMConfig`] for configuration. When calling methods like `from_pretrained` or `from_model`,
    the `rbln_config` parameter should be an instance of [`RBLNExaoneForCausalLMConfig`] or a dictionary conforming to its structure.

    See the [`RBLNExaoneForCausalLMConfig`] class for all available configuration options.

    Examples:
        ```python
        from optimum.rbln import RBLNExaoneForCausalLM

        # Simple usage using rbln_* arguments
        # `max_seq_len` is automatically inferred from the model config
        model = RBLNExaoneForCausalLM.from_pretrained(
            "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct",
            export=True,
            rbln_batch_size=1,
            rbln_tensor_parallel_size=4,
        )


        # Using a config dictionary
        rbln_config = {
            "batch_size": 1,
            "max_seq_len": 4096,
            "tensor_parallel_size": 4,
        }
        model = RBLNExaoneForCausalLM.from_pretrained(
            "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct",
            export=True,
            rbln_config=rbln_config
        )


        # Using a RBLNExaoneForCausalLMConfig instance (recommended for type checking)
        from optimum.rbln import RBLNExaoneForCausalLMConfig

        config = RBLNExaoneForCausalLMConfig(
            batch_size=1,
            max_seq_len=4096,
            tensor_parallel_size=4
        )
        model = RBLNExaoneForCausalLM.from_pretrained(
            "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct",
            export=True,
            rbln_config=config
        )
        ```
    """

    _decoder_wrapper_cls = ExaoneForCausalLMWrapper
    _hf_class = AutoModelForCausalLM
    _supports_cache_class = True

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        kwargs.setdefault("trust_remote_code", True)
        return super().from_pretrained(*args, **kwargs)

    def __getattr__(self, __name: str) -> Any:
        def redirect(func):
            return lambda *pargs, **kwargs: func(self, *pargs, **kwargs)

        val = getattr(GenerationMixin, __name)

        if isinstance(val, Callable) and "self" in set(inspect.signature(val).parameters):
            return redirect(val)
        return val
