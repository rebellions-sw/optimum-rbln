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

from transformers import PretrainedConfig

from ....utils import logging
from ...models.decoderonly import RBLNDecoderOnlyModelForCausalLM, RBLNDecoderOnlyModelForCausalLMConfig
from .mistral_architecture import MistralForCausalLMWrapper


logger = logging.get_logger(__name__)


class RBLNMistralForCausalLM(RBLNDecoderOnlyModelForCausalLM):
    """
    The Mistral Model transformer with a language modeling head (linear layer) on top.
    This model inherits from [`RBLNDecoderOnlyModelForCausalLM`]. Check the superclass documentation for the generic methods the library implements for all its models.

    A class to convert and run pre-trained transformers based MistralForCausalLM model on RBLN devices.
    It implements the methods to convert a pre-trained transformers MistralForCausalLM model into a RBLN transformer model by:
    - transferring the checkpoint weights of the original into an optimized RBLN graph,
    - compiling the resulting graph using the RBLN compiler.

    **Configuration:**
    This model uses [`RBLNMistralForCausalLMConfig`] for configuration. When calling methods like `from_pretrained` or `from_model`,
    the `rbln_config` parameter should be an instance of [`RBLNMistralForCausalLMConfig`] or a dictionary conforming to its structure.

    See the [`RBLNMistralForCausalLMConfig`] class for all available configuration options.

    Examples:
        ```python
        from optimum.rbln import RBLNMistralForCausalLM

        # Simple usage using rbln_* arguments
        # `max_seq_len` is automatically inferred from the model config
        model = RBLNMistralForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-v0.1",
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
        model = RBLNMistralForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-v0.1",
            export=True,
            rbln_config=rbln_config
        )

        # Using a RBLNMistralForCausalLMConfig instance (recommended for type checking)
        from optimum.rbln import RBLNMistralForCausalLMConfig

        config = RBLNMistralForCausalLMConfig(
            batch_size=1,
            max_seq_len=4096,
            tensor_parallel_size=4
        )
        model = RBLNMistralForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-v0.1",
            export=True,
            rbln_config=config
        )
        ```
    """

    _decoder_wrapper_cls = MistralForCausalLMWrapper

    @classmethod
    def _update_sliding_window_config(
        cls, model_config: PretrainedConfig, rbln_config: RBLNDecoderOnlyModelForCausalLMConfig
    ):
        rbln_config.cache_impl = "sliding_window"
        rbln_config.sliding_window = model_config.sliding_window
        rbln_config.sliding_window_layers = list(range(model_config.num_hidden_layers))

        return rbln_config
