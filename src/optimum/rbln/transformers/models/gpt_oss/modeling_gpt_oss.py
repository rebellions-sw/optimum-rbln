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

from typing import Optional, Union, TYPE_CHECKING

from transformers import PretrainedConfig

from ...models.decoderonly import RBLNDecoderOnlyModelForCausalLM, RBLNDecoderOnlyModelForCausalLMConfig
from .gpt_oss_architecture import RBLNGptOssWrapper

if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer
    from transformers import PreTrainedModel


class RBLNGptOssForCausalLM(RBLNDecoderOnlyModelForCausalLM):
    """
    The GPT-OSS Model transformer with a language modeling head (linear layer) on top.
    This model inherits from [`RBLNDecoderOnlyModelForCausalLM`]. Check the superclass documentation for the generic methods the library implements for all its models.

    A class to convert and run pre-trained transformers based GPT-OSSForCausalLM model on RBLN devices.
    It implements the methods to convert a pre-trained transformers GPT-OSSForCausalLM model into a RBLN transformer model by:
    - transferring the checkpoint weights of the original into an optimized RBLN graph,
    - compiling the resulting graph using the RBLN compiler.

    **Configuration:**
    This model uses [`RBLNGptOssForCausalLMConfig`] for configuration. When calling methods like `from_pretrained` or `from_model`,
    the `rbln_config` parameter should be an instance of [`RBLNGptOssForCausalLMConfig`] or a dictionary conforming to its structure.

    See the [`RBLNGptOssForCausalLMConfig`] class for all available configuration options.

    Examples:
        ```python
        from optimum.rbln import RBLNGptOssForCausalLM

        # Simple usage using rbln_* arguments
        # `max_seq_len` is automatically inferred from the model config
        model = RBLNGptOssForCausalLM.from_pretrained(
            "openai/gpt-oss-20b",
            export=True,
            rbln_batch_size=1,
            rbln_tensor_parallel_size=4,
        )


        # Using a config dictionary
        rbln_config = {
            "batch_size": 1,
            "tensor_parallel_size": 4,
        }
        model = RBLNGptOssForCausalLM.from_pretrained(
            "openai/gpt-oss-20b",
            export=True,
            rbln_config=rbln_config
        )


        # Using a RBLNGptOssForCausalLMConfig instance (recommended for type checking)
        from optimum.rbln import RBLNGptOssForCausalLMConfig

        config = RBLNGptOssForCausalLMConfig(
            batch_size=1,
            tensor_parallel_size=4
        )
        model = RBLNGptOssForCausalLM.from_pretrained(
            "openai/gpt-oss-20b",
            export=True,
            rbln_config=config
        )
        ```
    """

    _decoder_wrapper_cls = RBLNGptOssWrapper

    @classmethod
    def _update_sliding_window_config(
        cls, model_config: PretrainedConfig, rbln_config: RBLNDecoderOnlyModelForCausalLMConfig
    ):
        rbln_config.cache_impl = "sliding_window"
        rbln_config.sliding_window = model_config.sliding_window
        sliding_window_layers = []
        for i in range(model_config.num_hidden_layers):
            # if model_config.layer_types[i] == "sliding_attention":
            sliding_window_layers.append(i)
        rbln_config.sliding_window_layers = sliding_window_layers

        return rbln_config

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Optional[Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"]] = None,
        model: Optional["PreTrainedModel"] = None,
        model_config: Optional["PretrainedConfig"] = None,
        rbln_config: Optional[RBLNDecoderOnlyModelForCausalLMConfig] = None,
    ) -> RBLNDecoderOnlyModelForCausalLMConfig:
        rbln_config = super()._update_rbln_config(preprocessors, model, model_config, rbln_config)

        if rbln_config.use_attention_mask:
            raise ValueError(
                "use_attention_mask is not supported for GPT-OSS because custom attention does not support attention sink for masked attention"
            )

        return rbln_config
