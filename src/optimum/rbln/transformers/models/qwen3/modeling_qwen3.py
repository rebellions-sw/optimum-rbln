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

from typing import TYPE_CHECKING

from transformers import PretrainedConfig

from ....utils import logging
from ...models.decoderonly import (
    RBLNDecoderOnlyModel,
    RBLNDecoderOnlyModelForCausalLM,
    RBLNDecoderOnlyModelForCausalLMConfig,
)
from .qwen3_architecture import Qwen3Wrapper


logger = logging.get_logger(__name__)

if TYPE_CHECKING:
    from transformers import (
        PretrainedConfig,
    )


class RBLNQwen3ForCausalLM(RBLNDecoderOnlyModelForCausalLM):
    _decoder_wrapper_cls = Qwen3Wrapper

    @classmethod
    def _update_sliding_window_config(
        cls, model_config: PretrainedConfig, rbln_config: RBLNDecoderOnlyModelForCausalLMConfig
    ):
        # https://github.com/huggingface/transformers/issues/35896
        # There seems to be a bug in transformers(v4.52.4). Therefore, similar to when attn_implementation is eager,
        # we set all layers to use sliding window in this version. This should be updated once the bug is fixed.

        rbln_config.cache_impl = "sliding_window"
        rbln_config.sliding_window = model_config.sliding_window
        rbln_config.sliding_window_layers = list(range(model_config.num_hidden_layers))
        return rbln_config

    def forward(self, *args, **kwargs):
        kwargs["return_dict"] = True
        return super().forward(*args, **kwargs)


class RBLNQwen3Model(RBLNDecoderOnlyModel):
    _decoder_wrapper_cls = Qwen3Wrapper
    _use_rotary_emb = True
