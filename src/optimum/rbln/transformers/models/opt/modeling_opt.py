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

import torch.nn as nn
from transformers import PreTrainedModel

from ....utils import logging
from ...models.decoderonly import RBLNDecoderOnlyModelForCausalLM
from ...models.decoderonly.configuration_decoderonly import RBLNDecoderOnlyModelForCausalLMConfig
from .opt_architecture import OPTWrapper


logger = logging.get_logger(__name__)


class MLP(nn.Module):
    def __init__(self, fc1, fc2, activation_fn):
        super(MLP, self).__init__()
        self.fc1 = fc1
        self.fc2 = fc2
        self.activation_fn = activation_fn

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        return x


class RBLNOPTForCausalLM(RBLNDecoderOnlyModelForCausalLM):
    """
    The OPT Model transformer with a language modeling head (linear layer) on top.
    This model inherits from [`RBLNDecoderOnlyModelForCausalLM`]. Check the superclass documentation for the generic methods the library implements for all its models.

    A class to convert and run pre-trained transformers based OPTForCausalLM model on RBLN devices.
    It implements the methods to convert a pre-trained transformers OPTForCausalLM model into a RBLN transformer model by:
    - transferring the checkpoint weights of the original into an optimized RBLN graph,
    - compiling the resulting graph using the RBLN compiler.
    """

    _decoder_wrapper_cls = OPTWrapper
    _use_rotary_emb = False

    def modify_opt_decoder_layer(layer):
        mlp = MLP(layer.fc1, layer.fc2, layer.activation_fn)
        layer.mlp = mlp
        del layer.fc1
        del layer.fc2
        del layer.activation_fn

        return layer

    @classmethod
    def wrap_model_if_needed(cls, model: PreTrainedModel, rbln_config: RBLNDecoderOnlyModelForCausalLMConfig):
        wrapper_cfg = {
            "max_seq_len": rbln_config.max_seq_len,
            "attn_impl": rbln_config.attn_impl,
            "kvcache_partition_len": rbln_config.kvcache_partition_len,
            "kvcache_block_size": rbln_config.kvcache_block_size,
            "use_rotary_emb": cls._use_rotary_emb,
            "use_attention_mask": rbln_config.use_attention_mask,
            "use_position_ids": rbln_config.use_position_ids,
            "use_inputs_embeds": rbln_config.use_inputs_embeds,
        }

        for i in range(len(model.model.decoder.layers)):
            model.model.decoder.layers[i] = cls.modify_opt_decoder_layer(model.model.decoder.layers[i])

        return cls._decoder_wrapper_cls(model, **wrapper_cfg).eval()
