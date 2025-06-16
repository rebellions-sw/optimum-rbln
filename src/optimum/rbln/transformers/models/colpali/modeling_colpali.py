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

import importlib
import inspect
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import torch
from transformers import (
    ColPaliForRetrieval,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_utils import no_init_weights
from transformers.models.colpali.modeling_colpali import ColPaliForRetrievalOutput

from ....configuration_utils import RBLNCompileConfig, RBLNModelConfig
from ....modeling import RBLNModel
from ....utils.logging import get_logger


logger = get_logger(__name__)

if TYPE_CHECKING:
    from transformers import (
        AutoFeatureExtractor,
        AutoProcessor,
        AutoTokenizer,
        PretrainedConfig,
    )


class RBLNColPaliForRetrieval(RBLNModel):
    auto_model_class = None
    _rbln_submodules = [
        {"name": "vlm"},
    ]

    def __post_init__(self, **kwargs):
        self.vlm = self.rbln_submodules[0]
        self.embedding_proj_layer = self.model[0]

        return super().__post_init__(**kwargs)

    def __getattr__(self, __name: str) -> Any:
        def redirect(func):
            return lambda *pargs, **kwargs: func(self, *pargs, **kwargs)

        val = getattr(ColPaliForRetrieval, __name)

        if isinstance(val, Callable) and "self" in set(inspect.signature(val).parameters):
            return redirect(val)
        return val

    def can_generate(self):
        return False

    def get_attn_impl(self) -> str:
        return self.rbln_config.vlm.language_model.attn_impl

    def get_kvcache_num_blocks(self) -> int:
        return self.rbln_config.vlm.language_model.kvcache_num_blocks

    def get_input_embeddings(self):
        return self.vlm.language_model.get_input_embeddings()

    @classmethod
    def wrap_model_if_needed(cls, model: "PreTrainedModel", rbln_config: RBLNModelConfig):
        return model.embedding_proj_layer

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Optional[Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"]],
        model: Optional["PreTrainedModel"] = None,
        model_config: Optional["PretrainedConfig"] = None,
        rbln_config: Optional[RBLNModelConfig] = None,
    ) -> RBLNModelConfig:
        hidden_size = model_config.vlm_config.text_config.hidden_size
        seq_len = rbln_config.vlm.language_model.max_seq_len
        input_info = [("last_hidden_state", [rbln_config.batch_size, seq_len, hidden_size], "float32")]
        rbln_compile_config = RBLNCompileConfig(input_info=input_info)
        rbln_config.set_compile_cfgs([rbln_compile_config])

        return rbln_config

    @classmethod
    def get_pytorch_model(cls, *args, **kwargs):
        model = super().get_pytorch_model(*args, **kwargs)

        with no_init_weights():
            model_cls_name = model.vlm.language_model.__class__.__name__
            causal_model_cls_name = model_cls_name.replace("Model", "ForCausalLM")
            causal_model_cls = getattr(importlib.import_module("transformers"), causal_model_cls_name)
            new_text_model = causal_model_cls(model.vlm.language_model.config)

        new_text_model.model = model.vlm.language_model
        model.vlm.model.language_model = new_text_model
        model.vlm.model.lm_head = None
        del model.vlm.model.lm_head
        return model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> ColPaliForRetrievalOutput:
        if pixel_values is not None:
            pixel_values = pixel_values.to(dtype=self.dtype)

        if output_attentions is not None:
            logger.warning("output_attentions is not supported for RBLNColPaliForRetrieval")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vlm_output = self.vlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
            output_attentions=output_attentions,
            **kwargs,
        )
        vlm_hidden_states = vlm_output.hidden_states if output_hidden_states else None
        last_hidden_states = vlm_output.hidden_states[-1]  # (batch_size, sequence_length, hidden_size)

        embeddings = []
        for i in range(last_hidden_states.shape[0]):
            embeddings.append(self.embedding_proj_layer(last_hidden_states[i : i + 1]))
        embeddings = torch.cat(embeddings, dim=0)[:, : attention_mask.shape[-1]]

        # L2 normalization
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)  # (batch_size, sequence_length, dim)

        if attention_mask is not None:
            embeddings = embeddings * attention_mask.unsqueeze(-1)  # (batch_size, sequence_length, dim)

        return ColPaliForRetrievalOutput(
            embeddings=embeddings,
            hidden_states=vlm_hidden_states,
        )
