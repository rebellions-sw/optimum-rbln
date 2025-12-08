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


from typing import TYPE_CHECKING, Optional, Tuple, Union

import torch
from transformers import ColQwen2Config, ColQwen2ForRetrieval
from transformers.modeling_utils import no_init_weights
from transformers.models.colqwen2.modeling_colqwen2 import ColQwen2ForRetrievalOutput

from ....modeling import RBLNModel
from ....transformers.modeling_outputs import _validate_output_hidden_states


if TYPE_CHECKING:
    from transformers import PreTrainedModel


class RBLNColQwen2ForRetrieval(RBLNModel):
    _rbln_submodule_postfix = "model"
    _rbln_submodules = [
        {"name": "vlm"},
    ]
    _supports_non_fp32 = True

    def __post_init__(self, **kwargs):
        self.vlm_model = self.rbln_submodules[0]
        return super().__post_init__(**kwargs)

    @classmethod
    def _reconstruct_model_if_needed(cls, model: "PreTrainedModel"):
        # if model is from Colpali-engine, convert it to a ColQwen2ForRetrieval model
        if hasattr(model, "custom_text_proj"):
            with no_init_weights():
                model_config = ColQwen2Config(
                    vlm_config=model.config, embedding_dim=model.custom_text_proj.out_features
                )
                new_model = ColQwen2ForRetrieval._from_config(model_config)
            new_model.embedding_proj_layer = model.custom_text_proj
            new_model.vlm.model.visual.load_state_dict(model.visual.state_dict())
            new_model.vlm.model.language_model.load_state_dict(model.language_model.state_dict())
            model = new_model

        # replace the lm_head with the custom text projection layer for optimization
        model.vlm.model.lm_head = model.embedding_proj_layer
        model.vlm.model.config.embedding_dim = model.config.embedding_dim

        # Some of the model weights are different from the model.dtype(vidore/colqwen2-v1.0-hf)
        return model.to(model.dtype)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, ColQwen2ForRetrievalOutput]:
        output_hidden_states = _validate_output_hidden_states(output_hidden_states, self.rbln_config)

        if pixel_values is not None:
            pixel_values = pixel_values.to(dtype=self.dtype)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Handle the custom "pixel_values" input obtained with `ColQwen2Processor` through unpadding
        if pixel_values is not None and image_grid_thw is not None:
            offsets = image_grid_thw[:, 1] * image_grid_thw[:, 2]  # (batch_size,)
            pixel_values = torch.cat(
                [pixel_sequence[:offset] for pixel_sequence, offset in zip(pixel_values, offsets)],
                dim=0,
            )

        vlm_output = self.vlm_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )

        embeddings = vlm_output[0]
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        if attention_mask is not None:
            embeddings = embeddings * attention_mask.unsqueeze(-1)

        return ColQwen2ForRetrievalOutput(
            embeddings=embeddings,
        )
