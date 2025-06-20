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
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import torch
from transformers import (
    ColPaliForRetrieval,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_utils import no_init_weights
from transformers.models.colpali.modeling_colpali import ColPaliForRetrievalOutput
from transformers.models.paligemma.modeling_paligemma import PaliGemmaMultiModalProjector
from transformers.modeling_outputs import BaseModelOutputWithPooling

from ....configuration_utils import RBLNCompileConfig, RBLNModelConfig
from ....modeling import RBLNModel
from ....utils.logging import get_logger
from .colpali_architecture import RBLNColPaliForRetrievalWrapper


logger = get_logger(__name__)

if TYPE_CHECKING:
    from transformers import (
        AutoFeatureExtractor,
        AutoProcessor,
        AutoTokenizer,
        PretrainedConfig,
    )


class LoopVisionTower:
    def __init__(self, vision_tower: RBLNModel) -> None:
        self.vision_tower = vision_tower

    def forward(self, pixel_values, **kwargs):
        # Loop instead of batch
        # shape of pixel_values : [batch, num_patches, num_channel, height, width]
        batch_size = pixel_values.shape[0]
        outputs = []
        for i in range(batch_size):
            outputs.append(self.vision_tower(pixel_values[i : i + 1]))

        last_hidden_states = [output.last_hidden_state for output in outputs]
        last_hidden_states = torch.cat(last_hidden_states, dim=0)

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_states,
        )

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

    def __repr__(self) -> str:
        return repr(self.vision_tower)


class LoopLanguageModel:
    def __init__(self, language_model: RBLNModel, rbln_config: RBLNModelConfig) -> None:
        self.language_model = language_model
        self.rbln_config = rbln_config

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor, **kwargs):
        embeddings = []
        for i in range(inputs_embeds.shape[0]):
            inputs_embed = torch.nn.functional.pad(
                inputs_embeds[i : i + 1], (0, 0, 0, self.rbln_config.max_seq_len - inputs_embeds.shape[1])
            )
            attn_mask = torch.nn.functional.pad(
                attention_mask[i : i + 1], (0, self.rbln_config.max_seq_len - attention_mask.shape[1])
            ).to(torch.float32)
            embeddings.append(self.language_model(inputs_embeds=inputs_embed, attention_mask=attn_mask))
        embeddings = torch.cat(embeddings, dim=0)[:, : attention_mask.shape[-1]]

        return embeddings

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

    def __repr__(self) -> str:
        return repr(self.language_model)


class RBLNColPaliForRetrieval(RBLNModel):
    auto_model_class = None
    _rbln_submodules = [
        {"name": "vision_tower"},
    ]

    def __post_init__(self, **kwargs):
        self.vision_tower = LoopVisionTower(self.rbln_submodules[0])
        self.language_model = LoopLanguageModel(self.model[0], self.rbln_config)

        artifacts = torch.load(self.model_save_dir / self.subfolder / "torch_artifacts.pth", weights_only=False)
        self.embed_tokens = self._create_embedding_layer()
        self.embed_tokens.load_state_dict(artifacts["embed_tokens"])
        self.multi_modal_projector = self._create_multi_modal_projector()
        self.multi_modal_projector.load_state_dict(artifacts["multi_modal_projector"])

        return super().__post_init__(**kwargs)

    def _create_embedding_layer(self):
        with no_init_weights():
            embed_tokens = torch.nn.Embedding(
                self.config.text_config.vocab_size,
                self.config.text_config.hidden_size,
                self.config.text_config.pad_token_id,
            )
        return embed_tokens

    def _create_multi_modal_projector(self):
        with no_init_weights():
            multi_modal_projector = PaliGemmaMultiModalProjector(self.config.vlm_config)
        return multi_modal_projector

    def __getattr__(self, __name: str) -> Any:
        def redirect(func):
            return lambda *pargs, **kwargs: func(self, *pargs, **kwargs)

        val = getattr(ColPaliForRetrieval, __name)

        if isinstance(val, Callable) and "self" in set(inspect.signature(val).parameters):
            return redirect(val)
        return val

    def can_generate(self):
        return False

    @classmethod
    def wrap_model_if_needed(cls, model: "PreTrainedModel", rbln_config: RBLNModelConfig):
        return RBLNColPaliForRetrievalWrapper(
            language_model=model.vlm.model.language_model,
            embedding_proj_layer=model.embedding_proj_layer,
            max_seq_len=rbln_config.max_seq_len,
            output_hidden_states=rbln_config.output_hidden_states,
        )

    @classmethod
    def save_torch_artifacts(
        cls,
        model: "PreTrainedModel",
        save_dir_path: Path,
        subfolder: str,
        rbln_config: RBLNModelConfig,
    ):
        save_dict = {}
        save_dict["embed_tokens"] = model.vlm.get_input_embeddings().state_dict()
        save_dict["multi_modal_projector"] = model.vlm.model.multi_modal_projector.state_dict()
        torch.save(save_dict, save_dir_path / subfolder / "torch_artifacts.pth")

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Optional[Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"]],
        model: Optional["PreTrainedModel"] = None,
        model_config: Optional["PretrainedConfig"] = None,
        rbln_config: Optional[RBLNModelConfig] = None,
    ) -> RBLNModelConfig:
        hidden_size = model_config.vlm_config.text_config.hidden_size

        input_info = [
            ("inputs_embeds", [rbln_config.batch_size, rbln_config.max_seq_len, hidden_size], "float32"),
            ("attention_mask", [rbln_config.batch_size, rbln_config.max_seq_len], "float32"),
        ]
        rbln_compile_config = RBLNCompileConfig(input_info=input_info)
        rbln_config.set_compile_cfgs([rbln_compile_config])

        return rbln_config

    @classmethod
    def get_pytorch_model(cls, *args, **kwargs):
        model = super().get_pytorch_model(*args, **kwargs)
        model.vision_tower = model.vlm.model.vision_tower

        return model

    def get_image_features(self, pixel_values: torch.Tensor):
        """
        Projects the last hidden state from the vision model into language model space.
        Args:
            pixel_values (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`)
               The tensors corresponding to the input images.
        Returns:
            image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
        """
        vision_outputs = self.vision_tower(pixel_values).last_hidden_state
        image_features = self.multi_modal_projector(vision_outputs)
        image_features = image_features / (self.config.text_config.hidden_size**0.5)
        return image_features

    def _preprocess_inputs(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # Replace image id woth PAD if the image token if OOV, to avoid index-errors
        if input_ids is not None and self.config.vlm_config.image_token_index >= self.config.text_config.vocab_size:
            special_image_mask = input_ids == self.config.vlm_config.image_token_index
            llm_input_ids = input_ids.clone()
            llm_input_ids[special_image_mask] = 0
        else:
            llm_input_ids = input_ids

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(llm_input_ids)

        # Merge text and images
        image_features = None
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values)
            special_image_mask = (input_ids == self.config.vlm_config.image_token_index).unsqueeze(-1)
            special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)

            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        return inputs_embeds, image_features

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
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

        inputs_embeds, image_features = self._preprocess_inputs(
            input_ids=input_ids, inputs_embeds=inputs_embeds, pixel_values=pixel_values
        )

        embeddings = self.language_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        # L2 normalization
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)  # (batch_size, sequence_length, dim)

        if attention_mask is not None:
            embeddings = embeddings * attention_mask.unsqueeze(-1)  # (batch_size, sequence_length, dim)

        return ColPaliForRetrievalOutput(
            embeddings=embeddings,
            # hidden_states=vlm_hidden_states,
        )
