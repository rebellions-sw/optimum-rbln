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

import bisect
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

import torch
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.modeling_utils import no_init_weights
from transformers.models.colpali.modeling_colpali import ColPaliForRetrievalOutput
from transformers.models.paligemma.modeling_paligemma import PaliGemmaMultiModalProjector

from ....configuration_utils import RBLNCompileConfig, RBLNModelConfig
from ....modeling import RBLNModel
from .colpali_architecture import RBLNColPaliForRetrievalWrapper


if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer, PretrainedConfig


class LoopVisionTower:
    def __init__(self, vision_tower: RBLNModel) -> None:
        self.vision_tower = vision_tower

    def forward(self, pixel_values, **kwargs):
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

    def prepare_inputs(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor):
        input_len = inputs_embeds.shape[1]
        idx = bisect.bisect_left(self.rbln_config.max_seq_lens, input_len)
        if idx == len(self.rbln_config.max_seq_lens):
            raise ValueError(
                f"Required seq_len({input_len}) is larger than available max_seq_lens({self.rbln_config.max_seq_lens})."
            )
        else:
            max_seq_len = self.rbln_config.max_seq_lens[idx]

        inputs_embed = torch.nn.functional.pad(inputs_embeds, (0, 0, 0, max_seq_len - input_len))
        attn_mask = torch.nn.functional.pad(attention_mask, (0, max_seq_len - input_len)).to(torch.float32)
        position_ids = torch.arange(max_seq_len, dtype=torch.int32).view(1, -1)

        return inputs_embed, attn_mask, position_ids

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor, **kwargs):
        padded_inputs_embed, padded_attn_mask, padded_position_ids = self.prepare_inputs(inputs_embeds, attention_mask)
        input_batch_size = inputs_embeds.shape[0]
        input_seq_len = inputs_embeds.shape[1]

        all_embeddings = []
        all_hidden_states = []
        for i in range(input_batch_size):
            outputs = self.language_model(
                inputs_embeds=padded_inputs_embed[i : i + 1],
                attention_mask=padded_attn_mask[i : i + 1],
                position_ids=padded_position_ids,
            )

            if self.rbln_config.output_hidden_states:
                embedding = outputs[0]
                hidden_states = outputs[1:]
            else:
                embedding = outputs
                hidden_states = None

            all_embeddings.append(embedding)
            all_hidden_states.append(hidden_states)

        embeddings = torch.cat(all_embeddings, dim=0)[:, :input_seq_len]
        if self.rbln_config.output_hidden_states:
            hidden_states = [
                torch.cat(
                    [batch_hidden_states[layer_idx][:, :input_seq_len] for batch_hidden_states in all_hidden_states],
                    dim=0,
                )
                for layer_idx in range(len(all_hidden_states[0]))
            ]
            return embeddings, tuple(hidden_states)
        else:
            return embeddings

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

    def __repr__(self) -> str:
        return repr(self.language_model)


class RBLNColPaliForRetrieval(RBLNModel):
    """
    The ColPali Model transformer for document retrieval using vision-language models.
    This model inherits from [`RBLNModel`]. Check the superclass documentation for the generic methods the library implements for all its models.

    A class to convert and run pre-trained transformers based ColPaliForRetrieval model on RBLN devices.
    It implements the methods to convert a pre-trained transformers ColPaliForRetrieval model into a RBLN transformer model by:

    - transferring the checkpoint weights of the original into an optimized RBLN graph,
    - compiling the resulting graph using the RBLN compiler.

    **Configuration:**
    This model uses [`RBLNColPaliForRetrievalConfig`] for configuration. When calling methods like `from_pretrained` or `from_model`,
    the `rbln_config` parameter should be an instance of [`RBLNColPaliForRetrievalConfig`] or a dictionary conforming to its structure.

    See the [`RBLNColPaliForRetrievalConfig`] class for all available configuration options.

    Examples:
        ```python
        from optimum.rbln import RBLNColPaliForRetrieval

        # Simple usage using rbln_* arguments
        # `max_seq_lens` is automatically inferred from the model config
        model = RBLNColPaliForRetrieval.from_pretrained(
            "vidore/colpali-v1.3-hf",
            export=True,
            rbln_max_seq_lens=1152,
        )

        # Using a config dictionary
        rbln_config = {
            "max_seq_lens": 1152,
            "output_hidden_states": False,
        }
        model = RBLNColPaliForRetrieval.from_pretrained(
            "vidore/colpali-v1.3-hf",
            export=True,
            rbln_config=rbln_config
        )

        # Using a RBLNColPaliForRetrievalConfig instance (recommended for type checking)
        from optimum.rbln import RBLNColPaliForRetrievalConfig

        config = RBLNColPaliForRetrievalConfig(
            max_seq_lens=1152,
            output_hidden_states=False,
            tensor_parallel_size=4
        )
        model = RBLNColPaliForRetrieval.from_pretrained(
            "vidore/colpali-v1.3-hf",
            export=True,
            rbln_config=config
        )
        ```
    """

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

    @classmethod
    def wrap_model_if_needed(cls, model: "PreTrainedModel", rbln_config: RBLNModelConfig):
        return RBLNColPaliForRetrievalWrapper(
            causal_lm=model.vlm.language_model,
            embedding_proj_layer=model.embedding_proj_layer,
            max_seq_len=max(rbln_config.max_seq_lens),
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
        save_dict["multi_modal_projector"] = model.vlm.multi_modal_projector.state_dict()
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
        if rbln_config.max_seq_lens is None:
            rbln_config.max_seq_lens = [model_config.vlm_config.text_config.max_position_embeddings]
        if isinstance(rbln_config.max_seq_lens, int):
            rbln_config.max_seq_lens = [rbln_config.max_seq_lens]
        rbln_config.max_seq_lens = sorted(set(rbln_config.max_seq_lens))

        if rbln_config.output_hidden_states is None:
            rbln_config.output_hidden_states = model_config.vlm_config.text_config.output_hidden_states

        input_infos = []
        for max_seq_len in rbln_config.max_seq_lens:
            input_info = [
                ("inputs_embeds", [1, max_seq_len, hidden_size], "float32"),
                ("attention_mask", [1, max_seq_len], "float32"),
                ("position_ids", [1, max_seq_len], "int32"),
            ]
            input_infos.append(input_info)

        rbln_compile_config = RBLNCompileConfig(input_info=input_infos)
        rbln_config.set_compile_cfgs([rbln_compile_config])

        return rbln_config

    @classmethod
    def from_model(cls, model: "PreTrainedModel", *args, **kwargs):
        if not hasattr(model, "vision_tower"):
            model.vision_tower = model.vlm.vision_tower
            del model.vlm.vision_tower
        model = super().from_model(model, *args, **kwargs)
        return model

    @classmethod
    def get_pytorch_model(cls, *args, **kwargs):
        model = super().get_pytorch_model(*args, **kwargs)
        model.vision_tower = model.vlm.vision_tower
        del model.vlm.vision_tower

        return model

    def get_image_features(self, pixel_values: torch.Tensor):
        # Projects the last hidden state from the vision model into language model space.
        # Args:
        #     pixel_values (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`)
        #        The tensors corresponding to the input images.
        # Returns:
        #     image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).

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

        if output_attentions:
            raise ValueError("output_attentions is not supported for RBLNColPaliForRetrieval")

        if output_hidden_states is not None and output_hidden_states != self.rbln_config.output_hidden_states:
            raise ValueError(
                f"Variable output_hidden_states {output_hidden_states} is not equal to rbln_config.output_hidden_states {self.rbln_config.output_hidden_states} "
                f"Please compile again with the correct argument."
            )

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        inputs_embeds, image_features = self._preprocess_inputs(
            input_ids=input_ids, inputs_embeds=inputs_embeds, pixel_values=pixel_values
        )

        # Embedding_proj_layer is fused on the bottom of the language model.
        outputs = self.language_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

        embeddings = outputs if not self.rbln_config.output_hidden_states else outputs[0]
        hidden_states = None if not self.rbln_config.output_hidden_states else outputs[1]

        # L2 normalization
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)  # (batch_size, sequence_length, dim)

        if attention_mask is not None:
            embeddings = embeddings * attention_mask.unsqueeze(-1)  # (batch_size, sequence_length, dim)

        if not return_dict:
            return (embeddings, hidden_states, image_features)
        else:
            return ColPaliForRetrievalOutput(
                embeddings=embeddings,
                hidden_states=hidden_states,
                image_hidden_states=image_features,
            )
