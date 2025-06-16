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
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Union

import torch
from transformers import (
    AutoModelForVision2Seq,
    PaliGemmaForConditionalGeneration,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import BaseModelOutputWithPooling

from ....configuration_utils import RBLNCompileConfig, RBLNModelConfig
from ....modeling import RBLNModel
from ....utils.logging import get_logger
from ..decoderonly.modeling_decoderonly import RBLNDecoderOnlyOutput


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

    def forward(self, *args, **kwargs):
        # Loop instead of batch
        # shape of pixel_values : [batch, num_patches, num_channel, height, width]
        pixel_values = args[0]

        batch_size = pixel_values.shape[0]
        outputs = []
        for i in range(batch_size):
            outputs.append(self.vision_tower.model[0](pixel_values[i : i + 1]))

        # FIXME:: This can be optimized using out= API of rbln runtime.
        last_hidden_states = torch.cat(outputs, dim=0)

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_states,
        )

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

    def __repr__(self) -> str:
        return repr(self.vision_tower)


class LoopProjector:
    def __init__(self, multi_modal_projector) -> None:
        self.multi_modal_projector = multi_modal_projector

    def forward(self, *args, **kwargs):
        # Loop instead of batch
        image_feature = args[0]

        batch_size = image_feature.shape[0]
        outputs = []
        for i in range(batch_size):
            outputs.append(self.multi_modal_projector(image_feature[i : i + 1]))

        # FIXME:: This can be optimized using out= API of rbln runtime.
        outputs = torch.cat(outputs, dim=0)
        return outputs

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

    def __repr__(self) -> str:
        return repr(self.multi_modal_projector)


class RBLNPaliGemmaForConditionalGeneration(RBLNModel):
    auto_model_class = AutoModelForVision2Seq
    _rbln_submodules = [
        {"name": "vision_tower"},
        {"name": "language_model"},
    ]

    def __getattr__(self, __name: str) -> Any:
        def redirect(func):
            return lambda *pargs, **kwargs: func(self, *pargs, **kwargs)

        val = getattr(PaliGemmaForConditionalGeneration, __name)

        if isinstance(val, Callable) and "self" in set(inspect.signature(val).parameters):
            return redirect(val)
        return val

    def can_generate(self):
        return True

    def __post_init__(self, **kwargs):
        self.vision_tower = LoopVisionTower(self.rbln_submodules[0])
        self.language_model = self.rbln_submodules[1]
        self.multi_modal_projector = LoopProjector(self.model[0])
        self.vocab_size = self.config.text_config.vocab_size

        # Copied from the original class
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        return super().__post_init__(**kwargs)

    def get_attn_impl(self) -> str:
        return self.rbln_config.language_model.attn_impl

    def get_kvcache_num_blocks(self) -> int:
        return self.rbln_config.language_model.kvcache_num_blocks

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    @classmethod
    def wrap_model_if_needed(cls, model: "PreTrainedModel", rbln_config: RBLNModelConfig):
        return model.multi_modal_projector

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Optional[Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"]],
        model: Optional["PreTrainedModel"] = None,
        model_config: Optional["PretrainedConfig"] = None,
        rbln_config: Optional[RBLNModelConfig] = None,
    ) -> RBLNModelConfig:
        image_feature_dim = (model_config.vision_config.image_size // model_config.vision_config.patch_size) ** 2
        feature_size = model_config.vision_config.hidden_size

        input_info = [("image_features", [rbln_config.batch_size, image_feature_dim, feature_size], "float32")]
        rbln_compile_config = RBLNCompileConfig(input_info=input_info)
        rbln_config.set_compile_cfgs([rbln_compile_config])
        return rbln_config

    def prepare_inputs_for_generation(
        self,
        input_ids,
        inputs_embeds=None,
        pixel_values=None,
        image_sizes=None,
        attention_mask=None,
        generate_idx=None,
        padded_cache_lengths=None,
        token_type_ids=None,
        **kwargs,
    ):
        # Prepare HF generation
        is_prefill_phase = generate_idx is None

        model_inputs = self.language_model.prepare_inputs_for_generation(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            generate_idx=generate_idx,  # Not affect
            attention_mask=attention_mask,
            padded_cache_lengths=padded_cache_lengths,
            **kwargs,
        )

        if is_prefill_phase:
            model_inputs.update(
                {
                    "pixel_values": pixel_values,
                    "image_sizes": image_sizes,
                    "token_type_ids": token_type_ids,
                }
            )

        model_inputs["attention_mask"] = attention_mask

        return model_inputs

    def _update_model_kwargs_for_generation(
        self,
        outputs: RBLNDecoderOnlyOutput,
        model_kwargs: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, Any]:
        # update generate_idx
        model_kwargs["generate_idx"] = outputs.generate_idx

        return model_kwargs

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

    def _preprocess_prefill(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # Replace image id woth PAD if the image token if OOV, to avoid index-errors
        if input_ids is not None and self.config.image_token_index >= self.vocab_size:
            special_image_mask = input_ids == self.config.image_token_index
            llm_input_ids = input_ids.clone()
            llm_input_ids[special_image_mask] = 0
        else:
            llm_input_ids = input_ids

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(llm_input_ids)

        # Merge text and images
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values)
            special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1)
            special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)

            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        return inputs_embeds

    def _update_causal_mask(
        self,
        attention_mask,
    ):
        if self.language_model.rbln_config.use_attention_mask:
            return attention_mask[0]
        else:
            sequence_length = torch.sum(attention_mask, dim=-1)
            target_length = self.rbln_config.language_model.max_seq_len
            prefill_chunk_size = self.rbln_config.language_model.prefill_chunk_size
            causal_mask = torch.full((prefill_chunk_size, target_length), fill_value=0, dtype=torch.float32)
            causal_mask[:, :sequence_length] = 1
            causal_mask = causal_mask[None, None, :, :]

            return causal_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        generate_idx: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **lm_kwargs,
    ) -> Union[Tuple, RBLNDecoderOnlyOutput]:
        # prefill
        if cache_position is None:
            logits = []
            inputs_embeds = self._preprocess_prefill(input_ids, inputs_embeds, pixel_values)
            batch_size = inputs_embeds.shape[0]

            for b_idx in range(batch_size):
                cache_position = torch.arange(0, generate_idx[b_idx].item(), dtype=torch.int32).unsqueeze(0)
                output = self.language_model.prefill_decoder(
                    inputs_embeds=inputs_embeds[b_idx : b_idx + 1, attention_mask[b_idx].bool()],
                    attention_mask=self._update_causal_mask(attention_mask[b_idx : b_idx + 1]),
                    cache_position=cache_position,
                    batch_idx=b_idx,
                )
                logits.append(output.logits)

            logits = torch.cat(logits, dim=0)
        # decoder
        else:
            inputs = inputs_embeds if inputs_embeds is not None else input_ids
            batch_size = inputs.shape[0]
            if batch_size not in self.language_model.decoders:
                raise ValueError(
                    f"No decoder runtime available for batch size {batch_size}. "
                    f"Available batch sizes are: {list(self.decoders.keys())}. "
                    f"Please run your model with one of these batch sizes or add support for batch size {batch_size}."
                )

            output = self.language_model.decoders[batch_size](
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                position_ids=position_ids if self.rbln_config.language_model.use_position_ids else None,
            )

        return RBLNDecoderOnlyOutput(logits=output.logits, generate_idx=generate_idx)
