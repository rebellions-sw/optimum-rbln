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
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple, Union

import torch
from transformers import AutoModelForImageTextToText, LlavaForConditionalGeneration, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.llava.modeling_llava import LlavaCausalLMOutputWithPast

from ....configuration_utils import RBLNCompileConfig, RBLNModelConfig
from ....modeling import RBLNModel
from ....utils.logging import get_logger
from ..decoderonly.modeling_decoderonly import RBLNDecoderOnlyForCausalLMOutput


logger = get_logger(__name__)

if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer, PretrainedConfig


class LoopVisionTower:
    def __init__(self, vision_tower: RBLNModel) -> None:
        self.vision_tower = vision_tower

    def forward(self, pixel_values, image_sizes: Optional[torch.Tensor] = None, **kwargs):
        outputs = []
        for i in range(pixel_values.shape[0]):
            outputs.append(
                self.vision_tower(
                    pixel_values[i : i + 1], image_sizes[i : i + 1] if image_sizes is not None else None, **kwargs
                )
            )

        if hasattr(self.vision_tower.rbln_config, "max_image_size"):
            last_hidden_states = [output.last_hidden_state for output in outputs]
            last_hidden_states = torch.cat(last_hidden_states, dim=1)
            hidden_states = tuple(
                torch.cat(
                    [output.hidden_states[layer_idx] for output in outputs],
                    dim=1,
                )
                for layer_idx in range(len(outputs[0].hidden_states))
            )

        else:
            last_hidden_states = [output.last_hidden_state for output in outputs]
            last_hidden_states = torch.cat(last_hidden_states, dim=0)
            hidden_states = [output.hidden_states for output in outputs]
            hidden_states = tuple(
                torch.cat(tuple((hidden_states[n][i] for n in range(pixel_values.shape[0]))), dim=0)
                for i in range(len(hidden_states[0]))
            )

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_states,
            hidden_states=hidden_states,
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

        outputs = []
        for i in range(image_feature.shape[0]):
            outputs.append(self.multi_modal_projector(image_feature[i : i + 1]))

        # FIXME:: This can be optimized using out= API of rbln runtime.
        outputs = torch.cat(outputs, dim=0)
        return outputs

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

    def __repr__(self) -> str:
        return repr(self.multi_modal_projector)


class RBLNLlavaForConditionalGeneration(RBLNModel):
    """
    RBLNLlavaForConditionalGeneration is a multi-modal model that combines vision and language processing capabilities,
    optimized for RBLN NPUs. It is designed for conditional generation tasks that involve both image and text inputs.
    This model inherits from [`RBLNModel`]. Check the superclass documentation for the generic methods the library implements for all its models.
    Important Note:
        This model includes a Large Language Model (LLM) as a submodule. For optimal performance, it is highly recommended to use
        tensor parallelism for the language model. This can be achieved by using the `rbln_config` parameter in the
        `from_pretrained` method. Refer to the `from_pretrained` documentation and the RBLNLlavaForConditionalGeneration class for details.
    Examples:
        ```python
        from optimum.rbln import RBLNLlavaForConditionalGeneration
        model = RBLNLlavaForConditionalGeneration.from_pretrained(
            "llava-hf/llava-1.5-7b-hf",
            export=True,
            rbln_config={
                "vision_tower": {"output_hidden_states": True},
                "language_model": {
                    "tensor_parallel_size": 4,
                    "use_inputs_embeds": True,  # In Llava, language model must use inputs_embeds as input.
                },
            },
        )
        model.save_pretrained("compiled-llava-1.5-7b-hf")

        # Using a RBLNLlavaForConditionalGenerationConfig instance (recommended for type checking)
        from optimum.rbln import RBLNLlavaForConditionalGenerationConfig
        vision_config = RBLNCLIPVisionModelConfig(
            batch_size=1,
            output_hidden_states=True
        )
        language_model_config = RBLNLlamaForCausalLMConfig(
            batch_size=1,
            max_seq_len=4096,
            use_inputs_embeds=True,
            tensor_parallel_size=4
        )
        llava_config = RBLNLlavaForConditionalGenerationConfig(
            batch_size=1,
            vision_tower=vision_config,
            language_model=language_model_config
        )
        model = RBLNLlavaForConditionalGeneration.from_pretrained(
            "llava-hf/llava-1.5-7b-hf",
            export=True,
            rbln_config=llava_config
        )
        ```
    """

    auto_model_class = AutoModelForImageTextToText
    _rbln_submodules = [
        {"name": "vision_tower"},
        {"name": "language_model"},
    ]

    def __getattr__(self, __name: str) -> Any:
        def redirect(func):
            return lambda *pargs, **kwargs: func(self, *pargs, **kwargs)

        val = getattr(LlavaForConditionalGeneration, __name)

        if isinstance(val, Callable) and "self" in set(inspect.signature(val).parameters):
            return redirect(val)
        return val

    def can_generate(self):
        return True

    def __post_init__(self, **kwargs):
        self.vision_tower = LoopVisionTower(self.rbln_submodules[0])
        self.language_model = self.rbln_submodules[1]
        self.multi_modal_projector = LoopProjector(self.model[0])
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
        # support for pixtral that needs padding
        if hasattr(rbln_config.vision_tower, "max_image_size"):
            num_positions = (
                rbln_config.vision_tower.batch_size
                * (rbln_config.vision_tower.max_image_size[0] // model_config.vision_config.patch_size)
                * (rbln_config.vision_tower.max_image_size[1] // model_config.vision_config.patch_size)
            )
            selected_image_feature_dim = num_positions

        else:
            num_positions = (model_config.vision_config.image_size // model_config.vision_config.patch_size) ** 2 + 1
            if model_config.vision_feature_select_strategy == "default":
                selected_image_feature_dim = num_positions - 1
            else:
                selected_image_feature_dim = num_positions

        input_info = [
            (
                "image_features",
                [rbln_config.batch_size, selected_image_feature_dim, model_config.vision_config.hidden_size],
                "float32",
            )
        ]

        rbln_compile_config = RBLNCompileConfig(input_info=input_info)
        rbln_config.set_compile_cfgs([rbln_compile_config])
        return rbln_config

    def prepare_inputs_for_generation(
        self,
        input_ids,
        inputs_embeds=None,
        pixel_values=None,
        attention_mask=None,
        cache_position=None,
        image_sizes=None,
        generate_idx=None,
        **kwargs,
    ):
        is_prefill_phase = generate_idx is None
        model_inputs = {}

        if is_prefill_phase:
            generate_idx = attention_mask.sum(dim=-1, keepdim=True).int()
            cache_position = None
            pixel_values = pixel_values
            model_inputs.update({"image_sizes": image_sizes})
        else:
            if inputs_embeds is not None:
                raise NotImplementedError("Specifying inputs_embeds in decoder phase is not supported.")

            pixel_values = None
            input_ids = input_ids[:, -1:]
            cache_position = generate_idx
            generate_idx = generate_idx + 1
            model_inputs.update({"input_ids": input_ids})

        if inputs_embeds is not None:
            if self.rbln_config.use_inputs_embeds:
                model_inputs.update({"inputs_embeds": inputs_embeds})
            else:
                raise ValueError(
                    "The specifying inputs_embeds is only supported when using a compiled RBLN model with 'rbln_use_inputs_embeds' set to True."
                )
        else:
            model_inputs.update({"input_ids": input_ids})

        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "cache_position": cache_position,
                "generate_idx": generate_idx,
            }
        )
        return model_inputs

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder, **kwargs):
        model_kwargs["generate_idx"] = outputs.generate_idx
        return model_kwargs

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        vision_feature_layer: Union[int, List[int]],
        vision_feature_select_strategy: str,
        **kwargs,
    ):
        if vision_feature_select_strategy not in ["default", "full"]:
            raise ValueError(f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}")

        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        image_outputs = self.vision_tower(pixel_values, output_hidden_states=True, **kwargs)

        if isinstance(vision_feature_layer, int):
            selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
            if vision_feature_select_strategy == "default":
                selected_image_feature = selected_image_feature[:, 1:]
        else:
            hs_pool = [image_outputs.hidden_states[layer_idx] for layer_idx in vision_feature_layer]
            if vision_feature_select_strategy == "default":
                hs_pool = [hs[:, 1:] for hs in hs_pool]
            selected_image_feature = torch.cat(hs_pool, dim=-1)

        if hasattr(self.rbln_config.vision_tower, "max_image_size"):
            num_real_patches = selected_image_feature.shape[1]
            max_patches = (
                (self.rbln_config.vision_tower.max_image_size[0] // self.config.vision_config.patch_size)
                * (self.rbln_config.vision_tower.max_image_size[1] // self.config.vision_config.patch_size)
                * pixel_values.shape[0]
            )
            num_padding_patches = max_patches - num_real_patches

            padding_tensor = torch.zeros(
                (selected_image_feature.shape[0], num_padding_patches, selected_image_feature.shape[2]),
                dtype=selected_image_feature.dtype,
            )
            padded_feature = torch.cat([selected_image_feature, padding_tensor], dim=1)
            padded_projected_feature = self.multi_modal_projector(padded_feature)
            image_features = padded_projected_feature[:, :num_real_patches, :]
        else:
            image_features = self.multi_modal_projector(selected_image_feature)

        return image_features

    def _preprocess_prefill(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[Union[int, List[int]]] = None,
        vision_feature_select_strategy: Optional[str] = None,
        return_dict: Optional[bool] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **lm_kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_features = self.get_image_features(
                pixel_values=pixel_values,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
                image_sizes=image_sizes,
            )

            special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1)
            special_image_mask = special_image_mask.expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        return inputs_embeds

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        generate_idx: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, LlavaCausalLMOutputWithPast]:
        # Prefill
        if cache_position is None:
            inputs_embeds = self._preprocess_prefill(
                input_ids=input_ids, inputs_embeds=inputs_embeds, pixel_values=pixel_values, image_sizes=image_sizes
            )
            logits = []
            inputs = inputs_embeds if inputs_embeds is not None else input_ids
            batch_size = inputs.shape[0]

            for b_idx in range(batch_size):
                cache_position = torch.arange(0, generate_idx[b_idx].item(), dtype=torch.int32).unsqueeze(0)
                output = self.language_model.prefill_decoder(
                    input_ids=inputs[b_idx : b_idx + 1] if inputs_embeds is None else None,
                    inputs_embeds=inputs[b_idx : b_idx + 1] if inputs_embeds is not None else None,
                    attention_mask=attention_mask[b_idx] if attention_mask is not None else None,
                    cache_position=cache_position,
                    batch_idx=b_idx,
                )
                logits.append(output.logits)

            logits = torch.cat(logits, dim=0)

        # Decoder
        else:
            logits = self.language_model.decoder(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
            ).logits

        if not return_dict:
            return logits, generate_idx
        else:
            return RBLNDecoderOnlyForCausalLMOutput(
                logits=logits,
                generate_idx=generate_idx,
            )
