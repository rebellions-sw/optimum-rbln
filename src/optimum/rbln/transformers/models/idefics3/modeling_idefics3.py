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
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple, Union

import rebel
import torch
from transformers import (
    AutoModelForVision2Seq,
    Idefics3ForConditionalGeneration,
    Idefics3VisionConfig,
    Idefics3VisionTransformer,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import no_init_weights
from transformers.models.idefics3.modeling_idefics3 import Idefics3CausalLMOutputWithPast, Idefics3VisionEmbeddings

from ....configuration_utils import RBLNCompileConfig, RBLNModelConfig
from ....modeling import RBLNModel
from ....utils.runtime_utils import RBLNPytorchRuntime
from ..decoderonly.modeling_decoderonly import (
    RBLNDecoderOnlyOutput,
)


if TYPE_CHECKING:
    from transformers import (
        AutoFeatureExtractor,
        AutoProcessor,
        AutoTokenizer,
    )


class RBLNRuntimeVisionModel(RBLNPytorchRuntime):
    mandatory_members = ["main_input_name"]

    def __init__(
        self,
        runtime: rebel.Runtime,
        config: Idefics3VisionConfig,
        **kwargs: Any,
    ) -> None:
        super().__init__(runtime, **kwargs)
        self.patch_size = config.patch_size
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

    def forward(
        self,
        pixel_values,
        patch_attention_mask: Optional[torch.BoolTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        batch_size = pixel_values.size(0)
        if patch_attention_mask is None:
            patch_size = self.patch_size
            patch_attention_mask = torch.ones(
                (
                    batch_size,
                    pixel_values.size(2) // patch_size,
                    pixel_values.size(3) // patch_size,
                )
            )
            patch_attention_mask = patch_attention_mask.to(dtype=torch.bool, device=pixel_values.device)

        hidden_states = self.embeddings(pixel_values=pixel_values, patch_attention_mask=patch_attention_mask)

        return super().forward(hidden_states.contiguous())


class RBLNIdefics3VisionTransformer(RBLNModel):
    def __post_init__(self, **kwargs):
        artifacts = torch.load(self.model_save_dir / self.subfolder / "torch_artifacts.pth", weights_only=False)
        with no_init_weights():
            self.embeddings = Idefics3VisionEmbeddings(self.config)
        self.embeddings.load_state_dict(artifacts["embeddings"])
        self.model = RBLNRuntimeVisionModel(
            self.model[0], main_input_name="pixel_values", config=self.config, embeddings=self.embeddings
        )

    @classmethod
    def save_torch_artifacts(
        cls,
        model: "PreTrainedModel",
        save_dir_path: Path,
        subfolder: str,
        rbln_config: RBLNModelConfig,
    ):
        """
        If you are unavoidably running on a CPU rather than an RBLN device,
        store the torch tensor, weight, etc. in this function.
        """
        save_dict = {}
        save_dict["embeddings"] = model.get_input_embeddings().state_dict()
        torch.save(save_dict, save_dir_path / subfolder / "torch_artifacts.pth")

    def get_input_embeddings(self):
        return self.embeddings

    @classmethod
    def wrap_model_if_needed(cls, model: torch.nn.Module, rbln_config: RBLNModelConfig) -> torch.nn.Module:
        class Idefics3VisionTransformerWrapper(torch.nn.Module):
            def __init__(self, model: "Idefics3VisionTransformer"):
                super().__init__()
                self.encoder = model.encoder
                self.post_layernorm = model.post_layernorm

            def forward(self, hidden_states, patch_attention_mask: Optional[torch.BoolTensor] = None):
                encoder_outputs = self.encoder(
                    inputs_embeds=hidden_states,
                    attention_mask=patch_attention_mask,
                    output_attentions=None,
                    output_hidden_states=None,
                    return_dict=False,
                )
                last_hidden_state = encoder_outputs[0]
                last_hidden_state = self.post_layernorm(last_hidden_state)
                return last_hidden_state

        return Idefics3VisionTransformerWrapper(model).eval()

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Optional[Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"]],
        model: Optional["PreTrainedModel"] = None,
        model_config: Optional["PretrainedConfig"] = None,
        rbln_config: Optional[RBLNModelConfig] = None,
    ) -> RBLNModelConfig:
        input_info = [
            (
                "hidden_states",
                [
                    # batch_size * num_patches (dependent on image size) -> compile with 1 and use for loop
                    1,
                    (model_config.image_size // model_config.patch_size) ** 2,
                    model_config.hidden_size,
                ],
                "float32",
            ),
        ]

        rbln_compile_config = RBLNCompileConfig(input_info=input_info)
        rbln_config.set_compile_cfgs([rbln_compile_config])
        return rbln_config

    def forward(
        self,
        pixel_values,
        patch_attention_mask: Optional[torch.BoolTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutput]:
        batch_size = pixel_values.shape[0]
        last_hidden_state = []
        for i in range(batch_size):
            if patch_attention_mask is not None:
                batch_attention_mask = patch_attention_mask[i : i + 1,]
            else:
                batch_attention_mask = None

            batch_hidden_state = self.model(
                pixel_values[i : i + 1,],
                batch_attention_mask,
                return_dict=False,
            )
            last_hidden_state.append(batch_hidden_state)
        last_hidden_state = torch.cat(last_hidden_state, dim=0)

        if not return_dict:
            return (last_hidden_state,)
        else:
            return BaseModelOutput(last_hidden_state=last_hidden_state)


class RBLNIdefics3ForConditionalGeneration(RBLNModel):
    auto_model_class = AutoModelForVision2Seq
    _rbln_submodules = [{"name": "vision_model"}, {"name": "text_model"}]
    _rbln_submodule_prefix = "model"

    def __getattr__(self, __name: str) -> Any:
        def redirect(func):
            return lambda *pargs, **kwargs: func(self, *pargs, **kwargs)

        val = getattr(Idefics3ForConditionalGeneration, __name)

        if isinstance(val, Callable) and "self" in set(inspect.signature(val).parameters):
            return redirect(val)
        return val

    def can_generate(self):
        return True

    @classmethod
    def get_pytorch_model(cls, *args, **kwargs):
        model = super().get_pytorch_model(*args, **kwargs)

        with no_init_weights():
            model_cls_name = model.model.text_model.__class__.__name__
            causal_model_cls_name = model_cls_name.replace("Model", "ForCausalLM")
            causal_model_cls = getattr(importlib.import_module("transformers"), causal_model_cls_name)
            new_text_model = causal_model_cls(model.model.text_model.config)

        new_text_model.lm_head = model.lm_head
        new_text_model.model = model.model.text_model
        model.model.text_model = new_text_model
        model.lm_head = None
        del model.lm_head
        return model

    def __post_init__(self, **kwargs):
        self.vision_model = self.rbln_submodules[0]
        self.connector = self.model[0]
        self.text_model = self.rbln_submodules[1]

    def get_attn_impl(self) -> str:
        return self.rbln_config.text_model.attn_impl

    def get_kvcache_num_blocks(self) -> int:
        return self.rbln_config.text_model.kvcache_num_blocks

    def get_input_embeddings(self):
        return self.text_model.get_input_embeddings()

    @classmethod
    def wrap_model_if_needed(cls, model, rbln_config):
        return model.model.connector

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Optional[Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"]],
        model: Optional["PreTrainedModel"] = None,
        model_config: Optional["PretrainedConfig"] = None,
        rbln_config: Optional[RBLNModelConfig] = None,
    ) -> RBLNModelConfig:
        input_info = [
            (
                "image_hidden_states",
                [
                    # batch_size * num_patches (dependent on image size) -> compile with 1 and use for loop
                    1,
                    (model_config.vision_config.image_size // model_config.vision_config.patch_size) ** 2,
                    model_config.vision_config.hidden_size,
                ],
                "float32",
            ),
        ]

        rbln_compile_config = RBLNCompileConfig(input_info=input_info)
        rbln_config.set_compile_cfgs([rbln_compile_config])

        return rbln_config

    def prepare_inputs_for_generation(
        self,
        input_ids,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        pixel_values=None,
        pixel_attention_mask=None,
        image_hidden_states=None,
        generate_idx=None,
        **kwargs,
    ):
        is_prefill_phase = generate_idx is None
        model_inputs = {}

        if is_prefill_phase:
            generate_idx = attention_mask.sum(dim=-1, keepdim=True).int()
            cache_position = None
            pixel_values = pixel_values
            pixel_attention_mask = pixel_attention_mask
        else:
            if inputs_embeds is not None:
                raise NotImplementedError("Specifying inputs_embeds in decoder phase is not supported.")

            pixel_values = None
            pixel_attention_mask = None
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
                "pixel_attention_mask": pixel_attention_mask,
                "image_hidden_states": image_hidden_states,
                "cache_position": cache_position,
                "generate_idx": generate_idx,
            }
        )
        return model_inputs

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder, **kwargs):
        model_kwargs["generate_idx"] = outputs.generate_idx
        return model_kwargs

    def inputs_merger(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: Optional[torch.Tensor],
        image_hidden_states: Optional[torch.Tensor],
    ):
        num_images, _, vision_hidden_size = image_hidden_states.shape
        special_image_token_mask = input_ids == self.config.image_token_id
        new_inputs_embeds = inputs_embeds.clone()
        reshaped_image_hidden_states = image_hidden_states.view(-1, vision_hidden_size)
        reshaped_image_hidden_states = reshaped_image_hidden_states.to(inputs_embeds.device, inputs_embeds.dtype)
        new_inputs_embeds[special_image_token_mask] = reshaped_image_hidden_states
        return new_inputs_embeds

    def _preprocess_prefill(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.BoolTensor] = None,
        image_hidden_states: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        if input_ids is not None:
            batch_size, _ = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, _, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is not None and input_ids is None:
            raise ValueError("When first calling the model, if input_embeds are passed, input_ids should not be None.")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids).to(self.device)

        if pixel_values is not None and image_hidden_states is not None:
            raise ValueError("You cannot specify both pixel_values and image_hidden_states at the same time")

        elif pixel_values is not None:
            batch_size, num_images, num_channels, height, width = pixel_values.shape
            pixel_values = pixel_values.to(dtype=self.dtype)
            pixel_values = pixel_values.view(batch_size * num_images, *pixel_values.shape[2:])

            nb_values_per_image = pixel_values.shape[1:].numel()
            real_images_inds = (pixel_values == 0.0).sum(dim=(-1, -2, -3)) != nb_values_per_image
            pixel_values = pixel_values[real_images_inds].contiguous()

            if pixel_attention_mask is None:
                pixel_attention_mask = torch.ones(
                    size=(pixel_values.size(0), pixel_values.size(2), pixel_values.size(3)),
                    dtype=torch.bool,
                    device=pixel_values.device,
                )
            else:
                pixel_attention_mask = pixel_attention_mask.view(
                    batch_size * num_images, *pixel_attention_mask.shape[2:]
                )
                pixel_attention_mask = pixel_attention_mask[real_images_inds].contiguous()

            patch_size = self.config.vision_config.patch_size
            patches_subgrid = pixel_attention_mask.unfold(dimension=1, size=patch_size, step=patch_size)
            patches_subgrid = patches_subgrid.unfold(dimension=2, size=patch_size, step=patch_size)
            patch_attention_mask = (patches_subgrid.sum(dim=(-1, -2)) > 0).bool()

            image_hidden_states = self.vision_model(
                pixel_values=pixel_values, patch_attention_mask=patch_attention_mask, return_dict=True
            ).last_hidden_state

            connector_outputs = []
            for i in range(image_hidden_states.shape[0]):
                connector_outputs.append(self.connector(image_hidden_states[i : i + 1,]))
            image_hidden_states = torch.cat(connector_outputs, dim=0)

        elif image_hidden_states is not None:
            image_hidden_states = image_hidden_states.to(dtype=self.dtype, device=input_ids.device)

        if inputs_embeds is not None and image_hidden_states is not None:
            inputs_embeds = self.inputs_merger(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_hidden_states=image_hidden_states,
            )

        return inputs_embeds

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.BoolTensor] = None,
        image_hidden_states: Optional[torch.FloatTensor] = None,
        cache_position: torch.Tensor = None,
        generate_idx: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, Idefics3CausalLMOutputWithPast]:
        # Prefill
        if cache_position is None:
            inputs_embeds = self._preprocess_prefill(
                input_ids, inputs_embeds, pixel_values, pixel_attention_mask, image_hidden_states
            )
            logits = []
            inputs = inputs_embeds if inputs_embeds is not None else input_ids
            batch_size = inputs.shape[0]

            for b_idx in range(batch_size):
                cache_position = torch.arange(0, generate_idx[b_idx].item(), dtype=torch.int32).unsqueeze(0)
                logit = self.text_model.prefill_decoder(
                    input_ids=inputs[b_idx : b_idx + 1] if inputs_embeds is None else None,
                    inputs_embeds=inputs[b_idx : b_idx + 1] if inputs_embeds is not None else None,
                    attention_mask=attention_mask[b_idx] if attention_mask is not None else None,
                    cache_position=cache_position,
                    batch_idx=b_idx,
                )
                logits.append(logit)

            logits = torch.cat(logits, dim=0)

        # Decoder
        else:
            logits = self.text_model.decoder(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
            )

        return RBLNDecoderOnlyOutput(
            logits=logits,
            generate_idx=generate_idx,
        )
