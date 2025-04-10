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
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Union, List

import numpy as np
import torch
from transformers import (
    AutoModelForVision2Seq,
    Qwen2_5_VLForConditionalGeneration,
    PretrainedConfig,
    PreTrainedModel,
)

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VisionPatchEmbed,
    Qwen2_5_VisionRotaryEmbedding,
)
from transformers.modeling_utils import no_init_weights
from ....modeling import RBLNModel
from ....modeling_config import RBLNCompileConfig, RBLNConfig
from ....utils.logging import get_logger
from .qwen2_5_vl_architecture import Qwen2_5_VisionTransformerWrapper

logger = get_logger(__name__)

if TYPE_CHECKING:
    from transformers import (
        AutoFeatureExtractor,
        AutoProcessor,
        AutoTokenizer,
        PretrainedConfig,
    )


class RBLNQwen2_5_VisionTransformerPretrainedModel(RBLNModel):
    auto_model_class = None

    def __post_init__(self, **kwargs):
        self.max_seq_lens = torch.tensor(self.rbln_config.model_cfg["max_seq_lens"])
        self.transformer = self.model[0]
        config = self.config
        with no_init_weights():
            self.patch_embed = Qwen2_5_VisionPatchEmbed(
                patch_size=config.patch_size,
                temporal_patch_size=config.temporal_patch_size,
                in_channels=config.in_channels,
                embed_dim=config.hidden_size,
            )
        artifacts = torch.load(self.model_save_dir / self.subfolder / "torch_artifacts.pth", weights_only=False)
        self.patch_embed.load_state_dict(artifacts["patch_embed"])
        self.spatial_merge_size = config.spatial_merge_size
        self.spatial_merge_unit = config.spatial_merge_size * config.spatial_merge_size
        self.window_size = config.window_size
        self.patch_size = config.spatial_patch_size
        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)

    @classmethod
    def save_torch_artifacts(
        cls,
        model: "Qwen2_5_VLForConditionalGeneration",
        save_dir_path: Path,
        subfolder: str,
        rbln_config: RBLNConfig,
    ):
        """
        If you are unavoidably running on a CPU rather than an RBLN device,
        store the torch tensor, weight, etc. in this function.
        """
        save_dict = {}
        save_dict["patch_embed"] = model.patch_embed.state_dict()

        torch.save(save_dict, save_dir_path / subfolder / "torch_artifacts.pth")

    @classmethod
    def wrap_model_if_needed(cls, model: "PreTrainedModel", rbln_config: RBLNConfig):
        return Qwen2_5_VisionTransformerWrapper(model)

    def __getattr__(self, __name: str) -> Any:
        def redirect(func):
            return lambda *pargs, **kwargs: func(self, *pargs, **kwargs)

        val = getattr(Qwen2_5_VisionTransformerPretrainedModel, __name)

        if isinstance(val, Callable) and "self" in set(inspect.signature(val).parameters):
            return redirect(val)
        return val

    @classmethod
    def _get_rbln_config(
        cls,
        preprocessors: Optional[Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"]],
        model_config: Optional["PretrainedConfig"] = None,
        rbln_kwargs: Dict[str, Any] = {},
    ) -> RBLNConfig:
        rbln_max_seq_lens = rbln_kwargs.get("max_seq_lens", None)
        if rbln_max_seq_lens is None:
            rbln_max_seq_lens = [1024, 2048, 4096, 8192, 16384, 20480]
        elif isinstance(rbln_max_seq_lens, int):
            rbln_max_seq_lens = [rbln_max_seq_lens]

        window_size = model_config.window_size
        patch_size = model_config.patch_size
        win_seq_len = (window_size // patch_size) ** 2

        input_infos = []
        for max_seq_len in rbln_max_seq_lens:
            input_info = [
                ("hidden_states", [max_seq_len, model_config.hidden_size], "float32"),
                ("full_attn_masks", [1, 1, max_seq_len, max_seq_len], "float32"),
                (
                    "window_attn_masks",
                    [max_seq_len // win_seq_len, 1, win_seq_len, win_seq_len],
                    "float32",
                ),
                (
                    "cos",
                    [1, 1, max_seq_len, model_config.hidden_size // model_config.num_heads],
                    "float32",
                ),
                (
                    "sin",
                    [1, 1, max_seq_len, model_config.hidden_size // model_config.num_heads],
                    "float32",
                ),
            ]
            input_infos.append(input_info)

        rbln_compile_config = RBLNCompileConfig(input_info=input_infos)
        rbln_config = RBLNConfig(
            rbln_cls=cls.__name__,
            compile_cfgs=[rbln_compile_config],
            rbln_kwargs=rbln_kwargs,
        )
        rbln_config.model_cfg.update({"max_seq_lens": rbln_max_seq_lens})

        return rbln_config

    @staticmethod
    def _pad_for_window_attn_layers(
        window_indice: List[int],
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        window_seq_len: int,
        max_seq_len: int,
    ):
        # Padding for Window Attention
        padded_hidden_state = []
        padded_cos = []
        padded_sin = []
        window_valid_lengths = []
        for i in range(len(window_indice) - 1):
            start, end = window_indice[i], window_indice[i + 1]
            segment = hidden_states[start:end]
            cos_segment = position_embeddings[0][start:end]
            sin_segment = position_embeddings[1][start:end]
            segment_len = end - start

            if segment_len < window_seq_len:
                padding_size = window_seq_len - segment_len
                padding = torch.zeros(
                    padding_size,
                    segment.shape[-1],
                    dtype=segment.dtype,
                )
                padding_pos = torch.zeros(
                    padding_size,
                    cos_segment.shape[-1],
                    dtype=cos_segment.dtype,
                )
                padded_segment = torch.cat([segment, padding], dim=0)
                padded_cos_segment = torch.cat([cos_segment, padding_pos], dim=0)
                padded_sin_segment = torch.cat([sin_segment, padding_pos], dim=0)
            else:
                padded_segment = segment
                padded_cos_segment = cos_segment
                padded_sin_segment = sin_segment
            padded_hidden_state.append(padded_segment)
            window_valid_lengths.append(segment_len)
            padded_cos.append(padded_cos_segment)
            padded_sin.append(padded_sin_segment)
        hidden_state_padded = torch.cat(padded_hidden_state)
        cos_padded = torch.cat(padded_cos, dim=0)
        sin_padded = torch.cat(padded_sin, dim=0)

        window_attn_masks = torch.ones(
            max_seq_len // window_seq_len,
            1,
            window_seq_len,
            window_seq_len,
            dtype=torch.float32,
        )
        for i, valid_len in enumerate(window_valid_lengths):
            if valid_len < window_seq_len:
                window_attn_masks[i, :, valid_len:, :] = 0
                window_attn_masks[i, :, :, valid_len:] = 0

        return hidden_state_padded, cos_padded, sin_padded, window_attn_masks, window_valid_lengths

    @staticmethod
    def _pad_for_full_attn_layers(
        hidden_state_padded, cos_padded, sin_padded, max_seq_len, window_valid_lengths, window_seq_len
    ):
        if hidden_state_padded.shape[0] < max_seq_len:
            full_padding_size = max_seq_len - hidden_state_padded.shape[0]
            full_padding_hidden = torch.zeros(
                full_padding_size,
                hidden_state_padded.shape[-1],
                dtype=hidden_state_padded.dtype,
            )
            hidden_state_full_padded = torch.cat([hidden_state_padded, full_padding_hidden], dim=0)  # [5120, 1280]
            full_padding_pos = torch.zeros(
                full_padding_size,
                cos_padded.shape[-1],
                dtype=cos_padded.dtype,
            )
            cos_full_padded = torch.cat([cos_padded, full_padding_pos], dim=0)
            sin_full_padded = torch.cat([sin_padded, full_padding_pos], dim=0)
            window_valid_lengths.extend([0] * (max_seq_len // window_seq_len - len(window_valid_lengths)))
        else:
            hidden_state_full_padded = hidden_state_padded
            cos_full_padded = cos_padded
            sin_full_padded = sin_padded

        full_attn_masks = torch.ones(
            1,
            1,
            max_seq_len,
            max_seq_len,
            dtype=torch.float32,
        )
        for i, valid_len in enumerate(window_valid_lengths):
            start = i * window_seq_len
            end = start + window_seq_len
            full_attn_masks[:, :, start + valid_len : end, :] = 0
            full_attn_masks[:, :, :, start + valid_len : end] = 0

        return hidden_state_full_padded, cos_full_padded, sin_full_padded, full_attn_masks

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            dtype=torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            dtype=torch.int32,
        )
        cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)

        num_images = len(grid_thw)
        cu_window_seqlens = cu_window_seqlens.tolist()
        window_seq_len = (self.window_size // self.patch_size) ** 2

        output_hidden_states = []
        for i in range(num_images):
            image_s, image_e = cu_seqlens[i], cu_seqlens[i + 1]
            window_indice = cu_window_seqlens[cu_window_seqlens.index(image_s) : cu_window_seqlens.index(image_e) + 1]

            # Select close max_seq_len.
            window_padded_len = len(window_indice) * window_seq_len
            try:
                ws_index = torch.searchsorted(self.max_seq_lens, window_padded_len).item()
                max_seq_len = self.max_seq_lens[ws_index]
            except Exception as e:
                raise ValueError(
                    f"Required seq_len({window_padded_len}) is larger than max_sed_len({self.max_seq_lens.tolist()})."
                )

            # Padding for Window Attention Layers
            hidden_state_padded, cos_padded, sin_padded, window_attn_masks, window_valid_lengths = (
                self._pad_for_window_attn_layers(window_indice, hidden_states, position_embeddings, window_seq_len)
            )

            # Padding for Full Attention Layers
            hidden_state_full_padded, cos_full_padded, sin_full_padded, full_attn_masks = (
                self._pad_for_full_attn_layers(
                    hidden_state_padded, cos_padded, sin_padded, max_seq_len, window_valid_lengths, window_seq_len
                )
            )

            output = self.transformer(
                hidden_state_full_padded,
                full_attn_masks,
                window_attn_masks,
                cos_full_padded[None, None, :, :],
                sin_full_padded[None, None, :, :],
            )

            # Depad
            depadded_output = []
            for i, valid_len in enumerate(window_valid_lengths):
                start = i * (window_seq_len // self.spatial_merge_unit)  # 64 / 4 = 16
                end = start + (valid_len // self.spatial_merge_unit)
                depadded_output.append(output[start:end])
            output = torch.cat(depadded_output, dim=0)
            output_hidden_states.append(output)

        hidden_states = torch.cat(output_hidden_states)

        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :]

        return hidden_states


# class RBLNQwen2_5_VLForConditionalGeneration(RBLNModel):
#     auto_model_class = AutoModelForVision2Seq
#     _rbln_submodules = [
#         {"name": "visual"},
#     ]

#     def __post_init__(self, **kwargs):
#         self.visual = self.rbln_submodules[0]

#         artifacts = torch.load(self.model_save_dir / self.subfolder / "torch_artifacts.pth", weights_only=False)
#         self.embed_tokens = artifacts["embed_tokens"]

#         return super().__post_init__(**kwargs)

#     def __getattr__(self, __name: str) -> Any:
#         def redirect(func):
#             return lambda *pargs, **kwargs: func(self, *pargs, **kwargs)

#         val = getattr(Qwen2_5_VLForConditionalGeneration, __name)

#         if isinstance(val, Callable) and "self" in set(inspect.signature(val).parameters):
#             return redirect(val)
#         return val

#     def can_generate(self):
#         return True

#     @classmethod
#     def save_torch_artifacts(
#         cls,
#         model: "Qwen2_5_VLForConditionalGeneration",
#         save_dir_path: Path,
#         subfolder: str,
#         rbln_config: RBLNConfig,
#     ):
#         """
#         If you are unavoidably running on a CPU rather than an RBLN device,
#         store the torch tensor, weight, etc. in this function.
#         """
#         save_dict = {}
#         save_dict["embed_tokens"] = model.embed_tokens
#         torch.save(save_dict, save_dir_path / subfolder / "torch_artifacts.pth")

#     @classmethod
#     def wrap_model_if_needed(cls, model: "PreTrainedModel", rbln_config: RBLNConfig):
#         return model.multi_modal_projector

#     @classmethod
#     def _get_rbln_config(
#         cls,
#         preprocessors: Optional[Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"]],
#         model_config: Optional["PretrainedConfig"] = None,
#         rbln_kwargs={},
#     ) -> RBLNConfig:
#         vision_feature_select_strategy = rbln_kwargs.get("vision_feature_select_strategy", None)

#         # 1. Multi-modal projection layer
#         batch_size = rbln_kwargs.get("rbln_batch_size", None)
#         if batch_size is None:
#             batch_size = 1

#         feature_size = model_config.vision_config.hidden_size

#         # See forward function to see more details.
#         vision_feature_select_strategy = (
#             vision_feature_select_strategy
#             if vision_feature_select_strategy is not None
#             else model_config.vision_feature_select_strategy
#         )

#         # Calculating `num_positions` : See CLIPVisionEmbeddings of transformers for more details.
#         num_positions = (model_config.vision_config.image_size // model_config.vision_config.patch_size) ** 2 + 1
#         if vision_feature_select_strategy == "default":
#             selected_image_feature_dim = num_positions - 1
#         else:
#             selected_image_feature_dim = num_positions

#         input_info = [("image_features", [batch_size, selected_image_feature_dim, feature_size], "float32")]
#         rbln_compile_config = RBLNCompileConfig(input_info=input_info)
#         rbln_config = RBLNConfig(rbln_cls=cls.__name__, compile_cfgs=[rbln_compile_config], rbln_kwargs=rbln_kwargs)
#         return rbln_config

#     def prepare_inputs_for_generation(
#         self,
#         input_ids,
#         inputs_embeds=None,
#         pixel_values=None,
#         image_sizes=None,
#         attention_mask=None,
#         generate_idx=None,
#         **kwargs,
#     ):
#         # Prepare HF generation
#         is_prefill_phase = generate_idx is None
#         batch_size = input_ids.shape[0]

#         model_inputs = self.language_model.prepare_inputs_for_generation(
#             input_ids=input_ids,
#             inputs_embeds=inputs_embeds,
#             generate_idx=generate_idx,  # Not affect
#             attention_mask=attention_mask,
#             **kwargs,
#         )

#         if is_prefill_phase:
#             model_inputs["generate_idx"] = torch.zeros((batch_size, 1), dtype=torch.int32)
#             model_inputs.update(
#                 {
#                     "pixel_values": pixel_values,
#                     "image_sizes": image_sizes,
#                 }
#             )

#         model_inputs["attention_mask"] = attention_mask
#         return model_inputs

#     def _update_model_kwargs_for_generation(
#         self,
#         outputs: RBLNDecoderOnlyOutput,
#         model_kwargs: Dict[str, Any],
#         **kwargs,
#     ) -> Dict[str, Any]:
#         # update generate_idx
#         model_kwargs["generate_idx"] = outputs.generate_idx

#         return model_kwargs

#     def forward(
#         self,
#         input_ids: torch.LongTensor = None,
#         attention_mask: torch.LongTensor = None,
#         pixel_values: torch.FloatTensor = None,
#         image_sizes: Optional[torch.LongTensor] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         vision_feature_layer: Optional[int] = None,
#         vision_feature_select_strategy: Optional[str] = None,
#         cache_position: torch.Tensor = None,
#         generate_idx: Optional[torch.Tensor] = None,
#         batch_idx: Optional[int] = None,
#         **kwargs,
#     ) -> Union[Tuple, RBLNDecoderOnlyOutput]:
#         vision_feature_layer = (
#             vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
#         )
#         vision_feature_select_strategy = (
#             vision_feature_select_strategy
#             if vision_feature_select_strategy is not None
#             else self.config.vision_feature_select_strategy
#         )

#         if inputs_embeds is not None:
#             raise NotImplementedError("Specifying inputs_embeds is not supported.")
#         inputs_embeds = self.get_input_embeddings()(input_ids)

#         if pixel_values is not None and pixel_values.size(0) > 0:
#             image_features, _ = self.image_embedding(
#                 pixel_values=pixel_values,
#                 image_sizes=image_sizes,
#                 vision_feature_layer=vision_feature_layer,
#                 vision_feature_select_strategy=vision_feature_select_strategy,
#             )

#             n_image_tokens = (input_ids == self.config.image_token_index).sum().item()
#             n_image_features = image_features.shape[0]
#             if n_image_tokens != n_image_features:
#                 raise ValueError(
#                     f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
#                 )
#             special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1)
#             special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
#             image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
#             inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

#         is_prefill_phase = not generate_idx.bool().all()

#         if is_prefill_phase:
#             logits = []
#             batch_size = input_ids.shape[0]
#             inputs_embeds = [inputs_embeds[i : i + 1, attention_mask[i].bool()] for i in range(batch_size)]
#             for batch_idx in range(batch_size):
#                 generate_idx[batch_idx] = inputs_embeds[batch_idx].shape[-2]
#                 logit = self.language_model.prefill_decoder(
#                     inputs_embeds=inputs_embeds[batch_idx],
#                     batch_idx=batch_idx,
#                     cache_position=torch.arange(
#                         0,
#                         generate_idx[batch_idx].item(),
#                         dtype=torch.int32,
#                     ).unsqueeze(0),
#                 )

#                 logits.append(logit)
#             logits = torch.cat(logits, dim=0)
#         else:
#             logits = self.language_model.decoder(
#                 inputs_embeds=inputs_embeds,
#                 cache_position=cache_position,
#             )

#         return RBLNDecoderOnlyOutput(logits=logits, generate_idx=generate_idx)
