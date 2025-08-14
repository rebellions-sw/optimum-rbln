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

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple, Union

import torch

from transformers.models.grounding_dino.modeling_grounding_dino import GroundingDinoEncoderOutput

from ....configuration_utils import RBLNCompileConfig, RBLNModelConfig
from ....modeling import RBLNModel
from ....utils.logging import get_logger
from .configuration_grounding_dino import RBLNGroundingDinoForObjectDetectionConfig, RBLNGroundingDinoEncoderConfig
# from transformers.models.grounding_dino.modeling_grounding_dino import generate_masks_with_special_tokens_and_transfer_map

logger = get_logger(__name__)

if TYPE_CHECKING:
    from transformers import (
        AutoFeatureExtractor,
        AutoProcessor,
        AutoTokenizer,
        GroundingDinoModel,
        PreTrainedModel,
        GroundingDinoForObjectDetection,
    )

    from ....diffusers.modeling_diffusers import RBLNDiffusionMixin, RBLNDiffusionMixinConfig


class _GroundingDinoModel(torch.nn.Module):
    def __init__(self, model: "GroundingDinoModel"):
        super().__init__()

        self.backbone = model.backbone
        self.config = model.config
        self.text_backbone = model.text_backbone
        self.text_projection = model.text_projection
        self.query_position_embeddings = model.query_position_embeddings
        self.input_proj_vision = model.input_proj_vision
        self.encoder = model.encoder
        self.level_embed = model.level_embed
        self.get_valid_ratio = model.get_valid_ratio

    def forward(
        self,
        pixel_values: torch.tensor,
        input_ids: torch.tensor,
        token_type_ids: Optional[torch.tensor] = None,
        # attention_mask: Optional[torch.tensor] = None,
        pixel_mask: Optional[torch.tensor] = None,
        text_self_attention_masks: Optional[torch.tensor] = None,
        position_ids: Optional[torch.tensor] = None,
        encoder_outputs=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=False,
    ):
        # if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        text_token_mask = attention_mask.bool()  # just to avoid renaming everywhere

        max_text_len = self.config.max_text_len
        if text_self_attention_masks.shape[1] > max_text_len:
            text_self_attention_masks = text_self_attention_masks[:, :max_text_len, :max_text_len]
            position_ids = position_ids[:, :max_text_len]
            input_ids = input_ids[:, :max_text_len]
            token_type_ids = token_type_ids[:, :max_text_len]
            text_token_mask = text_token_mask[:, :max_text_len]

        # Extract text features from text backbone
        text_outputs = self.text_backbone(
            input_ids, text_self_attention_masks, token_type_ids, position_ids, return_dict=return_dict
        )
        text_features = text_outputs.last_hidden_state if return_dict else text_outputs[0]
        text_features = self.text_projection(text_features)

        batch_size, num_channels, height, width = pixel_values.shape
        device = pixel_values.device

        if pixel_mask is None:
            pixel_mask = torch.ones(((batch_size, height, width)), dtype=torch.long, device=device)

        # Extract multi-scale feature maps of same resolution `config.d_model` (cf Figure 4 in paper)
        # First, sent pixel_values + pixel_mask through Backbone to obtain the features
        # which is a list of tuples
        vision_features, position_embeddings_list = self.backbone(pixel_values, pixel_mask)

        # Then, apply 1x1 convolution to reduce the channel dimension to d_model (256 by default)
        feature_maps = []
        masks = []
        for level, (source, mask) in enumerate(vision_features):
            feature_maps.append(self.input_proj_vision[level](source))
            masks.append(mask)

        # Lowest resolution feature maps are obtained via 3x3 stride 2 convolutions on the final stage
        if self.config.num_feature_levels > len(feature_maps):
            _len_sources = len(feature_maps)
            for level in range(_len_sources, self.config.num_feature_levels):
                if level == _len_sources:
                    source = self.input_proj_vision[level](vision_features[-1][0])
                else:
                    source = self.input_proj_vision[level](feature_maps[-1])
                mask = torch.nn.functional.interpolate(pixel_mask[None].float(), size=source.shape[-2:]).to(
                    torch.bool
                )[0]
                pos_l = self.backbone.position_embedding(source, mask).to(source.dtype)
                feature_maps.append(source)
                masks.append(mask)
                position_embeddings_list.append(pos_l)

        return text_features, vision_features[0], feature_maps[0]
        query_embeds = None
        if self.config.embedding_init_target or self.config.two_stage:
            query_embeds = self.query_position_embeddings.weight

        source_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes_list = []
        for level, (source, mask, pos_embed) in enumerate(zip(feature_maps, masks, position_embeddings_list)):
            batch_size, num_channels, height, width = source.shape
            spatial_shape = (height, width)
            spatial_shapes_list.append(spatial_shape)
            source = source.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[level].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            source_flatten.append(source)
            mask_flatten.append(mask)
        source_flatten = torch.cat(source_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes_list, dtype=torch.long, device=source_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        valid_ratios = valid_ratios.float()

        # Fourth, sent source_flatten + mask_flatten + lvl_pos_embed_flatten (backbone + proj layer output) through encoder
        # Also provide spatial_shapes, level_start_index and valid_ratios
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                vision_features=source_flatten,
                vision_attention_mask=~mask_flatten,
                vision_position_embedding=lvl_pos_embed_flatten,
                spatial_shapes=spatial_shapes,
                spatial_shapes_list=spatial_shapes_list,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                text_features=text_features,
                text_attention_mask=~text_token_mask,
                text_position_embedding=None,
                text_self_attention_masks=~text_self_attention_masks,
                text_position_ids=position_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        return encoder_outputs


class RBLNGroundingDinoForObjectDetection(RBLNModel):
    _rbln_submodules = [
        {"name": "encoder"},
        # {"name": "text_backbone"},
        # {"name": "backbone"},
        # {"name": "decoder"},
    ]
    """
    RBLN optimized CLIP text encoder model.

    This class provides hardware-accelerated inference for CLIP text encoders
    on RBLN devices, supporting text encoding for multimodal tasks.
    """

    @classmethod
    def get_pytorch_model(cls, *args, **kwargs):
        model = super().get_pytorch_model(*args, **kwargs)
        model.encoder = model.model.encoder
        model.encoder.config = model.config
        return model

    @classmethod
    def wrap_model_if_needed(
        cls, model: torch.nn.Module, rbln_config: RBLNGroundingDinoForObjectDetectionConfig
    ) -> torch.nn.Module:
        return _GroundingDinoModel(model.model).eval()

    @classmethod
    def update_rbln_config_using_pipe(
        cls, pipe: "RBLNDiffusionMixin", rbln_config: "RBLNDiffusionMixinConfig", submodule_name: str
    ) -> "RBLNDiffusionMixinConfig":
        return rbln_config

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"],
        model: Optional["PreTrainedModel"] = None,
        model_config: "GroundingDinoConfig" = None,
        rbln_config: Optional[RBLNGroundingDinoForObjectDetectionConfig] = None,
    ) -> RBLNGroundingDinoForObjectDetectionConfig:
        if rbln_config.image_size is None:
            longest_edge = None
            for processor in preprocessors:
                if hasattr(processor, "image_processor"):
                    longest_edge = processor.image_processor.size["longest_edge"]
                    break
            padded_longest_edge = longest_edge + ((64 - longest_edge) % 64)
            rbln_config.image_size = padded_longest_edge

        input_info = [
            (
                "pixel_values",
                [rbln_config.batch_size, 3, 1344, 1344], #800 < h,w<1333
                "float32",
            ),
            (
                "input_ids",
                [
                    rbln_config.batch_size,
                    model_config.max_text_len,
                ],
                "int64",
            ),
            (
                "token_type_ids",
                [
                    rbln_config.batch_size,
                    model_config.max_text_len,
                ],
                "int64",
            ),
            (
                "pixel_mask",
                [rbln_config.batch_size, 1344, 1344],
                "int16",
            ),
            (
                "text_self_attention_masks",
                [
                    rbln_config.batch_size,
                    model_config.max_text_len,
                    model_config.max_text_len,
                ],
                "int16",
            ),
            (
                "position_ids",
                [
                    rbln_config.batch_size,
                    model_config.max_text_len,
                ],
                "int64",
            ),
        ]

        rbln_config.set_compile_cfgs([RBLNCompileConfig(input_info=input_info)])
        return rbln_config

    def forward(self, *args, return_dict: bool = None, **kwargs) -> torch.FloatTensor:
        # To ignore using attention_mask, we override forward method.
        output = super().forward(*args, **kwargs, return_dict=return_dict)
        import pdb

        pdb.set_trace()
        return output

    # def _prepare_output(self, output, return_dict):
    #     # Prepare model output based on return_dict flag.
    #     # This method can be overridden by subclasses to provide task-specific output handling.

    #     if not return_dict:
    #         return (output,) if not isinstance(output, (tuple, list)) else output
    #     else:
    #         return CLIPTextModelOutput(
    #             text_embeds=output[0],
    #             last_hidden_state=output[1],
    #             hidden_states=output[2:],
    #         )


from typing import List, Tuple


class GroundingDinoEncoder(torch.nn.Module):
    def __init__(self, model: "GroundingDinoEncoder"):
        super().__init__()
        self.layers = model.layers
        self.config = model.config
        # FIXME: define Spatial shapes from config
        # self.spatial_shapes = torch.tensor([
        #     [100, 151],
        #     [ 50,  76],
        #     [ 25,  38],
        #     [ 13,  19]])
        # self.spatial_shapes_list = [
        #     (100, 151),
        #     (50, 76),
        #     (25, 38),
        #     (13, 19)
        # ]
        self.spatial_shapes = torch.tensor([[168, 168], [84, 84], [42, 42], [21, 21]])
        self.spatial_shapes_list = [(168, 168), (84, 84), (42, 42), (21, 21)]

    def forward(
        self,
        vision_features: torch.Tensor,
        vision_attention_mask: torch.Tensor,
        vision_position_embedding: torch.Tensor,
        # spatial_shapes: torch.Tensor,
        # spatial_shapes_list: List[Tuple[int, int]],
        # level_start_index: torch.Tensor,
        # valid_ratios=None,
        text_features: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        text_self_attention_masks: Optional[torch.Tensor] = None,
        text_position_ids: Optional[torch.Tensor] = None,
        reference_points: Optional[torch.Tensor] = None,
        text_position_embedding: Optional[torch.Tensor] = None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            vision_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Flattened feature map (output of the backbone + projection layer) that is passed to the encoder.
            vision_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding pixel features. Mask values selected in `[0, 1]`:
                - 0 for pixel features that are real (i.e. **not masked**),
                - 1 for pixel features that are padding (i.e. **masked**).
                [What are attention masks?](../glossary#attention-mask)
            vision_position_embedding (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Position embeddings that are added to the queries and keys in each self-attention layer.
            spatial_shapes (`torch.LongTensor` of shape `(num_feature_levels, 2)`):
                Spatial shapes of each feature map.
            spatial_shapes_list (`List[Tuple[int, int]]`):
                Spatial shapes of each feature map (but as list for export compatibility).
            level_start_index (`torch.LongTensor` of shape `(num_feature_levels)`):
                Starting index of each feature map.
            valid_ratios (`torch.FloatTensor` of shape `(batch_size, num_feature_levels, 2)`):
                Ratio of valid area in each feature level.
            text_features (`torch.FloatTensor` of shape `(batch_size, text_seq_len, hidden_size)`):
                Flattened text features that are passed to the encoder.
            text_attention_mask (`torch.Tensor` of shape `(batch_size, text_seq_len)`, *optional*):
                Mask to avoid performing attention on padding text features. Mask values selected in `[0, 1]`:
                - 0 for text features that are real (i.e. **not masked**),
                - 1 for text features that are padding (i.e. **masked**).
                [What are attention masks?](../glossary#attention-mask)
            text_position_embedding (`torch.FloatTensor` of shape `(batch_size, text_seq_len)`):
                Position embeddings that are added to the queries and keys in each self-attention layer.
            text_self_attention_masks (`torch.BoolTensor` of shape `(batch_size, text_seq_len, text_seq_len)`):
                Masks to avoid performing attention between padding text features. Mask values selected in `[0, 1]`:
                - 1 for text features that are real (i.e. **not masked**),
                - 0 for text features that are padding (i.e. **masked**).
            text_position_ids (`torch.LongTensor` of shape `(batch_size, num_queries)`):
                Position ids for text features.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=vision_features.device)

        encoder_vision_states = () if output_hidden_states else None
        encoder_text_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None
        all_attn_fused_text = () if output_attentions else None
        all_attn_fused_vision = () if output_attentions else None
        all_attn_enhanced_text = () if output_attentions else None
        all_attn_deformable = () if output_attentions else None
        for i, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_vision_states += (vision_features,)
                encoder_text_states += (text_features,)

            (vision_features, text_features), attentions = encoder_layer(
                vision_features=vision_features,
                vision_position_embedding=vision_position_embedding,
                spatial_shapes=self.spatial_shapes,
                spatial_shapes_list=self.spatial_shapes_list,
                level_start_index=None,
                key_padding_mask=vision_attention_mask,
                reference_points=reference_points,
                text_features=text_features,
                text_attention_mask=text_attention_mask,
                text_position_embedding=text_position_embedding,
                text_self_attention_masks=text_self_attention_masks,
                text_position_ids=text_position_ids,
            )
            if output_attentions:
                all_attn_fused_vision += (attentions[0],)
                all_attn_fused_text += (attentions[1],)
                all_attn_enhanced_text += (attentions[2],)
                all_attn_deformable += (attentions[3],)

        if output_hidden_states:
            encoder_vision_states += (vision_features,)
            encoder_text_states += (text_features,)

        if output_attentions:
            all_attns = (all_attn_fused_vision, all_attn_fused_text, all_attn_enhanced_text, all_attn_deformable)

        if not return_dict:
            enc_outputs = [vision_features, text_features, encoder_vision_states, encoder_text_states, all_attns]
            return tuple(v for v in enc_outputs if v is not None)
        return GroundingDinoEncoderOutput(
            last_hidden_state_vision=vision_features,
            last_hidden_state_text=text_features,
            vision_hidden_states=encoder_vision_states,
            text_hidden_states=encoder_text_states,
            attentions=all_attns,
        )


class RBLNGroundingDinoEncoder(RBLNModel):
    """
    RBLN optimized CLIP text encoder model.

    This class provides hardware-accelerated inference for CLIP text encoders
    on RBLN devices, supporting text encoding for multimodal tasks.
    """

    @classmethod
    def wrap_model_if_needed(
        cls, model: torch.nn.Module, rbln_config: RBLNGroundingDinoForObjectDetectionConfig
    ) -> torch.nn.Module:
        return GroundingDinoEncoder(model).eval()

    @classmethod
    def update_rbln_config_using_pipe(
        cls, pipe: "RBLNDiffusionMixin", rbln_config: "RBLNDiffusionMixinConfig", submodule_name: str
    ) -> "RBLNDiffusionMixinConfig":
        return rbln_config

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"],
        model: Optional["PreTrainedModel"] = None,
        model_config: "GroundingDinoEncoderConfig" = None,
        rbln_config: Optional[RBLNGroundingDinoEncoderConfig] = None,
    ) -> RBLNGroundingDinoEncoderConfig:
        input_info = [
            (
                "vision_features",
                [rbln_config.batch_size, 37485, model_config.d_model],
                "float32",
            ),
            (
                "vision_attention_mask",
                [
                    rbln_config.batch_size,
                    37485,
                ],
                "float32",
            ),
            (
                "vision_position_embedding",
                [rbln_config.batch_size, 37485, model_config.d_model],
                "float32",
            ),
            (
                "text_features",
                [rbln_config.batch_size, model_config.max_text_len, model_config.d_model],
                "float32",
            ),
            (
                "text_attention_mask",
                [
                    rbln_config.batch_size,
                    model_config.max_text_len,
                ],
                "float32",
            ),
            (
                "text_self_attention_masks",
                [
                    rbln_config.batch_size,
                    model_config.max_text_len,
                    model_config.max_text_len,
                ],
                "float32",
            ),
            (
                "text_position_ids",
                [
                    rbln_config.batch_size,
                    model_config.max_text_len,
                ],
                "int32",
            ),
            (
                "reference_points",
                [rbln_config.batch_size, 37485, 4, 2],
                "float32",
            ),
        ]

        rbln_config.set_compile_cfgs([RBLNCompileConfig(input_info=input_info)])
        return rbln_config

    def forward(self, *args, return_dict: bool = None, **kwargs) -> torch.FloatTensor:
        # To ignore using attention_mask, we override forward method.
        output = super().forward(*args, **kwargs, return_dict=return_dict)
        import pdb

        pdb.set_trace()
        return output
