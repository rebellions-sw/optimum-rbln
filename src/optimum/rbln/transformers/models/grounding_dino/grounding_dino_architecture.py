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

from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers.models.grounding_dino.modeling_grounding_dino import (
    GroundingDinoDecoderOutput,
    GroundingDinoEncoderOutput,
    get_sine_pos_embed,
)

from ....utils.logging import get_logger
from .configuration_grounding_dino import RBLNGroundingDinoDecoderConfig, RBLNGroundingDinoEncoderConfig


# from transformers.models.grounding_dino.modeling_grounding_dino import generate_masks_with_special_tokens_and_transfer_map

logger = get_logger(__name__)

if TYPE_CHECKING:
    from transformers import (
        GroundingDinoModel,
    )


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


class GroundingDinoEncoder(torch.nn.Module):
    def __init__(self, model: "GroundingDinoEncoder", rbln_config: RBLNGroundingDinoEncoderConfig):
        super().__init__()
        self.layers = model.layers
        self.config = model.config
        self.rbln_config = rbln_config
        # FIXME: define Spatial shapes from config
        self.spatial_shapes = torch.tensor([[167, 167], [84, 84], [42, 42], [21, 21]])
        self.spatial_shapes_list = [(167, 167), (84, 84), (42, 42), (21, 21)]
        self.text_position_embedding = model.layers[0].get_text_position_embeddings(
            torch.zeros(1, model.config.max_text_len, model.config.d_model),
            None,
            torch.arange(model.config.max_text_len, dtype=torch.int32).unsqueeze(0),
        )

    # def infer_spatial_shapes(self):

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
        # text_position_ids: Optional[torch.Tensor] = None,
        reference_points: Optional[torch.Tensor] = None,
        # text_position_embedding: Optional[torch.Tensor] = None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
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
                text_position_embedding=self.text_position_embedding,
                text_self_attention_masks=text_self_attention_masks,
                # text_position_ids=text_position_ids,
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


class GroundingDinoDecoder(torch.nn.Module):
    def __init__(self, model: "GroundingDinoDecoder", rbln_config: RBLNGroundingDinoDecoderConfig):
        super().__init__()
        self.layers = model.layers
        self.config = model.config
        # FIXME: define Spatial shapes from config
        self.spatial_shapes = torch.tensor([[167, 167], [84, 84], [42, 42], [21, 21]])
        self.spatial_shapes_list = [(167, 167), (84, 84), (42, 42), (21, 21)]
        self.reference_points_head = model.reference_points_head
        self.bbox_embed = model.bbox_embed
        self.layer_norm = model.layer_norm

    def forward(
        self,
        inputs_embeds,
        vision_encoder_hidden_states,
        vision_encoder_attention_mask=None,
        text_encoder_hidden_states=None,
        text_encoder_attention_mask=None,
        reference_points=None,
        valid_ratios=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    ):
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`):
                The query embeddings that are passed into the decoder.
            vision_encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Last hidden state from encoder related to vision feature map.
            vision_encoder_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding pixel features. Mask values selected in `[0, 1]`:
                - 1 for pixel features that are real (i.e. **not masked**),
                - 0 for pixel features that are padding (i.e. **masked**).
            text_encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, text_seq_len, hidden_size)`):
                Last hidden state from encoder related to text features.
            text_encoder_attention_mask (`torch.Tensor` of shape `(batch_size, text_seq_len)`, *optional*):
                Mask to avoid performing attention on padding text features. Mask values selected in `[0, 1]`:
                - 0 for text features that are real (i.e. **not masked**),
                - 1 for text features that are padding (i.e. **masked**).
            reference_points (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)` is `as_two_stage` else `(batch_size, num_queries, 2)` or , *optional*):
                Reference point in range `[0, 1]`, top-left (0,0), bottom-right (1, 1), including padding area.
            spatial_shapes (`torch.FloatTensor` of shape `(num_feature_levels, 2)`):
                Spatial shapes of the feature maps.
            spatial_shapes_list (`List[Tuple[int, int]]`):
                Spatial shapes of the feature maps (but as list for export compatibility).
            level_start_index (`torch.LongTensor` of shape `(num_feature_levels)`, *optional*):
                Indexes for the start of each feature level. In range `[0, sequence_length]`.
            valid_ratios (`torch.FloatTensor` of shape `(batch_size, num_feature_levels, 2)`, *optional*):
                Ratio of valid area in each feature level.
            self_attn_mask (`torch.BoolTensor` of shape `(batch_size, text_seq_len)`):
                Masks to avoid performing self-attention between vision hidden state. Mask values selected in `[0, 1]`:
                - 1 for queries that are real (i.e. **not masked**),
                - 0 for queries that are padding (i.e. **masked**).
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

        if inputs_embeds is not None:
            hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_attns = () if output_attentions else None
        all_cross_attns_vision = () if (output_attentions and vision_encoder_hidden_states is not None) else None
        all_cross_attns_text = () if (output_attentions and text_encoder_hidden_states is not None) else None
        intermediate = ()
        intermediate_reference_points = ()

        if text_encoder_attention_mask is not None:
            dtype = text_encoder_hidden_states.dtype

            text_encoder_attention_mask = text_encoder_attention_mask[:, None, None, :]
            text_encoder_attention_mask = text_encoder_attention_mask.repeat(
                1, self.config.decoder_attention_heads, self.config.num_queries, 1
            )
            text_encoder_attention_mask = text_encoder_attention_mask.to(dtype=dtype)
            text_encoder_attention_mask = text_encoder_attention_mask * torch.finfo(dtype).min

        for idx, decoder_layer in enumerate(self.layers):
            num_coordinates = reference_points.shape[-1]
            if num_coordinates == 4:
                reference_points_input = (
                    reference_points[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
                )
            elif num_coordinates == 2:
                reference_points_input = reference_points[:, :, None] * valid_ratios[:, None]
            else:
                raise ValueError("Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]}")
            query_pos = get_sine_pos_embed(reference_points_input[:, :, 0, :], num_pos_feats=self.config.d_model // 2)
            query_pos = self.reference_points_head(query_pos)

            # In original implementation they apply layer norm before outputting intermediate hidden states
            # Though that's not through between layers so the layers use as input the output of the previous layer
            # withtout layer norm
            if output_hidden_states:
                all_hidden_states += (self.layer_norm(hidden_states),)

            layer_outputs = decoder_layer(
                hidden_states=hidden_states,
                position_embeddings=query_pos,
                reference_points=reference_points_input,
                spatial_shapes=self.spatial_shapes,
                spatial_shapes_list=self.spatial_shapes_list,
                level_start_index=None,
                vision_encoder_hidden_states=vision_encoder_hidden_states,
                vision_encoder_attention_mask=vision_encoder_attention_mask,
                text_encoder_hidden_states=text_encoder_hidden_states,
                text_encoder_attention_mask=text_encoder_attention_mask,
                self_attn_mask=None,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            # def custom_logits(x, eps=1e-5):
            #     z = torch.clamp(x, eps, 1 - eps)
            #     x = torch.clamp(1-z, eps, 1 - eps)
            #     return torch.log(z) - torch.log(x)

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[idx](hidden_states)
                num_coordinates = reference_points.shape[-1]
                if num_coordinates == 4:
                    new_reference_points = tmp + torch.special.logit(reference_points, eps=1e-5)
                    new_reference_points = new_reference_points.sigmoid()
                elif num_coordinates == 2:
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + torch.special.logit(reference_points, eps=1e-5)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    raise ValueError(
                        f"Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]}"
                    )
                reference_points = new_reference_points.detach()

            intermediate += (self.layer_norm(hidden_states),)
            intermediate_reference_points += (reference_points,)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if text_encoder_hidden_states is not None:
                    all_cross_attns_text += (layer_outputs[2],)

                if vision_encoder_hidden_states is not None:
                    all_cross_attns_vision += (layer_outputs[3],)

        # Keep batch_size as first dimension
        intermediate = torch.stack(intermediate, dim=1)
        intermediate_reference_points = torch.stack(intermediate_reference_points, dim=1)
        hidden_states = self.layer_norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if output_attentions:
            all_attns += (all_self_attns, all_cross_attns_text, all_cross_attns_vision)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    intermediate,
                    intermediate_reference_points,
                    all_hidden_states,
                    all_attns,
                ]
                if v is not None
            )
        return GroundingDinoDecoderOutput(
            last_hidden_state=hidden_states,
            intermediate_hidden_states=intermediate,
            intermediate_reference_points=intermediate_reference_points,
            hidden_states=all_hidden_states,
            attentions=all_attns,
        )


class _GroundingDinoMultiscaleDeformableAttention(torch.nn.Module):
    """
    Multiscale deformable attention as proposed in Deformable DETR.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        position_embeddings: Optional[torch.Tensor] = None,
        reference_points=None,
        spatial_shapes=None,
        spatial_shapes_list=None,
        level_start_index=None,
        output_attentions: bool = False,
    ):
        # add position embeddings to the hidden states before projecting to queries and keys
        if position_embeddings is not None:
            hidden_states = self.with_pos_embed(hidden_states, position_embeddings)

        batch_size, num_queries, _ = hidden_states.shape
        batch_size, sequence_length, _ = encoder_hidden_states.shape
        # Ignore copy
        if (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() != sequence_length:
            raise ValueError(
                "Make sure to align the spatial shapes with the sequence length of the encoder hidden states"
            )

        value = self.value_proj(encoder_hidden_states)
        if attention_mask is not None:
            # we invert the attention_mask
            # value = value.masked_fill(~attention_mask[..., None], float(0))

            # RBLN FIX
            mask = 1.0 - attention_mask[..., None]
            value = mask * value

        value = value.view(batch_size, sequence_length, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(hidden_states).view(
            batch_size, num_queries, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(hidden_states).view(
            batch_size, num_queries, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, -1).view(
            batch_size, num_queries, self.n_heads, self.n_levels, self.n_points
        )
        # batch_size, num_queries, n_heads, n_levels, n_points, 2
        num_coordinates = reference_points.shape[-1]
        if num_coordinates == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif num_coordinates == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(f"Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]}")

        output = self.attn(
            value,
            spatial_shapes,
            spatial_shapes_list,
            level_start_index,
            sampling_locations,
            attention_weights,
            self.im2col_step,
        )

        output = self.output_proj(output)

        return output, attention_weights


class _GroundingDinoBiMultiHeadAttention(torch.nn.Module):
    def forward(
        self,
        vision_features: torch.FloatTensor,
        text_features: torch.FloatTensor,
        vision_attention_mask: Optional[torch.BoolTensor] = None,
        text_attention_mask: Optional[torch.BoolTensor] = None,
    ) -> Tuple[Tuple[torch.FloatTensor, torch.FloatTensor], Tuple[torch.FloatTensor, torch.FloatTensor]]:
        """Image-to-text and text-to-image cross-attention

        Args:
            vision_features (`torch.FloatTensor` of shape `(batch_size, vision_sequence_length, hidden_dim)`):
                Projected flattened image features generated by the vision backbone.
            text_features (`torch.FloatTensor` of shape `(batch_size, text_sequence_length, hidden_dim)`):
                Projected text features generated by the text encoder.
            vision_attention_mask (`torch.BoolTensor`, **optional**):
                Attention mask for image-to-text cross-attention. False for real tokens and True for padding tokens.
            text_attention_mask (`torch.BoolTensor`, **optional**):
                Attention mask for text-to-image cross-attention. False for real tokens and True for padding tokens.

        Returns:
            `tuple(tuple(torch.FloatTensor), tuple(torch.FloatTensor))` where each inner tuple comprises an attention
            output and weights:
            - **vision_attn_output** (`torch.FloatTensor` of shape `(batch_size, vision_sequence_length, hidden_din)`)
              --
                Output of the image-to-text cross-attention layer.
            - **vision_attn_weights** (`torch.FloatTensor` of shape `(batch_size, num_heads, vision_sequence_length,
              vision_sequence_length)`) --
                Attention weights of the image-to-text cross-attention layer.
            - **text_attn_output** (`torch.FloatTensor` of shape `(batch_size, text_sequence_length, hidden_dim)`) --
                Output of the text-to-image cross-attention layer.
            - **text_attn_weights** (`torch.FloatTensor` of shape `(batch_size, num_heads, text_sequence_length,
              text_sequence_length)`) --
                Attention weights of the text-to-image cross-attention layer.
        """
        batch_size, tgt_len, _ = vision_features.size()

        vision_query_states = self.vision_proj(vision_features) * self.scale
        vision_query_states = self._reshape(vision_query_states, tgt_len, batch_size)

        text_key_states = self.text_proj(text_features)
        text_key_states = self._reshape(text_key_states, -1, batch_size)

        vision_value_states = self.values_vision_proj(vision_features)
        vision_value_states = self._reshape(vision_value_states, -1, batch_size)

        text_value_states = self.values_text_proj(text_features)
        text_value_states = self._reshape(text_value_states, -1, batch_size)

        proj_shape = (batch_size * self.num_heads, -1, self.head_dim)

        vision_query_states = vision_query_states.view(*proj_shape)
        text_key_states = text_key_states.view(*proj_shape)
        vision_value_states = vision_value_states.view(*proj_shape)
        text_value_states = text_value_states.view(*proj_shape)

        src_len = text_key_states.size(1)
        attn_weights = torch.bmm(vision_query_states, text_key_states.transpose(1, 2))  # bs*nhead, nimg, ntxt

        if attn_weights.size() != (batch_size * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        # RBLN FIX
        attn_weights = attn_weights - torch.max(attn_weights).reshape(1).repeat(src_len)
        # # Do not increase -50000/50000, data type half has quite limited range
        attn_weights = torch.clamp(attn_weights, min=-50000, max=50000)

        attn_weights_transposed = attn_weights.transpose(1, 2)
        # RBLN FIX
        text_attn_weights = attn_weights_transposed - torch.max(attn_weights_transposed, dim=-1, keepdim=True)[
            0
        ].repeat(1, 1, tgt_len)

        # # Do not increase -50000/50000, data type half has quite limited range
        text_attn_weights = torch.clamp(text_attn_weights, min=-50000, max=50000)

        # mask vision for language
        if vision_attention_mask is not None:
            vision_attention_mask = (
                vision_attention_mask[:, None, None, :].repeat(1, self.num_heads, 1, 1).flatten(0, 1)
            )
            # RBLN FIX
            mask = (1 - vision_attention_mask) * torch.finfo(torch.float32).min
            text_attn_weights = text_attn_weights + mask

        text_attn_weights = text_attn_weights.softmax(dim=-1)

        # mask language for vision
        if text_attention_mask is not None:
            text_attention_mask = text_attention_mask[:, None, None, :].repeat(1, self.num_heads, 1, 1).flatten(0, 1)
            # RBLN FIX
            attn_weights = attn_weights + ((1 - text_attention_mask) * float("-inf"))
        vision_attn_weights = attn_weights.softmax(dim=-1)

        vision_attn_probs = F.dropout(vision_attn_weights, p=self.dropout, training=self.training)
        text_attn_probs = F.dropout(text_attn_weights, p=self.dropout, training=self.training)

        vision_attn_output = torch.bmm(vision_attn_probs, text_value_states)
        text_attn_output = torch.bmm(text_attn_probs, vision_value_states)

        if vision_attn_output.size() != (batch_size * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`vision_attn_output` should be of size {(batch_size, self.num_heads, tgt_len, self.head_dim)}, but is {vision_attn_output.size()}"
            )

        if text_attn_output.size() != (batch_size * self.num_heads, src_len, self.head_dim):
            raise ValueError(
                f"`text_attn_output` should be of size {(batch_size, self.num_heads, src_len, self.head_dim)}, but is {text_attn_output.size()}"
            )

        vision_attn_output = vision_attn_output.view(batch_size, self.num_heads, tgt_len, self.head_dim)
        vision_attn_output = vision_attn_output.transpose(1, 2)
        vision_attn_output = vision_attn_output.reshape(batch_size, tgt_len, self.embed_dim)

        text_attn_output = text_attn_output.view(batch_size, self.num_heads, src_len, self.head_dim)
        text_attn_output = text_attn_output.transpose(1, 2)
        text_attn_output = text_attn_output.reshape(batch_size, src_len, self.embed_dim)

        vision_attn_output = self.out_vision_proj(vision_attn_output)
        text_attn_output = self.out_text_proj(text_attn_output)

        return (vision_attn_output, vision_attn_weights), (text_attn_output, text_attn_weights)


class _GroundingDinoEncoderLayer(torch.nn.Module):
    def forward(
        self,
        vision_features: Tensor,
        vision_position_embedding: Tensor,
        spatial_shapes: Tensor,
        spatial_shapes_list: List[Tuple[int, int]],
        level_start_index: Tensor,
        key_padding_mask: Tensor,
        reference_points: Tensor,
        text_features: Optional[Tensor] = None,
        text_attention_mask: Optional[Tensor] = None,
        text_position_embedding: Optional[Tensor] = None,
        text_self_attention_masks: Optional[Tensor] = None,
        text_position_ids: Optional[Tensor] = None,
    ):
        text_position_embedding = self.get_text_position_embeddings(
            text_features, text_position_embedding, text_position_ids
        )

        (vision_features, vision_fused_attn), (text_features, text_fused_attn) = self.fusion_layer(
            vision_features=vision_features,
            text_features=text_features,
            attention_mask_vision=key_padding_mask,
            attention_mask_text=text_attention_mask,
        )

        (text_features, text_enhanced_attn) = self.text_enhancer_layer(
            hidden_states=text_features,
            attention_masks=(1.0 - text_self_attention_masks),  # note we use ~ for mask here
            position_embeddings=(text_position_embedding if text_position_embedding is not None else None),
        )

        (vision_features, vision_deformable_attn) = self.deformable_layer(
            hidden_states=vision_features,
            attention_mask=(1.0 - key_padding_mask),
            position_embeddings=vision_position_embedding,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            level_start_index=level_start_index,
        )

        return (
            (vision_features, text_features),
            (vision_fused_attn, text_fused_attn, text_enhanced_attn, vision_deformable_attn),
        )


original_forward = None
original_bi_multihead_attention_forward = None
original_encoder_layer_forward = None


def monkey_patch():
    from transformers.models.grounding_dino.modeling_grounding_dino import (
        GroundingDinoBiMultiHeadAttention,
        GroundingDinoEncoderLayer,
        GroundingDinoMultiscaleDeformableAttention,
    )

    original_forward = GroundingDinoMultiscaleDeformableAttention.forward
    original_bi_multihead_attention_forward = GroundingDinoBiMultiHeadAttention.forward
    original_encoder_layer_forward = GroundingDinoEncoderLayer.forward

    # Patch the methods with the custom implementations
    GroundingDinoMultiscaleDeformableAttention.forward = _GroundingDinoMultiscaleDeformableAttention.forward
    GroundingDinoBiMultiHeadAttention.forward = _GroundingDinoBiMultiHeadAttention.forward
    GroundingDinoEncoderLayer.forward = _GroundingDinoEncoderLayer.forward


def restore_monkey_patch():
    from transformers.models.grounding_dino.modeling_grounding_dino import (
        GroundingDinoBiMultiHeadAttention,
        GroundingDinoEncoderLayer,
        GroundingDinoMultiscaleDeformableAttention,
    )

    # Restore the original methods
    GroundingDinoMultiscaleDeformableAttention.forward = original_forward
    GroundingDinoBiMultiHeadAttention.forward = original_bi_multihead_attention_forward
    GroundingDinoEncoderLayer.forward = original_encoder_layer_forward
