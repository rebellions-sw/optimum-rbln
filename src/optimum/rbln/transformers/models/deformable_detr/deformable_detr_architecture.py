# coding=utf-8
# Copyright 2022 SenseTime and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Deformable DETR model."""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.deformable_detr.configuration_deformable_detr import DeformableDetrConfig
from transformers.models.deformable_detr.modeling_deformable_detr import (
    DeformableDetrConvModel,
    DeformableDetrDecoderOutput,
    DeformableDetrForObjectDetection,
    DeformableDetrLearnedPositionEmbedding,
    DeformableDetrModelOutput,
)
from transformers.models.deformable_detr.modeling_deformable_detr import (
    DeformableDetrDecoder as HFDeformableDetrDecoder,
)
from transformers.models.deformable_detr.modeling_deformable_detr import (
    DeformableDetrDecoderLayer as HFDeformableDetrDecoderLayer,
)
from transformers.models.deformable_detr.modeling_deformable_detr import (
    DeformableDetrEncoder as HFDeformableDetrEncoder,
)
from transformers.models.deformable_detr.modeling_deformable_detr import (
    DeformableDetrEncoderLayer as HFDeformableDetrEncoderLayer,
)
from transformers.models.deformable_detr.modeling_deformable_detr import (
    DeformableDetrModel as HFDeformableDetrModel,
)
from transformers.models.deformable_detr.modeling_deformable_detr import (
    DeformableDetrMultiscaleDeformableAttention as HFDeformableDetrMultiscaleDeformableAttention,
)
from transformers.pytorch_utils import meshgrid
from transformers.utils import logging


logger = logging.get_logger(__name__)


class MultiScaleDeformableAttention(nn.Module):
    def forward(
        self,
        value: Tensor,
        value_spatial_shapes: Tensor,
        level_start_index: Tensor,
        sampling_locations: Tensor,
        attention_weights: Tensor,
        im2col_step: int,
    ):
        batch_size, _, num_heads, hidden_dim = value.shape
        _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
        # RBLN modified
        # value_list = value.split([height * width for height, width in value_spatial_shapes], dim=1)
        value_list = []
        index_list = [height * width for height, width in value_spatial_shapes]
        start = 0
        for index in index_list:
            end = start + index
            value_list.append(value[:, start:end])
            start = end
        sampling_grids = 2 * sampling_locations - 1
        sampling_value_list = []
        for level_id, (height, width) in enumerate(value_spatial_shapes):
            # batch_size, height*width, num_heads, hidden_dim
            # -> batch_size, height*width, num_heads*hidden_dim
            # -> batch_size, num_heads*hidden_dim, height*width
            # -> batch_size*num_heads, hidden_dim, height, width
            value_l_ = (
                value_list[level_id]
                .flatten(2)
                .transpose(1, 2)
                .reshape(batch_size * num_heads, hidden_dim, height, width)
            )
            # batch_size, num_queries, num_heads, num_points, 2
            # -> batch_size, num_heads, num_queries, num_points, 2
            # -> batch_size*num_heads, num_queries, num_points, 2
            sampling_grid_l_ = sampling_grids[:, :, :, level_id].transpose(1, 2).flatten(0, 1)
            # batch_size*num_heads, hidden_dim, num_queries, num_points
            sampling_value_l_ = nn.functional.grid_sample(
                value_l_,
                sampling_grid_l_,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            sampling_value_list.append(sampling_value_l_)
        # (batch_size, num_queries, num_heads, num_levels, num_points)
        # -> (batch_size, num_heads, num_queries, num_levels, num_points)
        # -> (batch_size, num_heads, 1, num_queries, num_levels*num_points)
        attention_weights = attention_weights.transpose(1, 2).reshape(
            batch_size * num_heads, 1, num_queries, num_levels * num_points
        )
        output = (
            (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
            .sum(-1)
            .view(batch_size, num_heads * hidden_dim, num_queries)
        )
        return output.transpose(1, 2).contiguous()


class DeformableDetrSinePositionEmbedding(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one used by the Attention is all you
    need paper, generalized to work on images.
    """

    def __init__(self, embedding_dim=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

        dim_t = torch.arange(self.embedding_dim, dtype=torch.int64, device="cpu")
        self.dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.embedding_dim)

    def forward(self, pixel_values, pixel_mask):
        if pixel_mask is None:
            raise ValueError("No pixel mask provided")
        y_embed = pixel_mask.cumsum(1, dtype=pixel_values.dtype)
        x_embed = pixel_mask.cumsum(2, dtype=pixel_values.dtype)
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        pos_x = x_embed[:, :, :, None] / self.dim_t
        pos_y = y_embed[:, :, :, None] / self.dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


# Copied from transformers.models.detr.modeling_detr.build_position_encoding with Detr->DeformableDetr
def build_position_encoding(config):
    n_steps = config.d_model // 2
    if config.position_embedding_type == "sine":
        # TODO find a better way of exposing other arguments
        position_embedding = DeformableDetrSinePositionEmbedding(n_steps, normalize=True)
    elif config.position_embedding_type == "learned":
        position_embedding = DeformableDetrLearnedPositionEmbedding(n_steps)
    else:
        raise ValueError(f"Not supported {config.position_embedding_type}")

    return position_embedding


class DeformableDetrMultiscaleDeformableAttention(nn.Module):
    """
    Multiscale deformable attention as proposed in Deformable DETR.
    """

    def __init__(
        self,
        model: HFDeformableDetrMultiscaleDeformableAttention,
        config: DeformableDetrConfig,
        num_heads: int,
        n_points: int,
    ):
        super().__init__()

        self._original_model = model
        self.attn = MultiScaleDeformableAttention()

        if config.d_model % num_heads != 0:
            raise ValueError(
                f"embed_dim (d_model) must be divisible by num_heads, but got {config.d_model} and {num_heads}"
            )

        self.im2col_step = 64

        self.d_model = config.d_model
        self.n_levels = config.num_feature_levels
        self.n_heads = num_heads
        self.n_points = n_points

        self.sampling_offsets = model.sampling_offsets  #  nn.Linear
        self.attention_weights = model.attention_weights  #  nn.Linear
        self.value_proj = model.value_proj  #  nn.Linear
        self.output_proj = model.output_proj  #  nn.Linear

        self.disable_custom_kernels = config.disable_custom_kernels

    def with_pos_embed(self, tensor: torch.Tensor, position_embeddings: Optional[Tensor]):
        return tensor if position_embeddings is None else tensor + position_embeddings

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        position_embeddings: Optional[torch.Tensor] = None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        output_attentions: bool = False,
    ):
        # add position embeddings to the hidden states before projecting to queries and keys
        if position_embeddings is not None:
            hidden_states = self.with_pos_embed(hidden_states, position_embeddings)

        batch_size, num_queries, _ = hidden_states.shape
        batch_size, sequence_length, _ = encoder_hidden_states.shape
        total_elements = sum(height * width for height, width in spatial_shapes)
        if total_elements != sequence_length:
            raise ValueError(
                "Make sure to align the spatial shapes with the sequence length of the encoder hidden states"
            )

        value = self.value_proj(encoder_hidden_states)
        if attention_mask is not None:
            # we invert the attention_mask
            value = value * attention_mask[..., None].to(value.dtype)
            # value = value * attention_mask[..., None]
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
            offset_normalizer = torch.concat([spatial_shapes[..., 1:2], spatial_shapes[..., 0:1]], dim=-1)
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
            level_start_index,
            sampling_locations,
            attention_weights,
            self.im2col_step,
        )

        output = self.output_proj(output)

        return output, attention_weights


class DeformableDetrEncoderLayer(nn.Module):
    def __init__(self, model: HFDeformableDetrEncoderLayer, config: DeformableDetrConfig):
        super().__init__()
        self._original_model = model
        self.embed_dim = config.d_model
        self.self_attn = DeformableDetrMultiscaleDeformableAttention(
            self._original_model.self_attn,
            config,
            num_heads=config.encoder_attention_heads,
            n_points=config.encoder_n_points,
        )
        self.self_attn_layer_norm = self._original_model.self_attn_layer_norm
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = self._original_model.fc1
        self.fc2 = self._original_model.fc2
        self.final_layer_norm = self._original_model.final_layer_norm

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: torch.Tensor = None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        output_attentions: bool = False,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Input to the layer.
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
                Attention mask.
            position_embeddings (`torch.FloatTensor`, *optional*):
                Position embeddings, to be added to `hidden_states`.
            reference_points (`torch.FloatTensor`, *optional*):
                Reference points.
            spatial_shapes (`torch.LongTensor`, *optional*):
                Spatial shapes of the backbone feature maps.
            level_start_index (`torch.LongTensor`, *optional*):
                Level start index.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        # Apply Multi-scale Deformable Attention Module on the multi-scale feature maps.
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            output_attentions=output_attentions,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=False)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=False)

        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=False)

        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class DeformableDetrEncoder(nn.Module):
    """
    Transformer encoder consisting of *config.encoder_layers* deformable attention layers. Each layer is a
    [`DeformableDetrEncoderLayer`].

    The encoder updates the flattened multi-scale feature maps through multiple deformable attention layers.

    Args:
        config: DeformableDetrConfig
    """

    def __init__(self, model: HFDeformableDetrEncoder, config: DeformableDetrConfig):
        super().__init__()
        self.config = config
        self._original_model = model

        self.dropout = config.dropout
        self.layers = nn.ModuleList(
            [DeformableDetrEncoderLayer(model.layers[idx], config) for idx in range(config.encoder_layers)]
        )

    def forward(
        self,
        inputs_embeds=None,
        attention_mask=None,
        position_embeddings=None,
        spatial_shapes=None,
        level_start_index=None,
        reference_points=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Flattened feature map (output of the backbone + projection layer) that is passed to the encoder.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding pixel features. Mask values selected in `[0, 1]`:
                - 1 for pixel features that are real (i.e. **not masked**),
                - 0 for pixel features that are padding (i.e. **masked**).
                [What are attention masks?](../glossary#attention-mask)
            position_embeddings (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Position embeddings that are added to the queries and keys in each self-attention layer.
            spatial_shapes (`torch.LongTensor` of shape `(num_feature_levels, 2)`):
                Spatial shapes of each feature map.
            level_start_index (`torch.LongTensor` of shape `(num_feature_levels)`):
                Starting index of each feature map.
            valid_ratios (`torch.FloatTensor` of shape `(batch_size, num_feature_levels, 2)`):
                Ratio of valid area in each feature level.
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

        hidden_states = inputs_embeds
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=False)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for i, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask,
                position_embeddings=position_embeddings,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )


class DeformableDetrDecoderLayer(nn.Module):
    def __init__(self, model: HFDeformableDetrDecoderLayer, config: DeformableDetrConfig):
        super().__init__()
        self._original_model = model
        self.embed_dim = config.d_model

        # self-attention
        self.self_attn = self._original_model.self_attn
        self.dropout = config.dropout
        self.activation_fn = self._original_model.activation_fn
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = self._original_model.self_attn_layer_norm
        # cross-attention
        self.encoder_attn = DeformableDetrMultiscaleDeformableAttention(
            self._original_model.encoder_attn,
            config,
            num_heads=config.decoder_attention_heads,
            n_points=config.decoder_n_points,
        )
        self.encoder_attn_layer_norm = self._original_model.encoder_attn_layer_norm
        # feedforward neural networks
        self.fc1 = self._original_model.fc1
        self.fc2 = self._original_model.fc2
        self.final_layer_norm = self._original_model.final_layer_norm

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[torch.Tensor] = None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(seq_len, batch, embed_dim)`.
            position_embeddings (`torch.FloatTensor`, *optional*):
                Position embeddings that are added to the queries and keys in the self-attention layer.
            reference_points (`torch.FloatTensor`, *optional*):
                Reference points.
            spatial_shapes (`torch.LongTensor`, *optional*):
                Spatial shapes.
            level_start_index (`torch.LongTensor`, *optional*):
                Level start index.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, target_len, source_len)` where padding elements are indicated by very large negative
                values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=False)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        second_residual = hidden_states

        # Cross-Attention
        cross_attn_weights = None
        hidden_states, cross_attn_weights = self.encoder_attn(
            hidden_states=hidden_states,
            attention_mask=encoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            position_embeddings=position_embeddings,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            output_attentions=output_attentions,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=False)
        hidden_states = second_residual + hidden_states

        hidden_states = self.encoder_attn_layer_norm(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=False)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=False)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        return outputs


class DeformableDetrDecoder(nn.Module):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`DeformableDetrDecoderLayer`].

    The decoder updates the query embeddings through multiple self-attention and cross-attention layers.

    Some tweaks for Deformable DETR:

    - `position_embeddings`, `reference_points`, `spatial_shapes` and `valid_ratios` are added to the forward pass.
    - it also returns a stack of intermediate outputs and reference points from all decoding layers.

    Args:
        config: DeformableDetrConfig
    """

    def __init__(self, model: HFDeformableDetrDecoder, config: DeformableDetrConfig):
        super().__init__()

        self._original_model = model
        self.config = config

        self.dropout = config.dropout
        self.layers = nn.ModuleList(
            [DeformableDetrDecoderLayer(model.layers[idx], config) for idx in range(config.decoder_layers)]
        )

        self.bbox_embed = None

    def forward(
        self,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        position_embeddings=None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        valid_ratios=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`):
                The query embeddings that are passed into the decoder.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing cross-attention on padding pixel_values of the encoder. Mask values selected
                in `[0, 1]`:
                - 1 for pixels that are real (i.e. **not masked**),
                - 0 for pixels that are padding (i.e. **masked**).
            position_embeddings (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`, *optional*):
                Position embeddings that are added to the queries and keys in each self-attention layer.
            reference_points (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)` is `as_two_stage` else `(batch_size, num_queries, 2)` or , *optional*):
                Reference point in range `[0, 1]`, top-left (0,0), bottom-right (1, 1), including padding area.
            spatial_shapes (`torch.FloatTensor` of shape `(num_feature_levels, 2)`):
                Spatial shapes of the feature maps.
            level_start_index (`torch.LongTensor` of shape `(num_feature_levels)`, *optional*):
                Indexes for the start of each feature level. In range `[0, sequence_length]`.
            valid_ratios (`torch.FloatTensor` of shape `(batch_size, num_feature_levels, 2)`, *optional*):
                Ratio of valid area in each feature level.

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
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        intermediate = ()
        intermediate_reference_points = ()

        for idx, decoder_layer in enumerate(self.layers):
            num_coordinates = reference_points.shape[-1]
            if num_coordinates == 4:
                reference_points_input = (
                    reference_points[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
                )
            elif reference_points.shape[-1] == 2:
                reference_points_input = reference_points[:, :, None] * valid_ratios[:, None]
            else:
                raise ValueError("Reference points' last dimension must be of size 2")

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                encoder_hidden_states=encoder_hidden_states,
                reference_points=reference_points_input,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                raise NotImplementedError(
                    "RBLNDetrObjectDetection not support to hack implementation for iterative bounding box refinement."
                )

            intermediate += (hidden_states,)
            intermediate_reference_points += (reference_points,)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # Keep batch_size as first dimension
        intermediate = torch.stack(intermediate, dim=1)
        intermediate_reference_points = torch.stack(intermediate_reference_points, dim=1)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    intermediate,
                    intermediate_reference_points,
                    all_hidden_states,
                    all_self_attns,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return DeformableDetrDecoderOutput(
            last_hidden_state=hidden_states,
            intermediate_hidden_states=intermediate,
            intermediate_reference_points=intermediate_reference_points,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class DeformableDetrModel(nn.Module):
    def __init__(self, model: HFDeformableDetrModel, config: DeformableDetrConfig):
        super().__init__()

        self.config = config
        self._original_model = model

        # Create backbone + positional encoding
        backbone = self._original_model.backbone.conv_encoder
        position_embeddings = build_position_encoding(config)
        self.backbone = DeformableDetrConvModel(backbone, position_embeddings)

        # Create input projection layers
        self.input_proj = model.input_proj

        if not config.two_stage:
            self.query_position_embeddings = model.query_position_embeddings

        self.encoder = DeformableDetrEncoder(model.encoder, config)
        self.decoder = DeformableDetrDecoder(model.decoder, config)

        self.level_embed = model.level_embed

        if config.two_stage:
            self.enc_output = model.enc_output
            self.enc_output_norm = model.enc_output_norm
            self.pos_trans = model.pos_trans
            self.pos_trans_norm = model.pos_trans_norm
        else:
            self.reference_points = model.reference_points

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_valid_ratio(self, mask, dtype=torch.float32):
        """Get the valid ratio of all feature maps."""

        _, height, width = mask.shape
        valid_height = torch.sum(mask[:, :, 0], 1)
        valid_width = torch.sum(mask[:, 0, :], 1)
        valid_ratio_height = valid_height.to(dtype) / height
        valid_ratio_width = valid_width.to(dtype) / width
        valid_ratio = torch.stack([valid_ratio_width, valid_ratio_height], -1)
        return valid_ratio

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """
        Get reference points for each feature map. Used in decoder.

        Args:
            spatial_shapes (`torch.LongTensor` of shape `(num_feature_levels, 2)`):
                Spatial shapes of each feature map.
            valid_ratios (`torch.FloatTensor` of shape `(batch_size, num_feature_levels, 2)`):
                Valid ratios of each feature map.
            device (`torch.device`):
                Device on which to create the tensors.
        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_queries, num_feature_levels, 2)`
        """

        # RBLN changes linspace to arange
        def linspace_to_arange(start, end, steps, dtype, device):
            assert steps != 1, "linspace_to_arange does not support step 1."
            step = (float(end) - float(start)) / (float(steps) - 1.0)
            return torch.arange(start, float(end) + float(step) / 2.0, int(step), dtype=dtype, device=device)

        reference_points_list = []
        for level, (height, width) in enumerate(spatial_shapes):
            ref_y, ref_x = meshgrid(
                linspace_to_arange(0.5, height - 0.5, height, dtype=valid_ratios.dtype, device=device),
                linspace_to_arange(0.5, width - 0.5, width, dtype=valid_ratios.dtype, device=device),
                indexing="ij",
            )
            # TODO: valid_ratios could be useless here. check https://github.com/fundamentalvision/Deformable-DETR/issues/36
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, level, 1] * height)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, level, 0] * width)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.FloatTensor] = None,
        encoder_outputs: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], DeformableDetrModelOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, DeformableDetrModel
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("SenseTime/deformable-detr")
        >>> model = DeformableDetrModel.from_pretrained("SenseTime/deformable-detr")

        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)

        >>> last_hidden_states = outputs.last_hidden_state
        >>> list(last_hidden_states.shape)
        [1, 300, 256]
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, num_channels, height, width = pixel_values.shape
        device = pixel_values.device

        if pixel_mask is None:
            pixel_mask = torch.ones(((batch_size, height, width)), dtype=torch.long, device=device)

        # Extract multi-scale feature maps of same resolution `config.d_model` (cf Figure 4 in paper)
        # First, sent pixel_values + pixel_mask through Backbone to obtain the features
        # which is a list of tuples
        features, position_embeddings_list = self.backbone(pixel_values, pixel_mask)

        # Then, apply 1x1 convolution to reduce the channel dimension to d_model (256 by default)
        sources = []
        masks = []
        for level, (source, mask) in enumerate(features):
            sources.append(self.input_proj[level](source))
            masks.append(mask)
            if mask is None:
                raise ValueError("No attention mask was provided")

        # Lowest resolution feature maps are obtained via 3x3 stride 2 convolutions on the final stage
        if self.config.num_feature_levels > len(sources):
            _len_sources = len(sources)
            for level in range(_len_sources, self.config.num_feature_levels):
                if level == _len_sources:
                    source = self.input_proj[level](features[-1][0])
                else:
                    source = self.input_proj[level](sources[-1])
                mask = nn.functional.interpolate(pixel_mask[None].to(pixel_values.dtype), size=source.shape[-2:]).to(
                    torch.bool
                )[0]
                pos_l = self.backbone.position_embedding(source, mask).to(source.dtype)
                sources.append(source)
                masks.append(mask)
                position_embeddings_list.append(pos_l)

        # Create queries
        query_embeds = None
        if not self.config.two_stage:
            query_embeds = self.query_position_embeddings.weight

        # Prepare encoder inputs (by flattening)
        source_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes_list = []
        for level, (source, mask, pos_embed) in enumerate(zip(sources, masks, position_embeddings_list)):
            batch_size, num_channels, height, width = source.shape
            spatial_shape = [height, width]
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
        valid_ratios = torch.stack([self.get_valid_ratio(m, dtype=source_flatten.dtype) for m in masks], 1)
        # RBLN moves get_reference_points from DeformableDetrForEncoder
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=source_flatten.device)

        # Fourth, sent source_flatten + mask_flatten + lvl_pos_embed_flatten (backbone + proj layer output) through encoder
        # Also provide spatial_shapes, level_start_index and valid_ratios
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                inputs_embeds=source_flatten,
                attention_mask=mask_flatten,
                position_embeddings=lvl_pos_embed_flatten,
                spatial_shapes=spatial_shapes,
                reference_points=reference_points,
                level_start_index=level_start_index,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # Fifth, prepare decoder inputs
        batch_size, _, num_channels = encoder_outputs[0].shape
        enc_outputs_class = None
        enc_outputs_coord_logits = None
        if self.config.two_stage:
            raise NotImplementedError("The two-stage mode is not supported in RBLNDetrObjectDetection.")
        else:
            query_embed, target = torch.split(query_embeds, num_channels, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(batch_size, -1, -1)
            target = target.unsqueeze(0).expand(batch_size, -1, -1)
            reference_points = self.reference_points(query_embed).sigmoid()
            init_reference_points = reference_points

        decoder_outputs = self.decoder(
            inputs_embeds=target,
            position_embeddings=query_embed,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            enc_outputs = tuple(value for value in [enc_outputs_class, enc_outputs_coord_logits] if value is not None)
            tuple_outputs = (init_reference_points,) + decoder_outputs + encoder_outputs + enc_outputs

            return tuple_outputs

        return DeformableDetrModelOutput(
            init_reference_points=init_reference_points,
            last_hidden_state=decoder_outputs.last_hidden_state,
            intermediate_hidden_states=decoder_outputs.intermediate_hidden_states,
            intermediate_reference_points=decoder_outputs.intermediate_reference_points,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            enc_outputs_class=enc_outputs_class,
            enc_outputs_coord_logits=enc_outputs_coord_logits,
        )


class DeformableDetrForObjectDetectionWrapper(nn.Module):
    def __init__(self, model: DeformableDetrForObjectDetection):
        super().__init__()
        self.model = model
        self.config = model.config

        model.model = DeformableDetrModel(model.model, self.config)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
