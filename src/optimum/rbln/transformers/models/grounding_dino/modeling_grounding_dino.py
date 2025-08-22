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
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from transformers.modeling_utils import no_init_weights
from transformers.models.grounding_dino.modeling_grounding_dino import (
    GroundingDinoContrastiveEmbedding,
    GroundingDinoConvEncoder,
    GroundingDinoConvModel,
    GroundingDinoDecoderOutput,
    GroundingDinoEncoderOutput,
    GroundingDinoMLPPredictionHead,
    GroundingDinoModel,
    GroundingDinoModelOutput,
    GroundingDinoObjectDetectionOutput,
    build_position_encoding,
    generate_masks_with_special_tokens_and_transfer_map,
)
from transformers.pytorch_utils import meshgrid

from ....configuration_utils import RBLNCompileConfig, RBLNModelConfig
from ....modeling import RBLNModel
from ....utils.logging import get_logger
from ....utils.runtime_utils import RBLNPytorchRuntime
from .configuration_grounding_dino import (
    RBLNGroundingDinoDecoderConfig,
    RBLNGroundingDinoEncoderConfig,
    RBLNGroundingDinoForObjectDetectionConfig,
)
from .grounding_dino_architecture import (
    _GroundingDinoDecoder,
    _GroundingDinoEncoder,
)


logger = get_logger(__name__)

if TYPE_CHECKING:
    from transformers import (
        AutoFeatureExtractor,
        AutoProcessor,
        AutoTokenizer,
        PreTrainedModel,
    )

    from ....diffusers.modeling_diffusers import RBLNDiffusionMixin, RBLNDiffusionMixinConfig


class RBLNGroundingDinoForObjectDetection(RBLNModel):
    _rbln_submodules = [
        {"name": "text_backbone"},
        # {"name": "backbone"},
        {"name": "encoder"},
        {"name": "decoder"},
    ]
    # FIXME: Update the docstring to reflect the actual model functionality.
    """
    RBLN optimized CLIP text encoder model.
    This class provides hardware-accelerated inference for CLIP text encoders
    on RBLN devices, supporting text encoding for multimodal tasks.
    """

    def __post_init__(self, **kwargs):
        self.setup_cpu_instances()
        self.text_projection = RBLNPytorchRuntime(self.model[0])
        self.text_backbone = self.rbln_submodules[0]
        self.encoder = self.rbln_submodules[1]
        self.decoder = self.rbln_submodules[2]

    def setup_cpu_instances(self):
        stacte_dict = torch.load(self.model_save_dir / self.subfolder / "torch_artifacts.pth", weights_only=False)
        with no_init_weights():
            config = self.config
            _class_embed = GroundingDinoContrastiveEmbedding(config)
            if config.decoder_bbox_embed_share:  # True
                _bbox_embed = GroundingDinoMLPPredictionHead(
                    input_dim=config.d_model, hidden_dim=config.d_model, output_dim=4, num_layers=3
                )
                self.bbox_embed = nn.ModuleList([_bbox_embed for _ in range(config.decoder_layers)])
            else:
                for _ in range(config.decoder_layers):
                    _bbox_embed = GroundingDinoMLPPredictionHead(
                        input_dim=config.d_model, hidden_dim=config.d_model, output_dim=4, num_layers=3
                    )
                    self.bbox_embed = nn.ModuleList([_bbox_embed for _ in range(config.decoder_layers)])
            self.class_embed = nn.ModuleList([_class_embed for _ in range(config.decoder_layers)])

            backbone = GroundingDinoConvEncoder(config)
            position_embeddings = build_position_encoding(config)
            self.backbone = GroundingDinoConvModel(backbone, position_embeddings).eval()

            # Create input projection layers
            if config.num_feature_levels > 1:
                num_backbone_outs = len(backbone.intermediate_channel_sizes)
                input_proj_list = []
                for i in range(num_backbone_outs):
                    in_channels = backbone.intermediate_channel_sizes[i]
                    input_proj_list.append(
                        nn.Sequential(
                            nn.Conv2d(in_channels, config.d_model, kernel_size=1),
                            nn.GroupNorm(32, config.d_model),
                        )
                    )
                for _ in range(config.num_feature_levels - num_backbone_outs):
                    input_proj_list.append(
                        nn.Sequential(
                            nn.Conv2d(in_channels, config.d_model, kernel_size=3, stride=2, padding=1),
                            nn.GroupNorm(32, config.d_model),
                        )
                    )
                    in_channels = config.d_model
                self.input_proj_vision = nn.ModuleList(input_proj_list)
            else:
                self.input_proj_vision = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Conv2d(backbone.intermediate_channel_sizes[-1], config.d_model, kernel_size=1),
                            nn.GroupNorm(32, config.d_model),
                        )
                    ]
                )

            if config.embedding_init_target or not config.two_stage:
                self.query_position_embeddings = nn.Embedding(config.num_queries, config.d_model)

            self.level_embed = nn.Parameter(torch.Tensor(config.num_feature_levels, config.d_model))

            if config.two_stage:
                self.enc_output = nn.Linear(config.d_model, config.d_model)
                self.enc_output_norm = nn.LayerNorm(config.d_model, config.layer_norm_eps)
                if (
                    config.two_stage_bbox_embed_share
                    and config.decoder_bbox_embed_share
                    and self.decoder.bbox_embed is not None
                ):
                    self.encoder_output_bbox_embed = self.decoder.bbox_embed
                else:
                    self.encoder_output_bbox_embed = GroundingDinoMLPPredictionHead(
                        input_dim=config.d_model, hidden_dim=config.d_model, output_dim=4, num_layers=3
                    )

                self.encoder_output_class_embed = GroundingDinoContrastiveEmbedding(config)
            else:
                self.reference_points = nn.Embedding(config.num_queries, 4)

        # # Create text backbone
        #     self.text_backbone = AutoModel.from_config(config.text_config, add_pooling_layer=False).eval()
        #     self._decoder = GroundingDinoDecoder(config).eval()
        #     self._encoder = GroundingDinoEncoder(config).eval()
        #     self.text_projection = nn.Linear(config.text_config.hidden_size, config.d_model)

        # self._encoder.load_state_dict(stacte_dict["encoder"])
        # # self._decoder.load_state_dict(stacte_dict["decoder"])
        # self.text_projection.load_state_dict(stacte_dict["text_projection"])
        # self.text_backbone.load_state_dict(stacte_dict["text_backbone"])

        self.backbone.load_state_dict(stacte_dict["backbone"])
        self.bbox_embed.load_state_dict(stacte_dict["bbox_embed"])
        self.class_embed.load_state_dict(stacte_dict["class_embed"])
        self.input_proj_vision.load_state_dict(stacte_dict["input_proj_vision"])
        with torch.no_grad():
            self.level_embed.copy_(stacte_dict["level_embed"])
        if self.config.two_stage:
            self.enc_output.load_state_dict(stacte_dict["enc_output"])
            self.enc_output_norm.load_state_dict(stacte_dict["enc_output_norm"])
            self.encoder_output_class_embed.load_state_dict(stacte_dict["encoder_output_class_embed"])
            self.encoder_output_bbox_embed.load_state_dict(stacte_dict["encoder_output_bbox_embed"])
        else:
            self.reference_points.load_state_dict(stacte_dict["reference_points"])
        if self.config.embedding_init_target or not self.config.two_stage:
            self.query_position_embeddings.load_state_dict(stacte_dict["query_position_embeddings"])

    @classmethod
    def save_torch_artifacts(
        cls,
        model: "PreTrainedModel",
        save_dir_path: Path,
        subfolder: str,
        rbln_config: RBLNGroundingDinoForObjectDetectionConfig,
    ):
        # If you are unavoidably running on a CPU rather than an RBLN device,
        # store the torch tensor, weight, etc. in this function.
        save_dict = {}
        save_dict["backbone"] = model.model.backbone.state_dict()
        save_dict["input_proj_vision"] = model.model.input_proj_vision.state_dict()
        save_dict["level_embed"] = model.model.level_embed
        if model.config.two_stage:
            save_dict["enc_output"] = model.model.enc_output.state_dict()
            save_dict["enc_output_norm"] = model.model.enc_output_norm.state_dict()
            save_dict["encoder_output_class_embed"] = model.model.encoder_output_class_embed.state_dict()
            save_dict["encoder_output_bbox_embed"] = model.model.encoder_output_bbox_embed.state_dict()
        else:
            save_dict["reference_points"] = model.model.reference_points.state_dict()
        if model.config.embedding_init_target or not model.config.two_stage:
            save_dict["query_position_embeddings"] = model.model.query_position_embeddings.state_dict()

        save_dict["class_embed"] = model.class_embed.state_dict()
        save_dict["bbox_embed"] = model.bbox_embed.state_dict()

        save_dict["encoder"] = model.model.encoder.state_dict()
        save_dict["decoder"] = model.model.decoder.state_dict()
        save_dict["text_projection"] = model.model.text_projection.state_dict()
        save_dict["text_backbone"] = model.model.text_backbone.state_dict()

        torch.save(save_dict, save_dir_path / subfolder / "torch_artifacts.pth")

    @classmethod
    def get_pytorch_model(cls, *args, **kwargs):
        model = super().get_pytorch_model(*args, **kwargs)
        model.encoder = model.model.encoder
        model.decoder = model.model.decoder
        model.text_backbone = model.model.text_backbone
        model.encoder.config = model.config
        model.decoder.config = model.config
        return model

    @classmethod
    def wrap_model_if_needed(
        cls, model: torch.nn.Module, rbln_config: RBLNGroundingDinoForObjectDetectionConfig
    ) -> torch.nn.Module:
        return model.model.text_projection

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
        model_config: RBLNGroundingDinoForObjectDetectionConfig = None,
        rbln_config: Optional[RBLNGroundingDinoForObjectDetectionConfig] = None,
    ) -> RBLNGroundingDinoForObjectDetectionConfig:
        input_info = [
            (
                "test_features",
                [rbln_config.batch_size, model_config.max_text_len, model_config.text_config.hidden_size],
                "float32",
            ),
        ]

        rbln_config.set_compile_cfgs([RBLNCompileConfig(input_info=input_info)])
        return rbln_config

    def generate_encoder_output_proposals(self, *args, **kwargs):
        return GroundingDinoModel.generate_encoder_output_proposals(self, *args, **kwargs)

    def get_valid_ratio(self, *args, **kwargs):
        return GroundingDinoModel.get_valid_ratio(self, *args, **kwargs)

    def _model_forward(
        self,
        pixel_values: Tensor,
        input_ids: Tensor,
        token_type_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        pixel_mask: Optional[Tensor] = None,
        encoder_outputs=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_self_attention_masks, position_ids = generate_masks_with_special_tokens_and_transfer_map(input_ids)

        if attention_mask is None:
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
            input_ids, text_self_attention_masks.to(torch.long), token_type_ids, position_ids, return_dict=return_dict
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
                mask = nn.functional.interpolate(pixel_mask[None].float(), size=source.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone.position_embedding(source, mask).to(source.dtype)
                feature_maps.append(source)
                masks.append(mask)
                position_embeddings_list.append(pos_l)

        # Create queries
        query_embeds = None
        if self.config.embedding_init_target or self.config.two_stage:
            query_embeds = self.query_position_embeddings.weight

        # Prepare encoder inputs (by flattening)
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
        # If the user passed a tuple for encoder_outputs, we wrap it in a GroundingDinoEncoderOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, GroundingDinoEncoderOutput):
            encoder_outputs = GroundingDinoEncoderOutput(
                last_hidden_state_vision=encoder_outputs[0],
                last_hidden_state_text=encoder_outputs[1],
                vision_hidden_states=encoder_outputs[2] if output_hidden_states else None,
                text_hidden_states=encoder_outputs[3] if output_hidden_states else None,
                attentions=encoder_outputs[-1] if output_attentions else None,
            )

        # Fifth, prepare decoder inputs
        topk_proposals = None
        enc_outputs_class = None
        enc_outputs_coord_logits = None
        encoder_logits = None
        encoder_pred_boxes = None
        if self.config.two_stage:
            object_query_embedding, output_proposals = self.generate_encoder_output_proposals(
                encoder_outputs[0], ~mask_flatten, spatial_shapes
            )

            # hack implementation as in two-stage Deformable DETR
            # apply a detection head to each pixel (A.4 in paper)
            # linear projection for bounding box binary classification (i.e. foreground and background)
            enc_outputs_class = self.encoder_output_class_embed(
                object_query_embedding, encoder_outputs[1], text_token_mask
            )
            # 3-layer FFN to predict bounding boxes coordinates (bbox regression branch)
            delta_bbox = self.encoder_output_bbox_embed(object_query_embedding)
            enc_outputs_coord_logits = delta_bbox + output_proposals

            # only keep top scoring `config.num_queries` proposals
            topk = self.config.num_queries
            topk_logits = enc_outputs_class.max(-1)[0]
            topk_proposals = torch.topk(topk_logits, topk, dim=1)[1]
            topk_coords_logits = torch.gather(
                enc_outputs_coord_logits, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
            )

            topk_coords_logits = topk_coords_logits.detach()
            reference_points = topk_coords_logits.sigmoid()
            init_reference_points = reference_points
            if query_embeds is not None:
                target = query_embeds.unsqueeze(0).repeat(batch_size, 1, 1)
            else:
                target = torch.gather(
                    object_query_embedding, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model)
                ).detach()

            # Set intermediate topk proposals (coords and class) for loss computation
            encoder_pred_boxes = reference_points
            encoder_logits = self.encoder_output_class_embed(target, text_features, text_token_mask)
        else:
            target = query_embeds.unsqueeze(0).repeat(batch_size, 1, 1)
            reference_points = self.reference_points.weight.unsqueeze(0).repeat(batch_size, 1, 1).sigmoid()
            init_reference_points = reference_points

        decoder_outputs = self.decoder(
            inputs_embeds=target,
            vision_encoder_hidden_states=encoder_outputs[0],
            vision_encoder_attention_mask=mask_flatten,
            text_encoder_hidden_states=encoder_outputs[1],
            text_encoder_attention_mask=~text_token_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            self_attn_mask=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            enc_outputs = tuple(
                value
                for value in [
                    enc_outputs_class,
                    enc_outputs_coord_logits,
                    encoder_logits,
                    encoder_pred_boxes,
                ]
                if value is not None
            )
            tuple_outputs = (
                (decoder_outputs[0], init_reference_points) + decoder_outputs[1:] + encoder_outputs + enc_outputs
            )

            return tuple_outputs

        return GroundingDinoModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            init_reference_points=init_reference_points,
            intermediate_hidden_states=decoder_outputs.intermediate_hidden_states,
            intermediate_reference_points=decoder_outputs.intermediate_reference_points,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            encoder_last_hidden_state_vision=encoder_outputs.last_hidden_state_vision,
            encoder_last_hidden_state_text=encoder_outputs.last_hidden_state_text,
            encoder_vision_hidden_states=encoder_outputs.vision_hidden_states,
            encoder_text_hidden_states=encoder_outputs.text_hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            enc_outputs_class=enc_outputs_class,
            enc_outputs_coord_logits=enc_outputs_coord_logits,
            encoder_logits=encoder_logits,
            encoder_pred_boxes=encoder_pred_boxes,
        )

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor,
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        pixel_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Union[GroundingDinoEncoderOutput, Tuple]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: List[Dict[str, Union[torch.LongTensor, torch.FloatTensor]]] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        with torch.inference_mode():
            # First, sent images through Grounding DINO base model to obtain encoder + decoder outputs
            outputs = self._model_forward(
                pixel_values=pixel_values,
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                pixel_mask=pixel_mask,
                encoder_outputs=encoder_outputs,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        idx = 5 + (1 if output_attentions else 0) + (1 if output_hidden_states else 0)
        enc_text_hidden_state = outputs.encoder_last_hidden_state_text if return_dict else outputs[idx]
        hidden_states = outputs.intermediate_hidden_states if return_dict else outputs[2]
        init_reference_points = outputs.init_reference_points if return_dict else outputs[1]
        inter_references_points = outputs.intermediate_reference_points if return_dict else outputs[3]

        # class logits + predicted bounding boxes
        outputs_classes = []
        outputs_coords = []

        # hidden_states are of shape (batch_size, num_stages, height, width)
        # predict class and bounding box deltas for each stage
        num_levels = hidden_states.shape[1]
        for level in range(num_levels):
            if level == 0:
                reference = init_reference_points
            else:
                reference = inter_references_points[:, level - 1]
            reference = torch.special.logit(reference, eps=1e-5)
            outputs_class = self.class_embed[level](
                vision_hidden_state=hidden_states[:, level],
                text_hidden_state=enc_text_hidden_state,
                text_token_mask=attention_mask.bool(),
            )
            delta_bbox = self.bbox_embed[level](hidden_states[:, level])

            reference_coordinates = reference.shape[-1]
            if reference_coordinates == 4:
                outputs_coord_logits = delta_bbox + reference
            elif reference_coordinates == 2:
                delta_bbox[..., :2] += reference
                outputs_coord_logits = delta_bbox
            else:
                raise ValueError(f"reference.shape[-1] should be 4 or 2, but got {reference.shape[-1]}")
            outputs_coord = outputs_coord_logits.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        logits = outputs_class[-1]
        pred_boxes = outputs_coord[-1]

        if not return_dict:
            auxiliary_outputs = []
            output = [logits, pred_boxes, *auxiliary_outputs, *outputs, input_ids]
            output = tuple(out for out in output if out is not None)
            return output

        return GroundingDinoObjectDetectionOutput(
            logits=logits,
            pred_boxes=pred_boxes,
            last_hidden_state=outputs.last_hidden_state,
            # auxiliary_outputs=auxiliary_outputs,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            encoder_last_hidden_state_vision=outputs.encoder_last_hidden_state_vision,
            encoder_last_hidden_state_text=outputs.encoder_last_hidden_state_text,
            encoder_vision_hidden_states=outputs.encoder_vision_hidden_states,
            encoder_text_hidden_states=outputs.encoder_text_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            intermediate_hidden_states=outputs.intermediate_hidden_states,
            intermediate_reference_points=outputs.intermediate_reference_points,
            init_reference_points=outputs.init_reference_points,
            enc_outputs_class=outputs.enc_outputs_class,
            enc_outputs_coord_logits=outputs.enc_outputs_coord_logits,
            encoder_logits=outputs.encoder_logits,
            encoder_pred_boxes=outputs.encoder_pred_boxes,
            input_ids=input_ids,
        )


def _update_spatial_shapes(model_config, rbln_config):
    def down_sampled_size(x, depth: int = 1):
        if depth == 0:
            return x
        return down_sampled_size((x + 1) // 2, depth - 1)

    def num_patches(image_size, patch_size):
        return (image_size + patch_size - 1) // patch_size

    # update spatial_shapes
    spatial_shapes = []
    backbone_config = model_config.backbone_config
    num_patched_h = num_patches(rbln_config.image_height, backbone_config.patch_size)
    num_patched_w = num_patches(rbln_config.image_height, backbone_config.patch_size)
    for out_layer in backbone_config.out_indices:
        spatial_shapes.append(
            [down_sampled_size(num_patched_h, out_layer - 1), down_sampled_size(num_patched_w, out_layer - 1)]
        )

    # Lowest resolution feature maps are obtained via 3x3 stride 2 convolutions on the final stage
    if model_config.num_feature_levels > len(spatial_shapes):
        last_h, last_w = spatial_shapes[-1][0], spatial_shapes[-1][1]
        h_out = (last_h - 1) // 2 + 1
        w_out = (last_w - 1) // 2 + 1
        spatial_shapes.append([h_out, w_out])

    rbln_config.spatial_shapes_list = spatial_shapes

    return rbln_config


class RBLNGroundingDinoEncoder(RBLNModel):
    """
    RBLN optimized CLIP text encoder model.

    This class provides hardware-accelerated inference for CLIP text encoders
    on RBLN devices, supporting text encoding for multimodal tasks.
    """

    def __post_init__(self, **kwargs):
        self.encoder_runtime = RBLNPytorchRuntime(self.model[0])

    @classmethod
    def wrap_model_if_needed(
        cls, model: torch.nn.Module, rbln_config: RBLNGroundingDinoForObjectDetectionConfig
    ) -> torch.nn.Module:
        model = _GroundingDinoEncoder(model, rbln_config).eval()
        return model

    @classmethod
    def _update_submodule_config(
        cls,
        model: "PreTrainedModel",
        rbln_config: RBLNModelConfig,
        preprocessors: Optional[Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"]],
    ):
        if rbln_config.image_size is None:
            longest_edge = None
            for processor in preprocessors:
                if hasattr(processor, "image_processor"):
                    longest_edge = processor.image_processor.size["longest_edge"]
                    break
            rbln_config.image_size = longest_edge

        rbln_config = _update_spatial_shapes(model.config, rbln_config)

        return rbln_config

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"],
        model: Optional["PreTrainedModel"] = None,
        model_config: RBLNGroundingDinoEncoderConfig = None,
        rbln_config: Optional[RBLNGroundingDinoEncoderConfig] = None,
    ) -> RBLNGroundingDinoEncoderConfig:
        if rbln_config.image_size is None:
            raise ValueError("RBLN config must have image_size set for RBLN optimized GroundingDinoDecoder.")

        vision_seq_len = int((rbln_config.spatial_shapes[:, 0] * rbln_config.spatial_shapes[:, 1]).sum())

        input_info = [
            (
                "vision_features",
                [rbln_config.batch_size, vision_seq_len, model_config.d_model],
                "float32",
            ),
            (
                "vision_attention_mask",
                [
                    rbln_config.batch_size,
                    vision_seq_len,
                    model_config.d_model,
                ],
                "float32",
            ),
            (
                "vision_position_embedding",
                [rbln_config.batch_size, vision_seq_len, model_config.d_model],
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
                "reference_points",
                [rbln_config.batch_size, vision_seq_len, 4, 2],
                "float32",
            ),
        ]

        rbln_config.set_compile_cfgs([RBLNCompileConfig(input_info=input_info)])

        return rbln_config

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """
        Get reference points for each feature map.

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
        reference_points_list = []
        for level, (height, width) in enumerate(spatial_shapes):
            ref_y, ref_x = meshgrid(
                torch.linspace(0.5, height - 0.5, height, dtype=torch.float32, device=device),
                torch.linspace(0.5, width - 0.5, width, dtype=torch.float32, device=device),
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
        vision_features: Tensor,
        vision_attention_mask: Tensor,
        vision_position_embedding: Tensor,
        spatial_shapes: Tensor,
        spatial_shapes_list: List[Tuple[int, int]],
        level_start_index: Tensor,
        valid_ratios=None,
        text_features: Optional[Tensor] = None,
        text_attention_mask: Optional[Tensor] = None,
        text_position_embedding: Optional[Tensor] = None,
        text_self_attention_masks: Optional[Tensor] = None,
        text_position_ids: Optional[Tensor] = None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        if output_attentions or output_hidden_states:
            raise NotImplementedError(
                "RBLN optimized GroundingDinoEncoder does not support output_attentions or output_hidden_states."
            )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device="cpu")

        enc_outputs = self.encoder_runtime(
            vision_features=vision_features,
            vision_attention_mask=vision_attention_mask.to(torch.float32)
            .unsqueeze(-1)
            .repeat(1, 1, self.config.d_model),
            vision_position_embedding=vision_position_embedding,
            text_features=text_features,
            text_attention_mask=text_attention_mask.to(torch.float32),
            text_self_attention_masks=text_self_attention_masks.to(torch.float32),
            reference_points=reference_points,
        )

        if not return_dict:
            return tuple(enc_outputs)

        return GroundingDinoEncoderOutput(
            last_hidden_state_vision=enc_outputs[0],
            last_hidden_state_text=enc_outputs[1],
            attentions=tuple(enc_outputs[2:]),
        )


class RBLNGroundingDinoDecoder(RBLNModel):
    """
    RBLN optimized CLIP text encoder model.

    This class provides hardware-accelerated inference for CLIP text encoders
    on RBLN devices, supporting text encoding for multimodal tasks.
    """

    def __post_init__(self, **kwargs):
        self.decoder_runtime = RBLNPytorchRuntime(self.model[0])

    @classmethod
    def wrap_model_if_needed(
        cls, model: torch.nn.Module, rbln_config: RBLNGroundingDinoForObjectDetectionConfig
    ) -> torch.nn.Module:
        return _GroundingDinoDecoder(model, rbln_config).eval()

    @classmethod
    def update_rbln_config_using_pipe(
        cls, pipe: "RBLNDiffusionMixin", rbln_config: "RBLNDiffusionMixinConfig", submodule_name: str
    ) -> "RBLNDiffusionMixinConfig":
        return rbln_config

    @classmethod
    def _update_submodule_config(
        cls,
        model: "PreTrainedModel",
        rbln_config: RBLNModelConfig,
        preprocessors: Optional[Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"]],
    ):
        if rbln_config.image_size is None:
            longest_edge = None
            for processor in preprocessors:
                if hasattr(processor, "image_processor"):
                    longest_edge = processor.image_processor.size["longest_edge"]
                    break
            rbln_config.image_size = longest_edge

        rbln_config = _update_spatial_shapes(model.config, rbln_config)

        return rbln_config

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"],
        model: Optional["PreTrainedModel"] = None,
        model_config: RBLNGroundingDinoDecoderConfig = None,
        rbln_config: Optional[RBLNGroundingDinoEncoderConfig] = None,
    ) -> RBLNGroundingDinoEncoderConfig:
        if rbln_config.image_size is None:
            raise ValueError("RBLN config must have image_size set for RBLN optimized GroundingDinoDecoder.")

        vision_seq_len = int((rbln_config.spatial_shapes[:, 0] * rbln_config.spatial_shapes[:, 1]).sum())

        input_info = [
            (
                "inputs_embeds",
                [rbln_config.batch_size, model_config.num_queries, model_config.d_model],
                "float32",
            ),
            (
                "vision_encoder_hidden_states",
                [
                    rbln_config.batch_size,
                    vision_seq_len,
                    model_config.d_model,
                ],
                "float32",
            ),
            (
                "vision_encoder_attention_mask",
                [rbln_config.batch_size, vision_seq_len, model_config.d_model],
                "float32",
            ),
            (
                "text_encoder_hidden_states",
                [rbln_config.batch_size, model_config.max_text_len, model_config.d_model],
                "float32",
            ),
            (
                "text_encoder_attention_mask",
                [
                    rbln_config.batch_size,
                    model_config.max_text_len,
                ],
                "float32",
            ),
            (
                "reference_points",
                [
                    rbln_config.batch_size,
                    model_config.num_queries,
                    4,
                ],
                "float32",
            ),
            (
                "valid_ratios",
                [
                    rbln_config.batch_size,
                    4,
                    2,
                ],
                "float32",
            ),
        ]

        rbln_config.set_compile_cfgs([RBLNCompileConfig(input_info=input_info)])
        return rbln_config

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
        **kwargs,
    ):
        if output_attentions or output_hidden_states:
            raise NotImplementedError(
                "RBLN optimized GroundingDinoDecoder does not support output_attentions or output_hidden_states."
            )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        reshaped_vision_encoder_attention_mask = (
            vision_encoder_attention_mask[:, :, None].repeat(1, 1, self.config.d_model).to(torch.float32)
        )

        # Forward pass through the decoder
        outputs = self.decoder_runtime(
            inputs_embeds=inputs_embeds,
            vision_encoder_hidden_states=vision_encoder_hidden_states,
            vision_encoder_attention_mask=reshaped_vision_encoder_attention_mask,
            text_encoder_hidden_states=text_encoder_hidden_states,
            text_encoder_attention_mask=text_encoder_attention_mask.to(torch.float32),
            reference_points=reference_points,
            valid_ratios=valid_ratios,
        )

        if not return_dict:
            return outputs

        return GroundingDinoDecoderOutput(
            last_hidden_state=outputs[0],
            intermediate_hidden_states=outputs[1],
            intermediate_reference_points=outputs[2],
            attentions=tuple(outputs[3:]),
        )
