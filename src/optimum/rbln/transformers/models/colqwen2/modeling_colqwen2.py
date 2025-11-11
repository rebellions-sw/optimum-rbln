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

from typing import TYPE_CHECKING, Optional, Union

import torch
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_utils import no_init_weights
from transformers.models.colqwen2.modeling_colqwen2 import ColQwen2ForRetrievalOutput
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLModel,
    Qwen2_5_VLRotaryEmbedding,
)
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VLModel,
    Qwen2VLRotaryEmbedding,
)

from optimum.rbln.transformers.models.decoderonly.modeling_decoderonly import (
    RBLNDecoderOnlyModel,
)

from .configuration_colqwen2 import (
    RBLNColQwen2ForRetrievalConfig,
)


if TYPE_CHECKING:
    from transformers import (
        AutoFeatureExtractor,
        AutoProcessor,
        AutoTokenizer,
        PretrainedConfig,
    )

from .colqwen2_architecture import ColQwen2LanguageModelWrapper


class RBLNColQwen2ForRetrieval(RBLNDecoderOnlyModel):
    """
    The ColQwen Model transformer for document retrieval using vision-language models.
    This model inherits from [`RBLNDecoderOnlyModel`]. Check the superclass documentation for the generic methods the library implements for all its models.

    A class to convert and run pre-trained transformers based `ColQwen2ForRetrieval` model on RBLN devices.
    It implements the methods to convert a pre-trained transformers `ColQwen2ForRetrieval` model into a RBLN transformer model by:

    - transferring the checkpoint weights of the original into an optimized RBLN graph,
    - compiling the resulting graph using the RBLN compiler.

    **Configuration:**
    This model uses [`RBLNColQwen2ForRetrievalConfig`] for configuration. When calling methods like `from_pretrained` or `from_model`,
    the `rbln_config` parameter should be an instance of [`RBLNColQwen2ForRetrievalConfig`] or a dictionary conforming to its structure.

    See the [`RBLNColQwen2ForRetrievalConfig`] class for all available configuration options.

    Examples:
        ```python
        from optimum.rbln import RBLNColQwen2ForRetrieval

        # Using a config dictionary
        rbln_config = {
            "visual": {
                "max_seq_lens": 6400,
            },
            "max_seq_len": 32_768,
            "tensor_parallel_size": 4,
            "device": [0, 1, 2, 3],
            "output_hidden_states": False,
        }
        model = RBLNColQwen2ForRetrieval.from_pretrained(
            "vidore/colqwen2-v1.0-hf",
            export=True,
            rbln_config=rbln_config
        )

        # Using a RBLNColQwen2ForRetrievalConfig instance (recommended for type checking)
        from optimum.rbln import RBLNColQwen2ForRetrievalConfig

        config = RBLNColQwen2ForRetrievalConfig(
            visual={
                "max_seq_lens": 6400,
                "device": 0,
            },
            max_seq_len=32_768,
            tensor_parallel_size=4,
            device=[0, 1, 2, 3],
            output_hidden_states=False,
        )
        model = RBLNColQwen2ForRetrieval.from_pretrained(
            "vidore/colqwen2-v1.0-hf",
            export=True,
            rbln_config=config
        )
        ```
    """

    main_input_name = "inputs_embeds"
    auto_model_class = None
    _rbln_submodules = [
        {"name": "visual"},
    ]
    _decoder_wrapper_cls = ColQwen2LanguageModelWrapper
    _use_rotary_emb = False

    def __post_init__(self, **kwargs):
        self.config = self.config.vlm_config if hasattr(self.config, "vlm_config") else self.config

        artifacts = torch.load(
            self.model_save_dir / self.subfolder / "torch_artifacts.pth",
            weights_only=False,
        )
        self.embed_tokens = self._create_embedding_layer()
        self.embed_tokens.load_state_dict(artifacts["embed_tokens"])
        self.visual = self.rbln_submodules[0]
        self.prefill_runtime = self.model[0]
        self.mrope_section = self.config.text_config.rope_scaling["mrope_section"]
        self.is_colqwen2_5 = "qwen2_5_vl" in self.config.model_type

        if self.is_colqwen2_5:
            self.rotary_emb = Qwen2_5_VLRotaryEmbedding(self.config.text_config)
        else:
            self.rotary_emb = Qwen2VLRotaryEmbedding(self.config.text_config)
        self.block_tables = torch.arange(self.rbln_config.kvcache_num_blocks, dtype=torch.int16)

    @classmethod
    def _reconstruct_model_if_needed(cls, model: "PreTrainedModel"):
        if hasattr(model, "vlm"):
            model.visual = model.vlm.visual
            model.language_model = model.vlm.language_model

        # FIXME: temporary fix for ColQwen2ForRetrieval dtype issue
        return model.to(torch.float32)

    def _create_embedding_layer(self):
        with no_init_weights():
            embed_tokens = torch.nn.Embedding(
                self.config.text_config.vocab_size,
                self.config.text_config.hidden_size,
                self.config.text_config.pad_token_id,
            )
        return embed_tokens

    @classmethod
    def get_input_info(
        cls,
        batch_size: int,
        query_length: int,
        rbln_config: RBLNColQwen2ForRetrievalConfig,
        model_config: PretrainedConfig,
    ):
        text_config = model_config.text_config
        input_info = super().get_input_info(
            batch_size,
            query_length,
            rbln_config=rbln_config,
            model_config=text_config,
        )

        pos_idx = 3
        input_info.insert(
            pos_idx,
            (
                "position_emb",
                [
                    2,
                    batch_size,
                    1,
                    query_length,
                    text_config.hidden_size // text_config.num_attention_heads,
                ],
                rbln_config.torch_dtype,
            ),
        )

        # remove query postion from input_info
        if "query_position" in input_info:
            query_position = input_info.pop(4)
            assert query_position[0] == "query_position", print(query_position[0], "is deleted.")
        return input_info

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Optional[Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"]] = None,
        model: Optional["PreTrainedModel"] = None,
        model_config: Optional["PretrainedConfig"] = None,
        rbln_config: Optional[RBLNColQwen2ForRetrievalConfig] = None,
    ) -> RBLNColQwen2ForRetrievalConfig:
        model_config = model_config.vlm_config if hasattr(model_config, "vlm_config") else model_config
        if rbln_config.output_hidden_states is None:
            rbln_config.output_hidden_states = getattr(model_config.text_config, "output_hidden_states", False)

        return super()._update_rbln_config(
            preprocessors=preprocessors, model=model, model_config=model_config, rbln_config=rbln_config
        )

    def _get_position_embeddings(self, hidden_states, position_ids):
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        mrope_section = self.mrope_section * 2
        cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(1)
        sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(1)
        return torch.stack([cos, sin])

    def get_rope_index(self, *args, **kwargs):
        if self.is_colqwen2_5:
            return Qwen2_5_VLModel.get_rope_index(self, *args, **kwargs)
        else:
            return Qwen2VLModel.get_rope_index(self, *args, **kwargs)

    def _preprocess_visual(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        pixel_values: torch.Tensor = None,
        pixel_values_videos: torch.FloatTensor = None,
        image_grid_thw: torch.LongTensor = None,
        video_grid_thw: torch.LongTensor = None,
        second_per_grid_ts: torch.Tensor = None,
    ):
        batch_size = input_ids.shape[0]
        inputs_embeds = self.embed_tokens(input_ids)

        if pixel_values is not None:
            image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
            n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )

            mask = input_ids == self.config.image_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)

            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(mask_expanded, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
            n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
            n_video_features = video_embeds.shape[0]
            if n_video_tokens != n_video_features:
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                )

            mask = input_ids == self.config.video_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(mask_expanded, video_embeds)

        max_inputs_len = input_ids.shape[1]
        head_dim = self.config.text_config.hidden_size // self.config.text_config.num_attention_heads
        all_position_embeds = torch.zeros(2, batch_size, 1, max_inputs_len, head_dim)
        all_rope_deltas = []

        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        image_idx, video_idx = 0, 0

        for b_idx in range(batch_size):
            input_id = input_ids[b_idx : b_idx + 1][:, attention_mask[b_idx].bool()]
            vision_start_indices = torch.argwhere(input_id == vision_start_token_id).squeeze(1)
            vision_tokens = input_id[0][vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            args = [
                input_id,
                image_grid_thw[image_idx : image_idx + image_nums] if image_grid_thw is not None else None,
                video_grid_thw[video_idx : video_idx + video_nums] if video_grid_thw is not None else None,
            ]
            if self.config.model_type == "qwen2_5_vl":
                args.append(
                    second_per_grid_ts[video_idx : video_idx + video_nums] if second_per_grid_ts is not None else None
                )
            position_ids, rope_deltas = self.get_rope_index(*args)
            image_idx += image_nums
            video_idx += video_nums

            position_embed = self._get_position_embeddings(inputs_embeds, position_ids)
            mask_indices = torch.nonzero(attention_mask[b_idx], as_tuple=True)[0]
            all_position_embeds[:, b_idx : b_idx + 1].index_copy_(dim=-2, index=mask_indices, source=position_embed)
            all_rope_deltas.append(rope_deltas)

        rope_deltas = torch.stack(all_rope_deltas)

        return inputs_embeds, all_position_embeds, rope_deltas

    def _preprocess_chunked_prefill(self, inputs_embeds, attention_mask, position_embed):
        # valid sequence length of inputs_embeds
        query_length = inputs_embeds.shape[1] if attention_mask is None else torch.sum(attention_mask.view(-1)).item()

        # extract valid inputs
        inputs_embeds = inputs_embeds[:, attention_mask.bool()] if attention_mask is not None else inputs_embeds
        position_embed = (
            position_embed[:, :, :, attention_mask.bool(), :] if attention_mask is not None else position_embed
        )

        # add padding for chunked prefill
        padding_size = (
            self.rbln_config.prefill_chunk_size - (query_length % self.rbln_config.prefill_chunk_size)
        ) % self.rbln_config.prefill_chunk_size
        padded_len = query_length + padding_size

        inputs_embeds = torch.nn.functional.pad(inputs_embeds, (0, 0, 0, padding_size))
        position_embed = torch.nn.functional.pad(position_embed, (0, 0, 0, padding_size))
        cache_position = torch.arange(padded_len, dtype=torch.int32).unsqueeze(0)

        return inputs_embeds, position_embed, cache_position, query_length

    def _chunked_prefill_forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embed: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = False,
    ):
        padded_inputs_embeds, padded_position_embed, cache_position, query_length = self._preprocess_chunked_prefill(
            inputs_embeds, attention_mask, position_embed
        )

        # Chunked prefill
        projs = []
        all_hidden_states = [] if output_hidden_states else None
        for step in range(0, query_length, self.rbln_config.prefill_chunk_size):
            # Extract the current chunk of inputs and cache positions
            input_chunk = padded_inputs_embeds[:, step : step + self.rbln_config.prefill_chunk_size]
            cache_pos_chunk = cache_position[:, step : step + self.rbln_config.prefill_chunk_size]
            position_embed_chunk = padded_position_embed[:, :, :, step : step + self.rbln_config.prefill_chunk_size, :]

            # Forward pass for the current chunk
            proj = self.prefill_runtime(
                inputs_embeds=input_chunk,
                cache_position=cache_pos_chunk,
                block_tables=self.block_tables,
                position_emb=position_embed_chunk,
            )

            if output_hidden_states:
                projs.append(proj[0])
                all_hidden_states.append(proj[1:])
            else:
                projs.append(proj)

        projs = torch.concat(projs, dim=-2)[:, :query_length]
        if output_hidden_states:
            # Concatenate chunks for each layer
            concatenated_hidden_states = [
                torch.concat(hs_chunks, dim=-2)[:, :query_length] for hs_chunks in list(zip(*all_hidden_states))
            ]
            all_hidden_states = tuple(concatenated_hidden_states)

        return self._postprocess_chunked_prefill(projs, attention_mask), all_hidden_states

    def _postprocess_chunked_prefill(self, projs, attention_mask):
        # index copy for attention mask
        if attention_mask is not None:
            embedding = torch.full(
                (1, attention_mask.shape[-1], projs.shape[-1]),
                fill_value=1e-10,
                dtype=projs.dtype,
            )
            mask_indices = torch.nonzero(attention_mask, as_tuple=True)[0]
            embedding.index_copy_(dim=-2, index=mask_indices, source=projs)
        else:
            embedding = projs
        return embedding

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ) -> torch.Tensor:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.rbln_config.output_hidden_states
        )

        if output_hidden_states != self.rbln_config.output_hidden_states:
            raise ValueError(
                f"Variable output_hidden_states {output_hidden_states} is not equal to rbln_config.output_hidden_states {self.rbln_config.output_hidden_states} "
                f"Please compile again with the correct argument."
            )

        # Handle the custom "pixel_values" input obtained with `ColQwen2Processor` through unpadding
        if pixel_values is not None and image_grid_thw is not None:
            offsets = image_grid_thw[:, 1] * image_grid_thw[:, 2]  # (batch_size,)
            pixel_values = torch.cat(
                [pixel_sequence[:offset] for pixel_sequence, offset in zip(pixel_values, offsets)],
                dim=0,
            )
        # visual preprocessing
        inputs_embeds, position_embed, _ = self._preprocess_visual(
            input_ids,
            attention_mask,
            pixel_values,
            pixel_values_videos,
            image_grid_thw,
            video_grid_thw,
            second_per_grid_ts,
        )
        batch_size = inputs_embeds.shape[0]

        projs = []
        for b_idx in range(batch_size):
            proj = self._chunked_prefill_forward(
                inputs_embeds[b_idx : b_idx + 1],
                attention_mask[b_idx] if attention_mask is not None else None,
                position_embed[:, b_idx : b_idx + 1],
                output_hidden_states=output_hidden_states,
            )
            projs.append(proj[0])
            all_hidden_states = proj[1] if output_hidden_states else ()

        # postprocess
        projs = torch.cat(projs, dim=0)
        projs = projs / projs.norm(dim=-1, keepdim=True)
        projs = projs * attention_mask.unsqueeze(-1)

        return ColQwen2ForRetrievalOutput(
            embeddings=projs,
            hidden_states=all_hidden_states,
        )
