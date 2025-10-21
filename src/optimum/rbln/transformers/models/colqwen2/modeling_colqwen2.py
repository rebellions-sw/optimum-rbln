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

from typing import TYPE_CHECKING, List, Optional, Union

import rebel
import torch

from transformers.modeling_utils import no_init_weights

from .configuration_colqwen2 import (
    RBLNColQwen2ForRetrievalConfig,
)
from optimum.rbln.configuration_utils import RBLNCompileConfig
from optimum.rbln.transformers.models.decoderonly.modeling_decoderonly import (
    set_default_values,
    validate_attention_method,
    RBLNDecoderOnlyModel
)
from rebel.compile_context import CompileContext
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLRotaryEmbedding,
    Qwen2_5_VLModel,
)
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VLRotaryEmbedding,
    Qwen2VLModel,
)
from transformers.models.colqwen2.modeling_colqwen2 import ColQwen2ForRetrievalOutput

if TYPE_CHECKING:
    from transformers import (
        AutoFeatureExtractor,
        AutoProcessor,
        AutoTokenizer,
        PretrainedConfig,
    )

from .colqwen2_architecture import ColQwen2LanguageModelWrapper

# TODO(si): inherit from RBLNDecoderOnlyModel
class RBLNColQwen2ForRetrieval(RBLNDecoderOnlyModel):
    main_input_name = "inputs_embeds"
    auto_model_class = None
    _rbln_submodules = [
        {"name": "visual"},
    ]
    _rbln_submodule_prefix = "vlm"
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
        
        self.mrope_section = self.config.rope_scaling["mrope_section"]
        if self.config.model_type == "qwen2_vl":
            self.rotary_emb = Qwen2VLRotaryEmbedding(self.config.text_config)
        else :
            self.rotary_emb = Qwen2_5_VLRotaryEmbedding(self.config.text_config)
        self.block_tables = torch.arange(
            self.rbln_config.kvcache_num_blocks, dtype=torch.int16
        )
        
        
    def can_generate(self):
        return False
    
    @classmethod
    def wrap_model_if_needed(cls, model: PreTrainedModel, rbln_config: "RBLNDecoderOnlyModelConfig"):
        return cls._decoder_wrapper_cls(model, rbln_config, cls._use_rotary_emb).eval()
    
    @classmethod
    def get_pytorch_model(cls, *args, **kwargs):
        model = super().get_pytorch_model(*args, **kwargs)
        model.model.lm_head = model.lm_head
        model.lm_head = None
        del model.lm_head
        return model
    
    def _create_embedding_layer(self):
        with no_init_weights():
            embed_tokens = torch.nn.Embedding(
                self.config.text_config.vocab_size,
                self.config.text_config.hidden_size,
                self.config.text_config.pad_token_id,
            )
        return embed_tokens

    @classmethod
    def update_kwargs(cls, kwargs):
        kwargs.update(
            {
                "_attn_implementation": "eager",
            }
        )
        return super().update_kwargs(kwargs)
    
    @classmethod
    def get_input_info(
        cls,
        batch_size: int,
        query_length: int,
        rbln_config: RBLNColQwen2ForRetrievalConfig,
        model_config: PretrainedConfig,
    ):
        input_info = super().get_input_info(
            batch_size,
            query_length,
            rbln_config=rbln_config,
            model_config=model_config.vlm_config.text_config,
        )

        pos_idx = 3
        input_info.insert(
            pos_idx,
            (
                "position_emb",
                [2, batch_size, 1, query_length, model_config.vlm_config.text_config.hidden_size // model_config.vlm_config.text_config.num_attention_heads],
                "float32",
            ),
        )
        
        # remove query postion from input_info
        if "query_position" in input_info:
            query_position = input_info.pop(4)
            assert query_position[0] == "query_position", print(
                query_position[0], "is deleted."
            )
        return input_info

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Optional[
            Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"]
        ] = None,
        model: Optional["PreTrainedModel"] = None,
        model_config: Optional["PretrainedConfig"] = None,
        rbln_config: Optional[RBLNColQwen2ForRetrievalConfig] = None,
    ) -> RBLNColQwen2ForRetrievalConfig:

        if rbln_config.max_seq_len is None:
            rbln_config.max_seq_len = getattr(
                model_config, "max_position_embeddings", None
            ) or getattr(model_config, "n_positions", None)

        if rbln_config.max_seq_len is None:
            raise ValueError("`max_seq_len` should be specified.")

        (
            rbln_config.attn_impl,
            rbln_config.kvcache_partition_len,
            rbln_config.kvcache_block_size,
        ) = set_default_values(
            attn_impl=rbln_config.attn_impl,
            kvcache_partition_len=rbln_config.kvcache_partition_len,
            kvcache_block_size=rbln_config.kvcache_block_size,
            max_seq_len=rbln_config.max_seq_len,
        )

        validate_attention_method(
            attn_impl=rbln_config.attn_impl,
            kvcache_partition_len=rbln_config.kvcache_partition_len,
            kvcache_block_size=rbln_config.kvcache_block_size,
            max_seq_len=rbln_config.max_seq_len,
        )

        required_num_blocks = rbln_config.max_seq_len // rbln_config.kvcache_block_size
        max_num_blocks = required_num_blocks

        if rbln_config.kvcache_num_blocks is None:
            rbln_config.kvcache_num_blocks = max_num_blocks

        input_info = cls.get_input_info(
            batch_size=1,
            query_length=rbln_config.prefill_chunk_size,
            rbln_config=rbln_config,
            model_config=model_config,
        )
        
        if rbln_config.output_hidden_states is None:
            rbln_config.output_hidden_states = model_config.vlm_config.text_config.output_hidden_states

        prefill_compile_config = RBLNCompileConfig(
            compiled_model_name="prefill", input_info=input_info
        )
        rbln_config.set_compile_cfgs([prefill_compile_config])

        return rbln_config

    @classmethod
    def _create_runtimes(
        cls,
        compiled_models: List[rebel.RBLNCompiledModel],
        rbln_config: RBLNColQwen2ForRetrievalConfig,
    ) -> List[rebel.Runtime]:
        expected_model_names = ["prefill"]
        if any(
            model_name not in rbln_config.device_map
            for model_name in expected_model_names
        ):
            cls._raise_missing_compiled_file_error(expected_model_names)

        return [
            rebel.Runtime(
                compiled_models[0],
                tensor_type="pt",
                device=rbln_config.device_map["prefill"],
                activate_profiler=rbln_config.activate_profiler,
            ),
        ]

    def _get_position_embeddings(self, hidden_states, position_ids):
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        mrope_section = self.mrope_section * 2
        cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(1)
        sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(1)
        return torch.stack([cos, sin])
    
    def get_rope_index(self, *args, **kwargs):
        if self.config.model_type == "qwen2_vl":
            return Qwen2VLModel.get_rope_index(self, *args, **kwargs)
        else :
            return Qwen2_5_VLModel.get_rope_index(self, *args, **kwargs)

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

        head_dim = getattr(self.config, "head_dim", None) or self.config.hidden_size // self.config.num_attention_heads
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
            position_ids, rope_deltas = self.get_rope_index(
                # self,
                input_id,
                image_grid_thw[image_idx : image_idx + image_nums] if image_grid_thw is not None else None,
                video_grid_thw[video_idx : video_idx + video_nums] if video_grid_thw is not None else None,
                # second_per_grid_ts[video_idx : video_idx + video_nums] if second_per_grid_ts is not None else None,
            )
            image_idx += image_nums
            video_idx += video_nums

            position_embed = self._get_position_embeddings(inputs_embeds, position_ids)
            mask_indices = torch.nonzero(attention_mask[b_idx], as_tuple=True)[0]
            all_position_embeds[:, b_idx : b_idx + 1].index_copy_(dim=-2, index=mask_indices, source=position_embed)
            all_rope_deltas.append(rope_deltas)

        rope_deltas = torch.stack(all_rope_deltas)

        return inputs_embeds, all_position_embeds, rope_deltas

    def _preprocess_chunked_prefill(
        self, inputs_embeds, attention_mask, position_embed
    ):
        # valid sequence length of inputs_embeds
        query_length = (
            inputs_embeds.shape[1]
            if attention_mask is None
            else torch.sum(attention_mask.view(-1)).item()
        )

        # extract valid inputs
        inputs_embeds = (
            inputs_embeds[:, attention_mask.bool()]
            if attention_mask is not None
            else inputs_embeds
        )
        position_embed = (
            position_embed[:, :, :, attention_mask.bool(), :]
            if attention_mask is not None
            else position_embed
        )

        # add padding for chunked prefill
        padding_size = (
            self.rbln_config.prefill_chunk_size
            - (query_length % self.rbln_config.prefill_chunk_size)
        ) % self.rbln_config.prefill_chunk_size
        padded_len = query_length + padding_size

        inputs_embeds = torch.nn.functional.pad(inputs_embeds, (0, 0, 0, padding_size))
        position_embed = torch.nn.functional.pad(
            position_embed, (0, 0, 0, padding_size)
        )
        cache_position = torch.arange(padded_len, dtype=torch.int32).unsqueeze(0)

        return inputs_embeds, position_embed, cache_position, query_length

    def _chunked_prefill_forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embed: Optional[torch.Tensor] = None,
    ):
        padded_inputs_embeds, padded_position_embed, cache_position, query_length = (
            self._preprocess_chunked_prefill(
                inputs_embeds, attention_mask, position_embed
            )
        )

        # chunked prefill
        projs = []
        all_hidden_states = [] if self.rbln_config.output_hidden_states else None
        for step in range(0, query_length, self.rbln_config.prefill_chunk_size):
            # Extract the current chunk of inputs and cache positions
            input_chunk = padded_inputs_embeds[
                :, step : step + self.rbln_config.prefill_chunk_size
            ]
            cache_pos_chunk = cache_position[
                :, step : step + self.rbln_config.prefill_chunk_size
            ]
            position_embed_chunk = padded_position_embed[
                :, :, :, step : step + self.rbln_config.prefill_chunk_size, :
            ]

            # Forward pass for the current chunk
            proj = self.prefill_runtime(
                inputs_embeds=input_chunk,
                cache_position=cache_pos_chunk,
                block_tables=self.block_tables,
                position_emb=position_embed_chunk,
            )
            projs.append(proj[0])
            if self.rbln_config.output_hidden_states:
                all_hidden_states.append(proj[1:])
        projs = torch.concat(projs, dim=-2)[:, :query_length]
        if self.rbln_config.output_hidden_states:
            # Concatenate chunks for each layer
            concatenated_hidden_states = [
                torch.concat(hs_chunks, dim=-2)[:, :query_length]
                for hs_chunks in list(zip(*all_hidden_states))
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
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ) -> torch.Tensor:
        output_attentions = output_attentions if output_attentions is not None else self.rbln_config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.rbln_config.output_hidden_states
        )

        if output_attentions != self.rbln_config.output_attentions:
            raise ValueError(
                f"Variable output_attentions {output_attentions} is not equal to rbln_config.output_attentions {self.rbln_config.output_attentions} "
                f"Please compile again with the correct argument."
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
                [
                    pixel_sequence[:offset]
                    for pixel_sequence, offset in zip(pixel_values, offsets)
                ],
                dim=0,
            )

        # Preprocess of Qwen2VLForConditionalGeneration
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
            )
            projs.append(proj[0])
            all_hidden_states = proj[1] if output_hidden_states else ()

        # post process
        projs = torch.cat(projs, dim=0)
        projs = projs / projs.norm(
            dim=-1, keepdim=True
        )  # (batch_size, sequence_length, dim)
        projs = projs * attention_mask.unsqueeze(
            -1
        )  # (batch_size, sequence_length, dim)

        # return projs
        return ColQwen2ForRetrievalOutput(
            embeddings=projs,
            # past_key_values=vlm_output.past_key_values,
            hidden_states=all_hidden_states,
            # attentions=vlm_output.attentions, # need to attentions?
        )
    
    # def _prepare_output(self, output, return_dict):
    #     return ColQwen2ForRetrievalOutput(
    #         embeddings=embeddings,
    #         # past_key_values=vlm_output.past_key_values,
    #         hidden_states=vlm_hidden_states,
    #         # attentions=vlm_output.attentions, # need to attentions?
    #     )
