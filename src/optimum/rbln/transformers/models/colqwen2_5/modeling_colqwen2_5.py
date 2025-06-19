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
from collections import deque
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple, Union

import rebel
import torch
from rebel.compile_context import CompileContext
from transformers import (
    AutoModelForVision2Seq,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_utils import no_init_weights
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLRotaryEmbedding,
)

from ....configuration_utils import RBLNCompileConfig
from ....modeling import RBLNModel
from ....utils.logging import get_logger
from ..decoderonly.modeling_decoderonly import RBLNDecoderOnlyModelForCausalLM, RBLNDecoderOnlyOutput, set_default_values, validate_attention_method, RBLNRuntimeModel
from .colqwen2_5_architecture import ColQwen2_5_LanguageModelWrapper
from .configuration_colqwen2_5 import RBLNColQwen2_5ForConditionalGenerationConfig

logger = get_logger(__name__)

if TYPE_CHECKING:
    from transformers import (
        AutoFeatureExtractor,
        AutoProcessor,
        AutoTokenizer,
        PretrainedConfig,
    )

class RBLNColQwen2_5ForConditionalGeneration(RBLNDecoderOnlyModelForCausalLM):
    auto_model_class = AutoModelForVision2Seq
    _rbln_submodules = [
        {"name": "visual"},
    ]
    _decoder_wrapper_cls = ColQwen2_5_LanguageModelWrapper
    _use_rotary_emb = False

    def __post_init__(self, **kwargs):
        main_input_name = self.main_input_name

        if self.rbln_config.use_inputs_embeds:
            main_input_name = "inputs_embeds"
            artifacts = torch.load(self.model_save_dir / self.subfolder / "torch_artifacts.pth", weights_only=False)
            self.embed_tokens = self._create_embedding_layer()
            self.embed_tokens.load_state_dict(artifacts["embed_tokens"])
        else:
            self.embed_tokens = None

        # # Initialize shared resources to be used across Runtime instances (prefill and decode phases)
        # dec_attn_mask = torch.zeros(
        #     self.rbln_config.batch_size, 1, 1, self.rbln_config.max_seq_len, dtype=torch.float32
        # )
        # block_tables = torch.zeros(
        #     self.rbln_config.batch_size,
        #     self.rbln_config.max_seq_len // self.rbln_config.kvcache_block_size,
        #     dtype=torch.int16,
        # ).fill_(-1)
        # free_block_pool = deque(x for x in range(self.rbln_config.kvcache_num_blocks))

        # # TODO delete RBLNRuntimeModel 
        # self.prefill_decoder = RBLNRuntimeModel(
        #     runtime=self.model[0],
        #     main_input_name=main_input_name,
        #     embed_tokens=self.embed_tokens,
        #     phase="prefill",
        #     batch_size=self.rbln_config.batch_size,
        #     dec_attn_mask=dec_attn_mask,
        #     block_tables=block_tables,
        #     free_block_pool=free_block_pool,
        #     kvcache_block_size=self.rbln_config.kvcache_block_size,
        #     vocab_size=self.config.vocab_size,
        #     prefill_chunk_size=self.rbln_config.prefill_chunk_size,
        #     max_seq_len=self.rbln_config.max_seq_len,
        #     use_attention_mask=self.rbln_config.use_attention_mask,
        #     attn_impl=self.rbln_config.attn_impl,
        #     use_position_ids=self.rbln_config.use_position_ids,
        # )
        
        self.visual = self.rbln_submodules[0]
        self.mrope_section = self.config.rope_scaling["mrope_section"]
        self.rotary_emb = Qwen2_5_VLRotaryEmbedding(self.config)
        self.rope_deltas = torch.zeros(self.rbln_config.batch_size)
        self.mask_non_image_embeddings = kwargs.get("mask_non_image_embeddings", False)
        
        artifacts = torch.load(self.model_save_dir / self.subfolder / "torch_artifacts.pth", weights_only=False)
        self.custom_text_proj = self._create_custom_proj_layer()
        self.custom_text_proj.load_state_dict(artifacts["custom_text_proj"])

    @classmethod
    def save_torch_artifacts(
        cls,
        model: "PreTrainedModel",
        save_dir_path: Path,
        subfolder: str,
        rbln_config: RBLNColQwen2_5ForConditionalGenerationConfig,
    ):
        if rbln_config.use_inputs_embeds:
            save_dict = {}
            save_dict["embed_tokens"] = model.get_input_embeddings().state_dict()
            from collections import OrderedDict
            save_dict["custom_text_proj"] = OrderedDict({'weight' : model.custom_text_proj.state_dict()['base_layer.weight']})
            torch.save(save_dict, save_dir_path / subfolder / "torch_artifacts.pth")

    def _create_custom_proj_layer(self):
        with no_init_weights():
            custom_text_proj = torch.nn.Linear(
                in_features=self.config.hidden_size,
                out_features=128, # TODO(si) make generalize
                bias=False
            )
        return custom_text_proj
    
    @classmethod
    def update_kwargs(cls, kwargs):
        kwargs.update(
            {
                "_attn_implementation": "eager",
            }
        )
        return super().update_kwargs(kwargs)

    @classmethod
    @torch.inference_mode()
    def get_compiled_model(cls, model: "PreTrainedModel", rbln_config: RBLNColQwen2_5ForConditionalGenerationConfig):
        wrapped_model = cls.wrap_model_if_needed(model, rbln_config)

        rbln_compile_configs = rbln_config.compile_cfgs
        prefill_compile_config = rbln_compile_configs[0]

        context = CompileContext(use_weight_sharing=False)

        # Here we use meta tensor, for the memory efficiency.
        meta_tensor_names = [name for name, _, _ in prefill_compile_config.input_info if "past_key_values" in name]
        prefill_example_inputs = prefill_compile_config.get_dummy_inputs(fill=0, meta_tensor_names=meta_tensor_names)

        # Mark static tensors (self kv states)
        static_tensors = {}
        for (name, _, _), tensor in zip(prefill_compile_config.input_info, prefill_example_inputs):
            if "past_key_values" in name:
                static_tensors[name] = tensor
                context.mark_static_address(tensor)

        def compile_model(wrapped_model, compile_config, example_inputs, compile_context, quantization):
            try:
                if quantization:
                    quantization.maybe_set_quantization_env()
                original_linear = torch.nn.functional.linear
                torch.nn.functional.linear = torch.ops.rbln_custom_ops.linear
                compiled_model = RBLNModel.compile(
                    wrapped_model,
                    compile_config,
                    example_inputs=example_inputs,
                    compile_context=compile_context,
                )
                return compiled_model
            finally:
                torch.nn.functional.linear = original_linear
                if quantization:
                    quantization.maybe_reset_quantization_env()

        wrapped_model.phase = "prefill"
        compiled_prefill = compile_model(
            wrapped_model, prefill_compile_config, prefill_example_inputs, context, rbln_config.quantization
        )

        compiled_models = {"prefill": compiled_prefill}
        
        required_num_blocks = (rbln_config.max_seq_len // rbln_config.kvcache_block_size) * rbln_config.batch_size
        if rbln_config.kvcache_num_blocks < required_num_blocks:
            cls.maybe_suggest_kvcache_num_blocks(
                compiled_models=compiled_models,
                model_config=model.config,
                rbln_config=rbln_config,
            )

        return compiled_models

    @classmethod
    def get_input_info(
        cls,
        batch_size: int,
        query_length: int,
        rbln_config: RBLNColQwen2_5ForConditionalGenerationConfig,
        model_config: PretrainedConfig,
    ):
        num_attention_heads = getattr(model_config, "n_head", None) or getattr(model_config, "num_attention_heads")
        num_key_value_heads = getattr(model_config, "num_key_value_heads", None) or num_attention_heads
        num_hidden_layers = getattr(model_config, "n_layer", None) or getattr(model_config, "num_hidden_layers")
        hidden_size = getattr(model_config, "n_embd", None) or getattr(model_config, "hidden_size")
        head_dim = getattr(model_config, "head_dim", None) or hidden_size // num_attention_heads

        if rbln_config.use_inputs_embeds:
            main_input = ("inputs_embeds", [batch_size, query_length, hidden_size], "float32")
        else:
            main_input = ("input_ids", [batch_size, query_length], "int64")

        input_info = [
            main_input,
            (
                "cache_position",
                [batch_size, query_length],
                "int32",
            ),
        ]

        max_block_cnt = rbln_config.max_seq_len // rbln_config.kvcache_block_size

        if query_length > 1:
            input_info.extend([("block_tables", [max_block_cnt], "int16")])
        else:
            input_info.extend([("block_tables", [batch_size, max_block_cnt], "int16")])

        if query_length > 1:
            input_info.extend(
                [
                    ("query_position", [], "int16"),
                ]
            )
        if rbln_config.use_attention_mask:
            input_info.extend(
                [
                    ("attention_mask", [batch_size, 1, query_length, rbln_config.max_seq_len], "float32"),
                ]
            )
        if rbln_config.use_position_ids:
            input_info.append(("position_ids", [batch_size, query_length], "int32"))

        input_info.extend(
            [
                (
                    f"past_key_values_{i}",
                    [
                        rbln_config.kvcache_num_blocks,
                        num_key_value_heads,
                        rbln_config.kvcache_block_size,
                        head_dim,
                    ],
                    "float32",
                )
                for i in range(num_hidden_layers * 2)
            ]
        )
        
        pos_idx = 3
        input_info.insert(
            pos_idx,
            (
                "position_emb",
                [2, batch_size, 1, query_length, model_config.hidden_size // model_config.num_attention_heads],
                "float32",
            ),
        )
        query_position = input_info.pop(pos_idx+1) # remove query postion
        assert query_position[0] == "query_position", print(query_position[0], "is deleted.")

        return input_info
    
    # @classmethod
    # def get_input_info(
    #     cls,
    #     batch_size: int,
    #     query_length: int,
    #     rbln_config: RBLNColQwen2_5ForConditionalGenerationConfig,
    #     model_config: PretrainedConfig,
    # ):
    #     input_info = super().get_input_info(
    #         batch_size,
    #         query_length,
    #         rbln_config=rbln_config,
    #         model_config=model_config,
    #     )
    #     pos_idx = 3
    #     input_info.insert(
    #         pos_idx,
    #         (
    #             "position_emb",
    #             [2, batch_size, 1, query_length, model_config.hidden_size // model_config.num_attention_heads],
    #             "float32",
    #         ),
    #     )
    #     query_position = input_info.pop(pos_idx+1) # remove query postion
    #     assert query_position[0] == "query_position", print(query_position[0], "is deleted.")
    #     return input_info

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Optional[Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"]] = None,
        model: Optional["PreTrainedModel"] = None,
        model_config: Optional["PretrainedConfig"] = None,
        rbln_config: Optional[RBLNColQwen2_5ForConditionalGenerationConfig] = None,
    ) -> RBLNColQwen2_5ForConditionalGenerationConfig:
        if rbln_config.max_seq_len is None:
            rbln_config.max_seq_len = getattr(model_config, "max_position_embeddings", None) or getattr(
                model_config, "n_positions", None
            )
        if rbln_config.max_seq_len is None:
            raise ValueError("`max_seq_len` should be specified.")

        rbln_config.attn_impl, rbln_config.kvcache_partition_len, rbln_config.kvcache_block_size = set_default_values(
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

        required_num_blocks = (rbln_config.max_seq_len // rbln_config.kvcache_block_size) * rbln_config.batch_size
        max_num_blocks = required_num_blocks

        if rbln_config.kvcache_num_blocks is None:
            rbln_config.kvcache_num_blocks = max_num_blocks
        elif rbln_config.kvcache_num_blocks > max_num_blocks:
            logger.warning(
                f"The set `kvcache_num_blocks` ({rbln_config.kvcache_num_blocks}) is greater"
                f" than the estimated maximum number of blocks ({max_num_blocks})."
                "This can cause a failure during model compilation."
            )
        logger.info(f"[KVCache] Compiling with num_blocks: {rbln_config.kvcache_num_blocks}")
        num_attention_heads = getattr(model_config, "n_head", None) or getattr(model_config, "num_attention_heads")
        num_key_value_heads = getattr(model_config, "num_key_value_heads", None) or num_attention_heads
        num_hidden_layers = getattr(model_config, "n_layer", None) or getattr(model_config, "num_hidden_layers")
        hidden_size = getattr(model_config, "n_embd", None) or getattr(model_config, "hidden_size")
        head_dim = getattr(model_config, "head_dim", None) or hidden_size // num_attention_heads

        prefill_input_info = cls.get_input_info(
            batch_size=1,
            query_length=rbln_config.max_seq_len,
            rbln_config=rbln_config,
            model_config=model_config
        )

        prefill_compile_config = RBLNCompileConfig(compiled_model_name="prefill", input_info=prefill_input_info)
        rbln_config.set_compile_cfgs([prefill_compile_config])

        return rbln_config
    
    @classmethod
    def _create_runtimes(
        cls,
        compiled_models: List[rebel.RBLNCompiledModel],
        rbln_config: RBLNColQwen2_5ForConditionalGenerationConfig,
    ) -> List[rebel.Runtime]:
        expected_model_names = ["prefill"]
        if any(model_name not in rbln_config.device_map for model_name in expected_model_names):
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

    def _preprocess_prefill(
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
        
        if pixel_values is not None and image_grid_thw is not None:
            offsets = image_grid_thw[:, 1] * image_grid_thw[:, 2]  # (batch_size,)
            pixel_values = torch.cat(
                [pixel_sequence[:offset] for pixel_sequence, offset in zip(pixel_values, offsets)],
                dim=0,
            )

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
                input_id,
                image_grid_thw[image_idx : image_idx + image_nums] if image_grid_thw is not None else None,
                video_grid_thw[video_idx : video_idx + video_nums] if video_grid_thw is not None else None,
                second_per_grid_ts[video_idx : video_idx + video_nums] if second_per_grid_ts is not None else None,
            )
            image_idx += image_nums
            video_idx += video_nums

            position_embed = self._get_position_embeddings(inputs_embeds, position_ids)
            mask_indices = torch.nonzero(attention_mask[b_idx], as_tuple=True)[0]
            all_position_embeds[:, b_idx : b_idx + 1].index_copy_(dim=-2, index=mask_indices, source=position_embed)
            all_rope_deltas.append(rope_deltas)

        rope_deltas = torch.stack(all_rope_deltas)

        return inputs_embeds, all_position_embeds, rope_deltas

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embedding for text part.
            Examples:
                Temporal (Time): 3 patches, representing different segments of the video in time.
                Height: 2 patches, dividing each frame vertically.
                Width: 2 patches, dividing each frame horizontally.
                We also have some important parameters:
                fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
                tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
                temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
                interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [101, 102, 103, 104, 105]
                text height position_ids: [101, 102, 103, 104, 105]
                text width position_ids: [101, 102, 103, 104, 105]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
                The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        second_per_grid_t = 0
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image

                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        if second_per_grid_ts is not None:
                            second_per_grid_t = second_per_grid_ts[video_index]
                        else:
                            second_per_grid_t = 1.0
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                    expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                    time_tensor = expanded_range * second_per_grid_t * self.config.vision_config.tokens_per_second

                    time_tensor_long = time_tensor.long()
                    t_index = time_tensor_long.flatten()

                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

    def _prepare_prefill_inputs(
        self,
        inputs: torch.Tensor,
        cache_position: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embed: Optional[torch.Tensor] = None,
    ):
        """
        Prepare inputs for prefill phase.
        """
        # Handle continuous batching in a compiled graph by extracting valid inputs
        # If an attention mask is provided, select only the valid (non-masked) inputs
        inputs = inputs[:, attention_mask.bool()] if attention_mask is not None else inputs
        if position_embed is not None:
            position_embed = (
                position_embed[:, :, :, attention_mask.bool(), :] if attention_mask is not None else position_embed
            )

        query_length = inputs.shape[1]
        if query_length > self.rbln_config.max_seq_len:
            raise ValueError(
                f"Input length ({query_length}) exceeds the maximum allowed sequence length ({self.rbln_config.max_seq_len})."
            )

        # Initialize attention mask for chunked processing # NOTE never used
        # chunked_attention_mask = (
        #     torch.zeros(1, 1, self.rbln_config.prefill_chunk_size, self.rbln_config.max_seq_len, dtype=torch.float32)
        #     if self.rbln_config.use_attention_mask
        #     else None
        # )
        
        # left padding -> right padding # TODO(si) : make generalize
        fliped_attention_mask = torch.flip(attention_mask, dims=[-1]).unsqueeze(0)


        # Pad input and cache_position if the last chunk is smaller than `prefill_chunk_size`
        # if query_length % self.prefill_chunk_size != 0: # chunked prefill 써야하냐? 안써도 될 것 같은데 -> chunk 는 Default
            # padding_size = (self.prefill_chunk_size - query_length) % self.prefill_chunk_size
        if query_length % self.rbln_config.max_seq_len != 0: # chunked prefill 써야하냐? 안써도 될 것 같은데 -> chunk 는 Default
            padding_size = (self.rbln_config.max_seq_len - query_length) % self.rbln_config.max_seq_len

            # inputs_embeds
            if inputs.dim() == 3:
                inputs = torch.nn.functional.pad(inputs, (0, 0, 0, padding_size))
            # inputs_ids
            else:
                inputs = torch.nn.functional.pad(inputs, (0, padding_size))

            cache_position = torch.arange(
                        0,
                        query_length + padding_size,
                        dtype=torch.int32,
                    ).unsqueeze(0)

            if position_embed is not None:
                position_embed = torch.nn.functional.pad(position_embed, (0, 0, 0, padding_size))

        # Overwrite position_ids and padded_cache_lengths
        position_ids = None
        padded_cache_lengths = 0

        return (
            inputs,
            cache_position,
            # chunked_attention_mask,
            fliped_attention_mask,
            position_ids,
            position_embed,
            padded_cache_lengths,
            query_length,
        )
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        generate_idx: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> RBLNDecoderOnlyOutput:
        # Prefill

        inputs_embeds, position_embed, rope_deltas = self._preprocess_prefill(
            input_ids,
            attention_mask,
            pixel_values,
            pixel_values_videos,
            image_grid_thw,
            video_grid_thw,
            second_per_grid_ts,
        )
        # import pdb; pdb.set_trace()
        self.rope_deltas = rope_deltas
        batch_size = inputs_embeds.shape[0]

        projs = []
        attention_mask_batches = []
        for b_idx in range(batch_size):
            cache_position = torch.arange(0, inputs_embeds.shape[1], dtype=torch.int32).unsqueeze(0)
            # import pdb; pdb.set_trace()

            # outputs = self.prefill_decoder(
            #     inputs_embeds=inputs_embeds[b_idx : b_idx + 1],
            #     attention_mask=attention_mask[b_idx] if attention_mask is not None else None,
            #     cache_position=cache_position,
            #     batch_idx=b_idx,
            #     position_embed=position_embed[:, b_idx : b_idx + 1],
            # )
            
            (
                inputs_embeds_batch,
                cache_position_batch,
                attention_mask_batch,
                _,
                position_embed_batch,
                _,
                _,
            ) = self._prepare_prefill_inputs(
                inputs_embeds[b_idx : b_idx + 1], cache_position, attention_mask[b_idx], position_embed[:, b_idx : b_idx + 1],
            )
            
            # import pdb; pdb.set_trace()
            outputs = self.model[0](
                inputs_embeds=inputs_embeds_batch, 
                cache_position=cache_position_batch, 
                block_tables=torch.tensor([0], dtype=torch.int16),
                position_emb=position_embed_batch
                                    )

            proj = self.custom_text_proj(outputs) # TODO(si) RSD pattern 추가
            projs.append(proj)
            attention_mask_batches.append(attention_mask_batch)
            # import pdb; pdb.set_trace()
            
        projs = torch.cat(projs, dim=0)
        projs = projs[:, :inputs_embeds.shape[1]]
        
        attention_mask_batches = torch.cat(attention_mask_batches, dim=0)
        
        # import pdb; pdb.set_trace()
        
        projs = projs / projs.norm(dim=-1, keepdim=True)  # (batch_size, sequence_length, dim)
        # projs = projs * attention_mask.unsqueeze(-1)  # (batch_size, sequence_length, dim)
        projs = projs * attention_mask_batches.unsqueeze(-1)  # (batch_size, sequence_length, dim)
        # import pdb; pdb.set_trace()
        if pixel_values is not None and self.mask_non_image_embeddings:
            # Pools only the image embeddings
            image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1)
            projs = projs * image_mask
        return projs
