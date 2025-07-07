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
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import rebel
import torch
from rebel.compile_context import CompileContext
from transformers import (
    AutoModelForImageTextToText,
    Gemma3ForConditionalGeneration,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.modeling_utils import no_init_weights
from transformers.models.gemma3.modeling_gemma3 import Gemma3TextScaledWordEmbedding

from ....configuration_utils import RBLNCompileConfig, RBLNModelConfig
from ....modeling import RBLNModel
from ....utils.logging import get_logger
from ..decoderonly.modeling_decoderonly import RBLNDecoderOnlyModelForCausalLM, RBLNDecoderOnlyOutput, RBLNRuntimeModel
from .configuration_gemma3 import RBLNGemma3ForCausalLMConfig
from .gemma3_architecture import Gemma3ForCausalLMWrapper


logger = get_logger()


if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer, Gemma3ForConditionalGeneration


@dataclass
class RBLNGemma3ForCausalLMOutput(RBLNDecoderOnlyOutput):
    attention_mask: Optional[torch.Tensor] = None


class LoopVisionTower:
    def __init__(self, vision_tower: RBLNModel) -> None:
        self.vision_tower = vision_tower

    def forward(self, *args, **kwargs):
        # Loop instead of batch
        # shape of pixel_values : [batch, num_channel, height, width]
        pixel_values = args[0]

        batch_size = pixel_values.shape[0]
        outputs = []
        for i in range(batch_size):
            outputs.append(self.vision_tower(pixel_values=pixel_values[i : i + 1], return_dict=True))

        last_hidden_states = [output.last_hidden_state for output in outputs]

        # FIXME:: This can be optimized using out= API of rbln runtime.
        last_hidden_states = torch.cat(last_hidden_states, dim=0)

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_states,
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

        batch_size = image_feature.shape[0]
        outputs = []
        for i in range(batch_size):
            outputs.append(self.multi_modal_projector(image_feature[i : i + 1]))

        # FIXME:: This can be optimized using out= API of rbln runtime.
        outputs = torch.cat(outputs, dim=0)
        return outputs

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

    def __repr__(self) -> str:
        return repr(self.multi_modal_projector)


class RBLNGemma3ForConditionalGeneration(RBLNModel):
    auto_model_class = AutoModelForImageTextToText
    _rbln_submodules = [
        {"name": "vision_tower"},
        {"name": "language_model"},
    ]

    def __getattr__(self, __name: str) -> Any:
        def redirect(func):
            return lambda *pargs, **kwargs: func(self, *pargs, **kwargs)

        val = getattr(Gemma3ForConditionalGeneration, __name)

        if isinstance(val, Callable) and "self" in set(inspect.signature(val).parameters):
            return redirect(val)
        return val

    def can_generate(self):
        return True

    def __post_init__(self, **kwargs):
        self.vision_tower = LoopVisionTower(self.rbln_submodules[0])
        self.language_model = self.rbln_submodules[1]
        self.multi_modal_projector = LoopProjector(self.model[0])
        self.vocab_size = self.config.text_config.vocab_size

        # Copied from the original class
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
        image_feature_dim = (model_config.vision_config.image_size // model_config.vision_config.patch_size) ** 2
        feature_size = model_config.vision_config.hidden_size

        input_info = [("image_features", [rbln_config.batch_size, image_feature_dim, feature_size], "float32")]
        rbln_compile_config = RBLNCompileConfig(input_info=input_info)
        rbln_config.set_compile_cfgs([rbln_compile_config])
        return rbln_config

    def prepare_inputs_for_generation(
        self,
        input_ids,
        inputs_embeds=None,
        pixel_values=None,
        image_sizes=None,
        attention_mask=None,
        generate_idx=None,
        padded_cache_lengths=None,
        token_type_ids=None,
        **kwargs,
    ):
        # Prepare HF generation
        is_prefill_phase = generate_idx is None

        model_inputs = self.language_model.prepare_inputs_for_generation(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            generate_idx=generate_idx,  # Not affect
            attention_mask=attention_mask,
            padded_cache_lengths=padded_cache_lengths,
            **kwargs,
        )

        if is_prefill_phase:
            model_inputs.update(
                {
                    "pixel_values": pixel_values,
                    "image_sizes": image_sizes,
                    "token_type_ids": token_type_ids,
                }
            )

        model_inputs["attention_mask"] = attention_mask

        return model_inputs

    def _update_model_kwargs_for_generation(
        self,
        outputs: RBLNDecoderOnlyOutput,
        model_kwargs: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, Any]:
        # update generate_idx
        model_kwargs["generate_idx"] = outputs.generate_idx
        model_kwargs["padded_cache_lengths"] = outputs.padded_cache_lengths

        return model_kwargs

    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Projects the last hidden state from the vision model into language model space.

        Args:
            pixel_values: (`torch.FloatTensor` of shape `(batch_size, channels, height, width)`)
                The tensors corresponding to the input images.

        Returns:
            Image feature tensor of shape `(num_images, image_length, embed_dim)`.
        """
        vision_outputs = self.vision_tower(pixel_values).last_hidden_state
        image_features = self.multi_modal_projector(vision_outputs)
        return image_features

    def _preprocess_prefill(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # Replace image id woth PAD if the image token if OOV, to avoid index-errors
        if input_ids is not None and self.config.image_token_index >= self.vocab_size:
            special_image_mask = input_ids == self.config.image_token_index
            llm_input_ids = input_ids.clone()
            llm_input_ids[special_image_mask] = 0
        else:
            llm_input_ids = input_ids

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(llm_input_ids)

        # Merge text and images
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values)
            special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1)
            special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)

            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        return inputs_embeds

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        generate_idx: Optional[torch.Tensor] = None,
        padded_cache_lengths: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        **lm_kwargs: Dict[str, Any],
    ) -> Union[Tuple, RBLNDecoderOnlyOutput]:
        # prefill
        if cache_position is None:
            logits = []
            inputs_embeds = self._preprocess_prefill(input_ids, inputs_embeds, pixel_values)
            batch_size = inputs_embeds.shape[0]

            for b_idx in range(batch_size):
                cache_position = torch.arange(0, generate_idx[b_idx].item(), dtype=torch.int32).unsqueeze(0)
                output = self.language_model.prefill_decoder(
                    inputs_embeds=inputs_embeds[b_idx : b_idx + 1],
                    attention_mask=attention_mask[b_idx],
                    cache_position=cache_position,
                    batch_idx=b_idx,
                    token_type_ids=token_type_ids[b_idx : b_idx + 1] if token_type_ids is not None else None,
                )
                padded_cache_lengths[b_idx] += output.padded_cache_lengths
                logits.append(output.logits)

            logits = torch.cat(logits, dim=0)
        # decoder
        else:
            inputs = inputs_embeds if inputs_embeds is not None else input_ids
            batch_size = inputs.shape[0]
            if batch_size not in self.language_model.decoders:
                raise ValueError(
                    f"No decoder runtime available for batch size {batch_size}. "
                    f"Available batch sizes are: {list(self.decoders.keys())}. "
                    f"Please run your model with one of these batch sizes or add support for batch size {batch_size}."
                )

            logits = self.language_model.decoders[batch_size](
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                position_ids=position_ids if self.rbln_config.language_model.use_position_ids else None,
            ).logits

        return RBLNDecoderOnlyOutput(
            logits=logits, generate_idx=generate_idx, padded_cache_lengths=padded_cache_lengths
        )


class RBLNGemma3RuntimeModel(RBLNRuntimeModel):
    def __init__(self, *args, image_prefill: Optional[rebel.Runtime] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_prefill = image_prefill  # FIXME(taehoon)
        self.prefill = self.runtime if self.phase == "prefill" else None  # FIXME
        self.decode = self.runtime if self.phase == "decode" else None

    def pad_for_chunked_images(
        self,
        inputs: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]:
        """
        Pads inputs, attention_mask, and position_ids so image token groups (256 tokens with token_type_ids == 1)
        start at multiples of prefill_chunk_size (256). Returns padded tensors and total padded length.

        Args:
            inputs: (1, seq_len, hidden_size) tensor.
            attention_mask: (1, seq_len) tensor, 1 for valid, 0 for masked.
            position_ids: (1, seq_len) tensor for RoPE.
            token_type_ids: (1, seq_len) tensor, 0 for text, 1 for image.

        Returns:
            (inputs_padded, attention_mask_padded, position_ids_padded, padded_len, token_type_ids_padded).
        """

        if token_type_ids is None:
            return inputs, attention_mask, position_ids, 0, torch.zeros(inputs.shape[:2], dtype=torch.long)

        seq_len = inputs.shape[1]

        # Find image start positions
        image_starts = [
            s
            for s in range(seq_len - self.rbln_config.prefill_chunk_size + 1)
            if torch.all(token_type_ids[:, s : s + self.rbln_config.prefill_chunk_size] == 1)
        ]

        # Initialize padded tensors
        padded_input_len = seq_len
        for image_start in image_starts:
            pad_needed = (
                self.rbln_config.prefill_chunk_size
                - (image_start + padded_input_len - seq_len) % self.rbln_config.prefill_chunk_size
            ) % self.rbln_config.prefill_chunk_size
            padded_input_len += pad_needed
        total_padding = padded_input_len - seq_len

        if inputs.dim() == 3:
            inputs_padded = torch.zeros(1, padded_input_len, inputs.shape[2], dtype=inputs.dtype)
        else:
            inputs_padded = torch.zeros(1, padded_input_len, dtype=inputs.dtype)
        attention_mask_padded = torch.zeros(1, padded_input_len, dtype=attention_mask.dtype)
        position_ids_padded = torch.zeros(1, padded_input_len, dtype=position_ids.dtype)
        token_type_ids_padded = torch.zeros(1, padded_input_len, dtype=token_type_ids.dtype)

        # Fill padded tensors
        dest_pos = 0
        src_pos = 0
        last_pos_id = -1
        for image_start in image_starts + [seq_len]:
            # Text segment
            if src_pos < image_start:
                length = image_start - src_pos
                inputs_padded[:, dest_pos : dest_pos + length] = inputs[:, src_pos:image_start]
                attention_mask_padded[:, dest_pos : dest_pos + length] = attention_mask[:, src_pos:image_start]
                position_ids_padded[:, dest_pos : dest_pos + length] = position_ids[:, src_pos:image_start]
                token_type_ids_padded[:, dest_pos : dest_pos + length] = token_type_ids[:, src_pos:image_start]
                dest_pos += length
                last_pos_id = position_ids[0, image_start - 1].item()
                src_pos = image_start

            # Padding
            pad_needed = (
                self.rbln_config.prefill_chunk_size - dest_pos % self.rbln_config.prefill_chunk_size
            ) % self.rbln_config.prefill_chunk_size
            if pad_needed and dest_pos < padded_input_len:
                position_ids_padded[:, dest_pos : dest_pos + pad_needed] = torch.arange(
                    last_pos_id + 1, last_pos_id + pad_needed + 1, dtype=position_ids.dtype
                ).unsqueeze(0)
                dest_pos += pad_needed

            # Image segment
            if src_pos < seq_len and src_pos == image_start:
                inputs_padded[:, dest_pos : dest_pos + self.rbln_config.prefill_chunk_size] = inputs[
                    :, src_pos : src_pos + self.rbln_config.prefill_chunk_size
                ]
                attention_mask_padded[:, dest_pos : dest_pos + self.rbln_config.prefill_chunk_size] = attention_mask[
                    :, src_pos : src_pos + self.rbln_config.prefill_chunk_size
                ]
                position_ids_padded[:, dest_pos : dest_pos + self.rbln_config.prefill_chunk_size] = position_ids[
                    :, src_pos : src_pos + self.rbln_config.prefill_chunk_size
                ]
                token_type_ids_padded[:, dest_pos : dest_pos + self.rbln_config.prefill_chunk_size] = token_type_ids[
                    :, src_pos : src_pos + self.rbln_config.prefill_chunk_size
                ]
                dest_pos += self.rbln_config.prefill_chunk_size
                src_pos += self.rbln_config.prefill_chunk_size
                last_pos_id = position_ids[0, image_start + self.rbln_config.prefill_chunk_size - 1].item()

        return inputs_padded, attention_mask_padded, position_ids_padded, total_padding, token_type_ids_padded

    def _prepare_prefill_inputs(
        self,
        inputs: torch.Tensor,
        cache_position: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embed: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ):
        """
        Prepare inputs for prefill phase.
        """
        # Handle continuous batching in a compiled graph by extracting valid inputs
        # If an attention mask is provided, select only the valid (non-masked) inputs
        inputs = inputs[:, attention_mask.bool()] if attention_mask is not None else inputs
        token_type_ids = (
            token_type_ids[:, attention_mask.bool()]
            if attention_mask is not None and token_type_ids is not None
            else token_type_ids
        )

        if position_embed is not None:
            position_embed = (
                position_embed[:, :, :, attention_mask.bool(), :] if attention_mask is not None else position_embed
            )

        seq_len = inputs.shape[1]
        # Initialize attention mask for chunked processing
        if self.rbln_config.use_attention_mask:
            chunked_attention_mask = (
                torch.ones(1, seq_len, dtype=torch.float32)
                if self.rbln_config.use_position_ids
                else torch.zeros(
                    1, 1, self.rbln_config.prefill_chunk_size, self.rbln_config.max_seq_len, dtype=torch.float32
                )
            )
        else:
            chunked_attention_mask = None

        # Buffer for storing output logits
        out_buffers = [
            torch.empty(
                size=self.output_size,
                dtype=torch.float32,
                device="cpu",
            )
        ]

        inputs, chunked_attention_mask, position_ids, padded_cache_lengths, token_type_ids_padded = (
            self.pad_for_chunked_images(inputs, chunked_attention_mask, cache_position, token_type_ids)
        )

        query_length = inputs.shape[1]
        if query_length > self.rbln_config.max_seq_len:
            raise ValueError(
                f"Input length ({query_length}) exceeds the maximum allowed sequence length ({self.rbln_config.max_seq_len})."
            )

        # Align attention_mask to compiled shape
        if self.rbln_config.use_position_ids:
            chunked_attention_mask = torch.nn.functional.pad(
                chunked_attention_mask, (0, self.rbln_config.max_seq_len - query_length)
            )

        # Pad input and cache_position if the last chunk is smaller than `prefill_chunk_size`
        padding_size = 0
        if query_length % self.rbln_config.prefill_chunk_size != 0:
            padding_size = (self.rbln_config.prefill_chunk_size - query_length) % self.rbln_config.prefill_chunk_size
            # inputs_embeds
            if inputs.dim() == 3:
                inputs = torch.nn.functional.pad(inputs, (0, 0, 0, padding_size))
            # inputs_ids
            else:
                inputs = torch.nn.functional.pad(inputs, (0, padding_size))

            position_ids = torch.cat(
                [
                    position_ids,
                    torch.arange(
                        query_length,
                        query_length + padding_size,
                        dtype=torch.int32,
                    ).unsqueeze(0),
                ],
                dim=-1,
            )
            token_type_ids_padded = torch.nn.functional.pad(token_type_ids_padded, (0, padding_size))

            if position_embed is not None:
                position_embed = torch.nn.functional.pad(position_embed, (0, 0, 0, padding_size))

        cache_position = torch.arange(0, query_length + padding_size, dtype=torch.int32).unsqueeze(0)

        return (
            inputs,
            cache_position,
            chunked_attention_mask,
            out_buffers,
            position_ids,
            position_embed,
            padded_cache_lengths,
            query_length,
            token_type_ids_padded,
        )

    def prefill_forward(
        self,
        inputs: torch.Tensor,
        cache_position: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        batch_idx: int = None,
        block_tables: torch.Tensor = None,
        is_external_block_tables: bool = None,
        position_embed: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        local_block_tables: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        """
        Performs chunked prefill for efficient KV-cache updates and memory optimization.
        Instead of processing the entire sequence at once, the input is divided into chunks of size `prefill_chunk_size`,
        and each chunk is processed sequentially. This allows for better memory utilization and compatibility with continuous batching.
        """
        (
            inputs,
            cache_position,
            padded_attention_mask,
            out_buffers,
            position_ids,
            position_embed,
            padded_cache_lengths,
            query_length,
            token_type_ids_padded,
        ) = self._prepare_prefill_inputs(
            inputs, cache_position, attention_mask, position_embed, token_type_ids=token_type_ids
        )
        if not is_external_block_tables:
            local_block_tables = torch.tensor([batch_idx], dtype=torch.int16)
            self.dec_attn_mask[batch_idx : batch_idx + 1] = padded_attention_mask[:1]

        if self.rbln_config.use_attention_mask and self.rbln_config.use_position_ids:
            chunked_attention_mask = torch.zeros(1, self.rbln_config.max_seq_len, dtype=torch.float32)

        # Process input in chunks of size `prefill_chunk_size`
        for step in range(0, query_length, self.rbln_config.prefill_chunk_size):
            # Extract the current chunk of inputs and cache positions
            input_chunk = inputs[:, step : step + self.rbln_config.prefill_chunk_size]
            cache_pos_chunk = cache_position[:, step : step + self.rbln_config.prefill_chunk_size]
            position_ids_chunk = (
                position_ids[:, step : step + self.rbln_config.prefill_chunk_size]
                if position_ids is not None
                else None
            )

            if self.rbln_config.use_attention_mask:
                if self.rbln_config.use_position_ids:
                    chunked_attention_mask[0, step : step + self.rbln_config.prefill_chunk_size] = (
                        padded_attention_mask[0, step : step + self.rbln_config.prefill_chunk_size]
                    )

            # Define query position
            query_position = (
                torch.sum(
                    chunked_attention_mask[0][step : step + self.rbln_config.prefill_chunk_size],
                    dim=-1,
                    dtype=torch.int16,
                ).squeeze(0)
                - 1
            )
            if token_type_ids_padded[:, step] == 1:
                if torch.any(token_type_ids_padded[:, step : step + self.rbln_config.prefill_chunk_size] == 0):
                    raise ValueError("All tokens of image_prefill should be the same image.")
                else:
                    logits = self.image_prefill(
                        input_chunk,
                        cache_pos_chunk,
                        block_tables,
                        local_block_tables,
                        query_position,
                        chunked_attention_mask,
                        position_ids_chunk,
                        out=out_buffers,
                    )
            else:
                # Forward pass for the current chunk
                logits = self.prefill(
                    input_chunk,
                    cache_pos_chunk,
                    block_tables,
                    local_block_tables,
                    query_position,
                    chunked_attention_mask,
                    position_ids_chunk,
                    out=out_buffers,
                )

        return RBLNGemma3ForCausalLMOutput(
            logits=logits, padded_cache_lengths=padded_cache_lengths, attention_mask=chunked_attention_mask
        )

    def decode_forward(
        self,
        inputs: torch.Tensor,
        cache_position: torch.Tensor = None,
        block_tables: torch.Tensor = None,
        is_external_block_tables: bool = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_embed: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        local_block_tables: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        batch_size = inputs.shape[0]
        if batch_size != self.batch_size:
            raise RuntimeError(
                f"Batch size mismatch: got {batch_size}, expected {self.batch_size} (compiled batch size)."
            )

        if batch_size != cache_position.shape[0]:
            raise RuntimeError(f"Cache position size mismatch: got {cache_position.shape[0]}, expected {batch_size}.")

        # FIXME(taehoon): how to handle pos_attn_mask with external block tables
        if is_external_block_tables:
            if attention_mask is None:
                raise ValueError("attention_mask should be provided with external block tables.")
            if local_block_tables is None:
                raise ValueError("local_block_tables should be provided with external block tables.")
        else:
            local_block_tables = (
                local_block_tables
                if local_block_tables is not None
                else torch.arange(0, self.batch_size, dtype=torch.int16).view(self.batch_size, -1)
            )
            if self.rbln_config.use_attention_mask and attention_mask is None:
                for b_idx in range(batch_size):
                    decoding_step = cache_position[b_idx].item()
                    if not (0 <= decoding_step < self.dec_attn_mask.shape[-1]):
                        raise ValueError(
                            f"Decoding step {decoding_step} out of bounds for attention mask with shape {self.dec_attn_mask.shape}."
                        )
                    self.dec_attn_mask[b_idx, decoding_step] = 1

                attention_mask = self.dec_attn_mask

        if self.batch_size < block_tables.shape[0]:
            block_tables = block_tables[: self.batch_size]

        if attention_mask is not None and self.batch_size < attention_mask.shape[0]:
            attention_mask = attention_mask[: self.batch_size]

        logits = self.decode(inputs, cache_position, block_tables, local_block_tables, attention_mask, position_ids)

        return RBLNDecoderOnlyOutput(logits=logits)


class RBLNGemma3ForCausalLM(RBLNDecoderOnlyModelForCausalLM):
    """
    The Gemma3 Model transformer with a language modeling head (linear layer) on top.
    This model inherits from [`RBLNDecoderOnlyModelForCausalLM`]. Check the superclass documentation for the generic methods the library implements for all its models.

    A class to convert and run pre-trained transformers based Gemma3ForCausalLM model on RBLN devices.
    It implements the methods to convert a pre-trained transformers Gemma3ForCausalLM model into a RBLN transformer model by:
    - transferring the checkpoint weights of the original into an optimized RBLN graph,
    - compiling the resulting graph using the RBLN compiler.
    """

    _decoder_wrapper_cls = Gemma3ForCausalLMWrapper

    def __post_init__(self, **kwargs):
        main_input_name = self.main_input_name

        if self.rbln_config.use_inputs_embeds:
            main_input_name = "inputs_embeds"
            artifacts = torch.load(self.model_save_dir / self.subfolder / "torch_artifacts.pth", weights_only=False)
            self.embed_tokens = self._create_embedding_layer()
            self.embed_tokens.load_state_dict(artifacts["embed_tokens"])
        else:
            self.embed_tokens = None

        # Initialize shared resources to be used across Runtime instances (prefill and decode phases)
        dec_attn_mask = torch.zeros(self.rbln_config.batch_size, self.rbln_config.max_seq_len, dtype=torch.float32)
        block_tables = torch.zeros(
            self.rbln_config.batch_size,
            self.rbln_config.max_seq_len // self.rbln_config.kvcache_block_size,
            dtype=torch.int16,
        ).fill_(-1)
        free_block_pool = deque(x for x in range(self.rbln_config.kvcache_num_blocks))
        self.prefill_decoder = RBLNGemma3RuntimeModel(
            runtime=self.model[0],
            image_prefill=self.model[1],
            main_input_name=main_input_name,
            embed_tokens=self.embed_tokens,
            phase="prefill",
            batch_size=self.rbln_config.batch_size,
            dec_attn_mask=dec_attn_mask,
            block_tables=block_tables,
            vocab_size=self.config.vocab_size,
            free_block_pool=free_block_pool,
            rbln_config=self.rbln_config,
        )

        self.decoders = {}
        for i, batch_size in enumerate(self.rbln_config.decoder_batch_sizes):
            self.decoders[batch_size] = RBLNGemma3RuntimeModel(
                runtime=self.model[i + 2],
                main_input_name=main_input_name,
                embed_tokens=self.embed_tokens,
                phase="decode",
                batch_size=batch_size,
                dec_attn_mask=dec_attn_mask,
                block_tables=block_tables,
                free_block_pool=free_block_pool,
                rbln_config=self.rbln_config,
            )

        # NOTE(eunji): Use a decoder whose batch size matches the model's main batch size for compatibility.
        self.decoder = self.decoders[self.rbln_config.batch_size]

    def _create_embedding_layer(self):
        with no_init_weights():
            embed_tokens = Gemma3TextScaledWordEmbedding(
                self.config.vocab_size,
                self.config.hidden_size,
                self.config.pad_token_id,
                embed_scale=self.config.hidden_size**0.5,
            )
        return embed_tokens

    @classmethod
    def _update_sliding_window_config(cls, model_config: PretrainedConfig, rbln_config: RBLNGemma3ForCausalLMConfig):
        sliding_window = getattr(model_config, "sliding_window", None)
        sliding_window_pattern = getattr(model_config, "sliding_window_pattern", None)
        if sliding_window_pattern <= model_config.num_hidden_layers:
            rbln_config.cache_impl = "hybrid"
            rbln_config.sliding_window = sliding_window
            rbln_config.sliding_window_layers = [
                i for i in range(model_config.num_hidden_layers) if (i + 1) % sliding_window_pattern > 0
            ]

        return rbln_config

    @classmethod
    def _update_submodule_config(cls, model: "PreTrainedModel", rbln_config: RBLNModelConfig):
        if rbln_config.prefill_chunk_size is None:
            rbln_config.prefill_chunk_size = model.config.mm_tokens_per_image

        if rbln_config.prefill_chunk_size != model.config.mm_tokens_per_image:
            logger.warning(
                f"Prefill chunk size is different from mm_tokens_per_image: {rbln_config.prefill_chunk_size} != {model.config.mm_tokens_per_image}"
            )
        return rbln_config

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Optional[Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"]] = None,
        model: Optional["PreTrainedModel"] = None,
        model_config: Optional["PretrainedConfig"] = None,
        rbln_config: Optional[RBLNGemma3ForCausalLMConfig] = None,
    ) -> RBLNGemma3ForCausalLMConfig:
        # Update rbln_config with super class
        rbln_config = super()._update_rbln_config(preprocessors, model, model_config, rbln_config)

        # Assume that prefill compile config is at index 0
        compile_cfgs = rbln_config.compile_cfgs
        image_prefill_compile_config = RBLNCompileConfig(
            compiled_model_name="image_prefill", input_info=compile_cfgs[0].input_info
        )
        # Insert image_prefill compile config at index 1
        image_idx = 1
        compile_cfgs.insert(image_idx, image_prefill_compile_config)
        rbln_config.set_compile_cfgs(compile_cfgs)

        return rbln_config

    @classmethod
    @torch.inference_mode()
    def get_compiled_model(cls, model: "PreTrainedModel", rbln_config: RBLNGemma3ForCausalLMConfig):
        wrapped_model = cls.wrap_model_if_needed(model, rbln_config)

        rbln_compile_configs = rbln_config.compile_cfgs
        prefill_compile_config = rbln_compile_configs[0]

        context = CompileContext(use_weight_sharing=True)

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
                compiled_model = cls.compile(
                    wrapped_model,
                    compile_config,
                    create_runtimes=rbln_config.create_runtimes,
                    device=rbln_config.device,
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
            wrapped_model,
            prefill_compile_config,
            prefill_example_inputs,
            context,
            rbln_config.quantization,
        )

        image_prefill_compile_config = rbln_compile_configs[1]
        wrapped_model.phase = "image_prefill"
        compiled_image_prefill = compile_model(
            wrapped_model,
            image_prefill_compile_config,
            prefill_example_inputs,
            context,
            rbln_config.quantization,
        )

        compiled_models = {"prefill": compiled_prefill, "image_prefill": compiled_image_prefill}
        wrapped_model.phase = "decode"
        for batch_size, dec_compile_config in zip(rbln_config.decoder_batch_sizes, rbln_compile_configs[2:]):
            dec_example_inputs = dec_compile_config.get_dummy_inputs(fill=0, static_tensors=static_tensors)
            compiled_decoder = compile_model(
                wrapped_model,
                dec_compile_config,
                dec_example_inputs,
                context,
                rbln_config.quantization,
            )
            compiled_models[f"decoder_batch_{batch_size}"] = compiled_decoder

        return compiled_models

    @classmethod
    def _create_runtimes(
        cls,
        compiled_models: List[rebel.RBLNCompiledModel],
        rbln_config: RBLNGemma3ForCausalLMConfig,
    ) -> List[rebel.Runtime]:
        expected_model_names = [
            "prefill",
            "image_prefill",
            *[f"decoder_batch_{batch_size}" for batch_size in rbln_config.decoder_batch_sizes],
        ]
        if any(model_name not in rbln_config.device_map for model_name in expected_model_names):
            cls._raise_missing_compiled_file_error(expected_model_names)

        return [
            rebel.Runtime(
                compiled_models[0],
                tensor_type="pt",
                device=rbln_config.device_map["prefill"],
                activate_profiler=rbln_config.activate_profiler,
            ),
            rebel.Runtime(
                compiled_models[1],
                tensor_type="pt",
                device=rbln_config.device_map["image_prefill"],
                activate_profiler=rbln_config.activate_profiler,
            ),
            *[
                rebel.Runtime(
                    compiled_models[i + 2],
                    tensor_type="pt",
                    device=rbln_config.device_map[f"decoder_batch_{batch_size}"],
                    activate_profiler=rbln_config.activate_profiler,
                )
                for i, batch_size in enumerate(rbln_config.decoder_batch_sizes)
            ],
        ]
