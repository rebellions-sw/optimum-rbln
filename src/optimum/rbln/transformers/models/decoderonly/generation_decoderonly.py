from typing import TYPE_CHECKING, Any, Dict, Optional

import torch
from transformers.generation.utils import GenerationMixin


if TYPE_CHECKING:
    from ...modeling_outputs import RBLNDecoderOnlyForCausalLMOutput


class RBLNDecoderOnlyGenerationMixin(GenerationMixin):
    _supports_cache_class = False  # Needed for GenerationMixin
    _is_stateful = False  # Needed for GenerationMixin

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        generate_idx: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        padded_cache_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        model_inputs = {}
        is_prefill_phase = generate_idx is None

        if is_prefill_phase:
            generate_idx = attention_mask.sum(dim=-1, keepdim=True).int()
            padded_cache_lengths = torch.zeros_like(generate_idx)
            cache_position = attention_mask.cumsum(dim=-1, dtype=torch.int32) - 1
            position_ids = None
        else:
            if inputs_embeds is not None:
                # if `inputs_embeds` are passed, only use them in the 1st generation step for every prompt.
                inputs_embeds = None

            input_ids = input_ids[:, -1:]
            position_ids = generate_idx
            cache_position = generate_idx + padded_cache_lengths if padded_cache_lengths is not None else generate_idx
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
                "cache_position": cache_position,
                "generate_idx": generate_idx,
                "position_ids": position_ids,
                "padded_cache_lengths": padded_cache_lengths,
            }
        )

        return model_inputs

    def _update_model_kwargs_for_generation(
        self, outputs: "RBLNDecoderOnlyForCausalLMOutput", model_kwargs: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        # update generate_idx
        model_kwargs["generate_idx"] = outputs.generate_idx
        model_kwargs["padded_cache_lengths"] = outputs.padded_cache_lengths
        return model_kwargs

    # def _prefill_forward(self, *args, **kwargs):
    #     return self._forward(*args, phase="prefill", **kwargs)

    # def _decode_forward(self, *args, **kwargs):
    #     return self._forward(*args, phase="decode", **kwargs)

    # def _forward(
    #     self,
    #     input_ids: Optional[torch.LongTensor] = None,
    #     inputs_embeds: Optional[torch.Tensor] = None,
    #     cache_position: torch.Tensor = None,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     batch_idx: Optional[int] = None,
    #     block_tables: Optional[torch.Tensor] = None,
    #     position_embed: Optional[torch.Tensor] = None,
    #     position_ids: Optional[torch.Tensor] = None,
    #     token_type_ids: Optional[torch.Tensor] = None,
    #     local_block_tables: Optional[torch.Tensor] = None,
    #     phase: str = "prefill",
    # ):
    #     inputs = self.inputs_embeddings_if_needed(input_ids, inputs_embeds)
    #     block_tables, local_block_tables, is_external_block_tables = (
    #         self.page_table_manager.get_block_tables_if_needed(
    #             inputs.shape[0],
    #             cache_position,
    #             batch_idx=batch_idx,
    #             phase=phase,
    #             block_tables=block_tables,
    #             local_block_tables=local_block_tables,
    #         )
    #     )

    #     if phase == "decode":
    #         return self._decode(
    #             inputs,
    #             cache_position,
    #             block_tables,
    #             is_external_block_tables,
    #             attention_mask=attention_mask,
    #             position_embed=position_embed,
    #             position_ids=position_ids,
    #             local_block_tables=local_block_tables,
    #         )
    #     else:
    #         return self._chunked_prefill_forward(
    #             inputs,
    #             cache_position,
    #             attention_mask,
    #             batch_idx,
    #             block_tables,
    #             is_external_block_tables=is_external_block_tables,
    #             position_embed=position_embed,
    #             token_type_ids=token_type_ids,
    #             local_block_tables=local_block_tables,
    #         )

    # def _validate_decoder_batch_size(self, inputs: torch.Tensor, **kwargs):
    #     batch_size = inputs.shape[0]
    #     if batch_size not in self.rbln_config.decoder_batch_sizes:
    #         raise RuntimeError(
    #             f"Batch size mismatch: got {batch_size}, expected one of {self.rbln_config.decoder_batch_sizes} (compiled batch size)."
    #         )

    #     for arg_name, arg_value in kwargs.items():
    #         if arg_value is not None and arg_value.shape[0] != batch_size:
    #             raise RuntimeError(f"{arg_name} batch size mismatch: got {arg_value.shape[0]}, expected {batch_size}.")

    # def _decode(
    #     self,
    #     inputs: torch.Tensor,
    #     cache_position: torch.Tensor = None,
    #     block_tables: torch.Tensor = None,
    #     is_external_block_tables: bool = None,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     position_embed: Optional[torch.Tensor] = None,
    #     position_ids: Optional[torch.Tensor] = None,
    #     local_block_tables: Optional[torch.Tensor] = None,
    # ) -> torch.FloatTensor:
    #     batch_size = inputs.shape[0]
    #     self._validate_decoder_batch_size(
    #         inputs,
    #         cache_position=cache_position,
    #         block_tables=block_tables,
    #         attention_mask=attention_mask,
    #         position_embed=position_embed,
    #         position_ids=position_ids,
    #     )

    #     if self.rbln_config.use_attention_mask and attention_mask is None:
    #         for b_idx in range(batch_size):
    #             decoding_step = cache_position[b_idx].item()
    #             if not (0 <= decoding_step < self.dec_attn_mask.shape[-1]):
    #                 raise ValueError(
    #                     f"Decoding step {decoding_step} out of bounds for attention mask with shape {self.dec_attn_mask.shape}."
    #                 )

    #             if is_external_block_tables:
    #                 self.dec_attn_mask[b_idx].fill_(0)
    #                 self.dec_attn_mask[b_idx, :, :, : decoding_step + 1] = 1
    #             else:
    #                 self.dec_attn_mask[b_idx, :, :, decoding_step] = 1

    #         attention_mask = self.dec_attn_mask

    #     logits = self.decoders_runtime[batch_size](
    #         inputs,
    #         cache_position,
    #         block_tables,
    #         local_block_tables,
    #         position_embed,
    #         attention_mask if self.rbln_config.use_attention_mask else None,
    #         position_ids if self.rbln_config.use_position_ids else None,
    #     )

    #     return RBLNDecoderOnlyForCausalLMOutput(logits=logits)
