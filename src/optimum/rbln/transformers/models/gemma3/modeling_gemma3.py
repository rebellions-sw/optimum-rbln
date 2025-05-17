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
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Union

import rebel
import torch
from rebel.compile_context import CompileContext
from transformers import (
    AutoModelForVision2Seq,
    Gemma3ForConditionalGeneration,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.models.gemma3.modeling_gemma3 import Gemma3TextScaledWordEmbedding
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.modeling_utils import no_init_weights

from ....configuration_utils import RBLNCompileConfig, RBLNModelConfig
from ....modeling import RBLNModel
from ....utils.logging import get_logger
from ..decoderonly.decoderonly_architecture import (
    set_default_values,
    validate_attention_method,
)
from ...utils.rbln_quantization import QuantizationManager
from ..decoderonly.modeling_decoderonly import RBLNDecoderOnlyModelForCausalLM, RBLNDecoderOnlyOutput
from .configuration_gemma3 import RBLNGemma3ForCausalLMConfig
from .gemma3_architecture import Gemma3ForCausalLMWrapper


logger = get_logger()


if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer


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
    auto_model_class = AutoModelForVision2Seq
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
        **kwargs,
    ):
        # Prepare HF generation
        is_prefill_phase = generate_idx is None
        batch_size = input_ids.shape[0]

        model_inputs = self.language_model.prepare_inputs_for_generation(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            generate_idx=generate_idx,  # Not affect
            attention_mask=attention_mask,
            **kwargs,
        )

        if is_prefill_phase:
            model_inputs.update(
                {
                    "pixel_values": pixel_values,
                    "image_sizes": image_sizes,
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

        return model_kwargs

    def get_image_features(self, pixel_values: torch.Tensor):
        """
        Projects the last hidden state from the vision model into language model space.

        Args:
            pixel_values (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`)
               The tensors corresponding to the input images.
        Returns:
            image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
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
        token_type_ids: Optional[torch.Tensor] = None,
        **lm_kwargs,
    ) -> Union[Tuple, RBLNDecoderOnlyOutput]:
        # prefill
        if cache_position is None:
            logits = []
            inputs_embeds = self._preprocess_prefill(input_ids, inputs_embeds, pixel_values)
            batch_size = inputs_embeds.shape[0]

            for b_idx in range(batch_size):
                cache_position = torch.arange(0, generate_idx[b_idx].item(), dtype=torch.int32).unsqueeze(0)
                logit = self.language_model.prefill_decoder(
                    inputs_embeds=inputs_embeds[b_idx : b_idx + 1],
                    attention_mask=attention_mask[b_idx],
                    cache_position=cache_position,
                    batch_idx=b_idx,
                )
                logits.append(logit)

            logits = torch.cat(logits, dim=0)
        # decoder
        else:
            inputs = inputs_embeds if inputs_embeds is not None else input_ids
            batch_size = inputs.shape[0]

            logits = self.language_model.decoders[batch_size](
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
            )

        return RBLNDecoderOnlyOutput(
            logits=logits,
            generate_idx=generate_idx,
        )


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

    def _embedding_instance(self):
        with no_init_weights():
            embed_tokens = Gemma3TextScaledWordEmbedding(
                self.config.vocab_size,
                self.config.hidden_size,
                self.config.pad_token_id,
                embed_scale=self.config.hidden_size**0.5,
            )
        return embed_tokens

    @classmethod
    def get_input_info(
        cls,
        batch_size: int,
        query_length: int,
        use_inputs_embeds: bool,
        use_attention_mask: bool,
        max_seq_len: int,
        kvcache_block_size: int,
        kvcache_num_blocks: int,
        num_key_value_heads: int,
        num_hidden_layers: int,
        hidden_size: int,
        head_dim: int,
        sliding_window: int,
        sliding_window_pattern: int,
    ):
        if use_inputs_embeds:
            main_input = ("inputs_embeds", [batch_size, query_length, hidden_size], "float32")
        else:
            main_input = ("input_ids", [batch_size, query_length], "int64")

        # FIXME: thkim, change the name 'use_attention_mask' as 'attention_mask_type': Enum['2d','4d','None']
        input_info = [
            main_input,
            (
                "attention_mask",
                [batch_size, 1, query_length, max_seq_len] if use_attention_mask else [batch_size, max_seq_len],
                "float32",
            ),
            (
                "cache_position",
                [batch_size, query_length],
                "int32",
            ),
            (
                "position_ids",
                [batch_size, query_length],
                "int32",
            ),
        ]

        if query_length > 1:
            input_info.extend(
                [
                    ("query_position", [], "int16"),
                ]
            )

        max_block_cnt = max_seq_len // kvcache_block_size

        if query_length > 1:
            input_info.extend([("global_block_tables", [max_block_cnt], "int16")])
            input_info.extend([("local_block_tables", [1], "int16")])
        else:
            input_info.extend([("global_block_tables", [batch_size, max_block_cnt], "int16")])
            input_info.extend([("local_block_tables", [batch_size, 1], "int16")])

        def is_sliding(layer_idx: int) -> bool:
            return bool((layer_idx + 1) % sliding_window_pattern)

        local_kvcache_shape = [batch_size, num_key_value_heads, sliding_window, head_dim]
        global_kvcache_shape = [kvcache_num_blocks, num_key_value_heads, kvcache_block_size, head_dim]
        input_info.extend(
            [
                (
                    f"past_key_values_{i}",
                    local_kvcache_shape if is_sliding(i // 2) else global_kvcache_shape,
                    "float32",
                )
                for i in range(num_hidden_layers * 2)
            ]
        )

        return input_info

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Optional[Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"]] = None,
        model: Optional["PreTrainedModel"] = None,
        model_config: Optional["PretrainedConfig"] = None,
        rbln_config: Optional[RBLNGemma3ForCausalLMConfig] = None,
    ) -> RBLNGemma3ForCausalLMConfig:
        rbln_config.prefill_chunk_size = 256  # FIXME: Hard-coded

        if rbln_config.max_seq_len is None:
            rbln_config.max_seq_len = getattr(model_config, "max_position_embeddings", None)
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

        if rbln_config.attn_impl == "flash_attn":
            # TODO(taehoon): override the get_maximum_num_blocks function
            # estimated_max_num_blocks = cls.get_maximum_num_blocks(
            #     config=model_config,
            #     tensor_parallel_size=rbln_config.tensor_parallel_size or 1,
            #     kvcache_block_size=rbln_config.kvcache_block_size,
            #     nbits_per_param=16 if not rbln_config.quantization else 4,  # TODO(jongho): FIX Ad-hoc
            #     n_model_params=sum(p.numel() for p in model.parameters()),
            # )

            # max_num_blocks = min(max_num_blocks, estimated_max_num_blocks)

            flash_min_blocks = rbln_config.max_seq_len // rbln_config.kvcache_block_size + 1
            if max_num_blocks < flash_min_blocks:
                max_num_blocks = flash_min_blocks

            if max_num_blocks < rbln_config.batch_size:
                raise RuntimeError(
                    f"Batch size ({rbln_config.batch_size}) exceeds available KV cache blocks ({max_num_blocks}). "
                    "Ensure the number of blocks is at least equal to the batch size."
                )

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
        sliding_window = getattr(model_config, "sliding_window", None)
        sliding_window_pattern = getattr(model_config, "sliding_window_pattern", None)

        prefill_input_info = cls.get_input_info(
            batch_size=1,
            query_length=rbln_config.prefill_chunk_size,
            use_inputs_embeds=rbln_config.use_inputs_embeds,
            use_attention_mask=rbln_config.use_attention_mask,
            max_seq_len=rbln_config.max_seq_len,
            kvcache_block_size=rbln_config.kvcache_block_size,
            kvcache_num_blocks=rbln_config.kvcache_num_blocks,
            num_key_value_heads=num_key_value_heads,
            num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_size,
            head_dim=head_dim,
            sliding_window=sliding_window,
            sliding_window_pattern=sliding_window_pattern,
        )
        prefill_compile_config = RBLNCompileConfig(compiled_model_name="text_prefill", input_info=prefill_input_info)
        image_prefill_compile_config = RBLNCompileConfig(
            compiled_model_name="image_prefill", input_info=prefill_input_info
        )

        dec_compile_configs = []
        for batch_size in rbln_config.decoder_batch_sizes:
            dec_input_info = cls.get_input_info(
                batch_size=batch_size,
                query_length=1,
                use_inputs_embeds=rbln_config.use_inputs_embeds,
                use_attention_mask=rbln_config.use_attention_mask,
                max_seq_len=rbln_config.max_seq_len,
                kvcache_block_size=rbln_config.kvcache_block_size,
                kvcache_num_blocks=rbln_config.kvcache_num_blocks,
                num_key_value_heads=num_key_value_heads,
                num_hidden_layers=num_hidden_layers,
                hidden_size=hidden_size,
                head_dim=head_dim,
                sliding_window=sliding_window,
                sliding_window_pattern=sliding_window_pattern,
            )
            dec_compile_configs.append(
                RBLNCompileConfig(compiled_model_name=f"decoder_batch_{batch_size}", input_info=dec_input_info)
            )
        rbln_config.set_compile_cfgs([prefill_compile_config, image_prefill_compile_config, *dec_compile_configs])

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

        @QuantizationManager.with_quantization_env
        def compile_model(wrapped_model, compile_config, example_inputs, compile_context, **kwargs):
            try:
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

        wrapped_model.phase = "text_prefill"
        compiled_prefill = compile_model(
            wrapped_model,
            prefill_compile_config,
            prefill_example_inputs,
            context,
            quantize_config=rbln_config.quantization,
        )

        image_prefill_compile_config = rbln_compile_configs[1]
        wrapped_model.phase = "image_prefill"
        compiled_image_prefill = compile_model(
            wrapped_model,
            image_prefill_compile_config,
            prefill_example_inputs,
            context,
            quantize_config=rbln_config.quantization,
        )

        wrapped_model.phase = "decode"
        compiled_models = {"text_prefill": compiled_prefill, "image_prefill": compiled_image_prefill}
        for batch_size, dec_compile_config in zip(rbln_config.decoder_batch_sizes, rbln_compile_configs[2:]):
            dec_example_inputs = dec_compile_config.get_dummy_inputs(fill=0, static_tensors=static_tensors)
            compiled_decoder = compile_model(
                wrapped_model,
                dec_compile_config,
                dec_example_inputs,
                context,
                quantize_config=rbln_config.quantization,
            )
            compiled_models[f"decoder_batch_{batch_size}"] = compiled_decoder

        # TODO overide maybe_suggest_kvcache_num_blocks
        # check if the memory is enough to have additional blocks
        # required_num_blocks = (rbln_config.max_seq_len // rbln_config.kvcache_block_size) * rbln_config.batch_size
        # if rbln_config.kvcache_num_blocks < required_num_blocks:
        #     cls.maybe_suggest_kvcache_num_blocks(
        #         compiled_models=compiled_models,
        #         model_config=model.config,
        #         rbln_config=rbln_config,
        #     )

        return compiled_models
