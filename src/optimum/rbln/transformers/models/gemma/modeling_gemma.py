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
from typing import TYPE_CHECKING

import torch
from rebel.compile_context import CompileContext

from ....modeling import RBLNModel
from ...models.decoderonly import RBLNDecoderOnlyModelForCausalLM, RBLNDecoderOnlyModelForCausalLMConfig
from .gemma_architecture import GemmaWrapper


if TYPE_CHECKING:
    from transformers import PreTrainedModel


class RBLNGemmaForCausalLM(RBLNDecoderOnlyModelForCausalLM):
    """
    The Gemma Model transformer with a language modeling head (linear layer) on top.
    This model inherits from [`RBLNDecoderOnlyModelForCausalLM`]. Check the superclass documentation for the generic methods the library implements for all its models.

    A class to convert and run pre-trained transformers based GemmaForCausalLM model on RBLN devices.
    It implements the methods to convert a pre-trained transformers GemmaForCausalLM model into a RBLN transformer model by:

    - transferring the checkpoint weights of the original into an optimized RBLN graph,
    - compiling the resulting graph using the RBLN compiler.

    **Configuration:**
    This model uses [`RBLNGemmaForCausalLMConfig`] for configuration. When calling methods like `from_pretrained` or `from_model`,
    the `rbln_config` parameter should be an instance of [`RBLNGemmaForCausalLMConfig`] or a dictionary conforming to its structure.

    See the [`RBLNGemmaForCausalLMConfig`] class for all available configuration options.

    Examples:
        ```python
        from optimum.rbln import RBLNGemmaForCausalLM

        # Simple usage using rbln_* arguments
        # `max_seq_len` is automatically inferred from the model config
        model = RBLNGemmaForCausalLM.from_pretrained(
            "google/gemma-7b",
            export=True,
            rbln_batch_size=1,
            rbln_tensor_parallel_size=4,
        )


        # Using a config dictionary
        rbln_config = {
            "batch_size": 1,
            "max_seq_len": 4096,
            "tensor_parallel_size": 4,
        }
        model = RBLNGemmaForCausalLM.from_pretrained(
            "google/gemma-7b",
            export=True,
            rbln_config=rbln_config
        )


        # Using a RBLNGemmaForCausalLMConfig instance (recommended for type checking)
        from optimum.rbln import RBLNGemmaForCausalLMConfig

        config = RBLNGemmaForCausalLMConfig(
            batch_size=1,
            max_seq_len=4096,
            tensor_parallel_size=4
        )
        model = RBLNGemmaForCausalLM.from_pretrained(
            "google/gemma-7b",
            export=True,
            rbln_config=config
        )
        ```
    """

    _decoder_wrapper_cls = GemmaWrapper

    # FIXME: Workaround patch only for Test ColPali compile.
    @classmethod
    @torch.inference_mode()
    def get_compiled_model(cls, model: "PreTrainedModel", rbln_config: RBLNDecoderOnlyModelForCausalLMConfig):
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

        wrapped_model.phase = "image_prefill"
        compiled_prefill = compile_model(
            wrapped_model, prefill_compile_config, prefill_example_inputs, context, rbln_config.quantization
        )
        compiled_models = {"prefill": compiled_prefill}
        if rbln_config.is_generation_mode:
            wrapped_model.phase = "decode"
            for batch_size, dec_compile_config in zip(rbln_config.decoder_batch_sizes, rbln_compile_configs[1:]):
                dec_example_inputs = dec_compile_config.get_dummy_inputs(fill=0, static_tensors=static_tensors)
                compiled_decoder = compile_model(
                    wrapped_model, dec_compile_config, dec_example_inputs, context, rbln_config.quantization
                )
                compiled_models[f"decoder_batch_{batch_size}"] = compiled_decoder

            # check if the memory is enough to have additional blocks
            required_num_blocks = (rbln_config.max_seq_len // rbln_config.kvcache_block_size) * rbln_config.batch_size
            if rbln_config.kvcache_num_blocks < required_num_blocks:
                cls.maybe_suggest_kvcache_num_blocks(
                    compiled_models=compiled_models,
                    model_config=model.config,
                    rbln_config=rbln_config,
                )

        return compiled_models
