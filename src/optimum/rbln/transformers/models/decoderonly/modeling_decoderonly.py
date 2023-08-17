# Copyright 2024 Rebellions Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Portions of this software are licensed under the Apache License,
# Version 2.0. See the NOTICE file distributed with this work for
# additional information regarding copyright ownership.

# All other portions of this software, including proprietary code,
# are the intellectual property of Rebellions Inc. and may not be
# copied, modified, or distributed without prior written permission
# from Rebellions Inc.

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import rebel
import torch
from rebel.compile_context import CompileContext
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.modeling_utils import no_init_weights
from transformers.utils import ModelOutput

from ....modeling import RBLNModel
from ....modeling_config import RBLNCompileConfig, RBLNConfig
from ....utils.logging import get_logger
from ....utils.runtime_utils import RBLNPytorchRuntime
from ...utils.rbln_quantization import QuantizationManager
from .decoderonly_architecture import (
    DecoderOnlyWrapper,
    validate_attention_method,
)


logger = get_logger()

if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer, PretrainedConfig


class RBLNRuntimeModel(RBLNPytorchRuntime):
    mandatory_members = ["main_input_name", "embed_tokens"]

    def forward(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        cache_position: torch.Tensor,
        **kwargs,
    ):
        if inputs_embeds is None:
            inp = input_ids
            if self.embed_tokens is not None:
                inp = self.embed_tokens(inp)
        else:
            inp = inputs_embeds

        return super().forward(
            inp,
            attention_mask,
            cache_position,
            **kwargs,
        )


@dataclass
class RBLNDecoderOnlyOutput(ModelOutput):
    logits: torch.FloatTensor = None
    generate_idx: torch.Tensor = None


class RBLNDecoderOnlyModelForCausalLM(RBLNModel):
    """
    A base class for decoder-only transformer models optimized for causal language modeling tasks on RBLN devices.
    This class serves as the foundation for various decoder-only architectures like GPT, LLaMA, etc.

    The class provides core functionality for:
    1. Converting pre-trained transformer models to RBLN-optimized format
    2. Handling the compilation process for RBLN devices
    3. Managing inference operations for causal language modeling

    This class inherits from RBLNModel and implements specific methods required for
    decoder-only architectures and causal language modeling tasks.

    Note:
        - This class is designed to be subclassed by specific model implementations
          (e.g., RBLNLlamaForCausalLM, RBLNGPT2LMHeadModel)
        - Subclasses should implement model-specific conversion logic.
        - The class handles RBLN-specific optimizations automatically during compilation
    """

    main_input_name = "input_ids"
    auto_model_class = AutoModelForCausalLM
    _decoder_wrapper_cls = DecoderOnlyWrapper
    _use_rotary_emb = True

    def __post_init__(self, **kwargs):
        self.batch_size = self.rbln_config.model_cfg["batch_size"]
        self.max_seq_len = self.rbln_config.model_cfg["max_seq_len"]
        self.prefill_chunk_size = self.rbln_config.model_cfg["prefill_chunk_size"]

        self.prefill_attention_mask = torch.zeros(1, 1, self.prefill_chunk_size, self.max_seq_len, dtype=torch.float32)
        self.causal_mask = 1 - torch.triu(
            torch.ones(1, 1, self.prefill_chunk_size, self.prefill_chunk_size), diagonal=1
        )
        self.dec_attn_mask_init = torch.zeros(1, 1, 1, self.max_seq_len, dtype=torch.float32)
        self.dec_attn_mask = torch.zeros(self.batch_size, 1, 1, self.max_seq_len, dtype=torch.float32)

        main_input_name = self.main_input_name
        if self.rbln_config.model_cfg["use_inputs_embeds"]:
            main_input_name = "inputs_embeds"
            artifacts = torch.load(self.model_save_dir / self.subfolder / "torch_artifacts.pth", weights_only=False)
            with no_init_weights():
                self.embed_tokens = torch.nn.Embedding(
                    self.config.vocab_size,
                    self.config.hidden_size,
                    self.config.pad_token_id,
                )
            self.embed_tokens.load_state_dict(artifacts["embed_tokens"])
        else:
            self.embed_tokens = None

        self.prefill_decoder = RBLNRuntimeModel(
            runtime=self.model[0], main_input_name=main_input_name, embed_tokens=self.embed_tokens
        )
        self.decoder = RBLNRuntimeModel(
            runtime=self.model[1], main_input_name=main_input_name, embed_tokens=self.embed_tokens
        )

    @classmethod
    def save_torch_artifacts(
        cls,
        model: "PreTrainedModel",
        save_dir_path: Path,
        subfolder: str,
        rbln_config: RBLNConfig,
    ):
        """
        If you are unavoidably running on a CPU rather than an RBLN device,
        store the torch tensor, weight, etc. in this function.
        """
        if rbln_config.model_cfg["use_inputs_embeds"]:
            save_dict = {}
            save_dict["embed_tokens"] = model.get_input_embeddings().state_dict()
            torch.save(save_dict, save_dir_path / subfolder / "torch_artifacts.pth")

    def get_input_embeddings(self):
        return self.embed_tokens

    @classmethod
    def get_quantized_model(
        cls,
        model_id: str,
        config: Optional[PretrainedConfig] = None,
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        subfolder: str = "",
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        **kwargs,
    ):
        from ...utils.rbln_quantization import prepare_model_for_quantization

        kwargs = cls.update_kwargs(kwargs)

        if config is None:
            config = AutoConfig.from_pretrained(
                model_id,
                use_auth_token=use_auth_token,
                revision=revision,
                force_download=force_download,
                cache_dir=cache_dir,
                trust_remote_code=trust_remote_code,
                **kwargs,
            )

        with no_init_weights():
            model = AutoModelForCausalLM.from_config(config)

        prepare_model_for_quantization(model, model_id, kwargs.get("num_hidden_layers"))

        return model

    def __getattr__(self, __name: str) -> Any:
        """
        Special method to delegate attribute access to the original Huggingface LM class.
        This method is called when an attribute is not found in the current instance's dictionary.
        It enables transparent access to the original model's attributes and methods while maintaining
        proper method binding.

        The method implements a delegation pattern that:
        1. For methods: Creates a wrapper that properly binds 'self' to method calls
        2. For other attributes: Returns them directly from the original class
        """

        def redirect(func):
            return lambda *pargs, **kwargs: func(self, *pargs, **kwargs)

        val = getattr(self.hf_class, __name, None) or getattr(PreTrainedModel, __name)
        if isinstance(val, Callable) and "self" in set(inspect.signature(val).parameters):
            return redirect(val)
        return val

    @classmethod
    def get_pytorch_model(cls, *args, **kwargs) -> "PreTrainedModel":
        rbln_kwargs = kwargs.get("rbln_kwargs", {})
        rbln_quantization = rbln_kwargs.get("quantization", None)

        if rbln_quantization is not None and rbln_quantization["format"] == "rbln":
            model = cls.get_quantized_model(*args, **kwargs)
        else:
            model = super().get_pytorch_model(*args, **kwargs)

        return model

    @classmethod
    def wrap_model_if_needed(cls, model: "PreTrainedModel", rbln_config: "RBLNConfig"):
        wrapper_cfg = {"max_seq_len": rbln_config.model_cfg["max_seq_len"]}
        wrapper_cfg["attn_impl"] = rbln_config.model_cfg.get("attn_impl")
        wrapper_cfg["kvcache_partition_len"] = rbln_config.model_cfg.get("kvcache_partition_len")
        wrapper_cfg["use_rotary_emb"] = cls._use_rotary_emb

        return cls._decoder_wrapper_cls(model, **wrapper_cfg).eval()

    @classmethod
    @torch.inference_mode()
    def get_compiled_model(cls, model: "PreTrainedModel", rbln_config: RBLNConfig):
        wrapped_model = cls.wrap_model_if_needed(model, rbln_config)

        rbln_compile_configs = rbln_config.compile_cfgs
        prefill_compile_config = rbln_compile_configs[0]
        dec_compile_config = rbln_compile_configs[1]

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

        dec_example_inputs = dec_compile_config.get_dummy_inputs(fill=0, static_tensors=static_tensors)

        quantize_config = rbln_config.model_cfg.get("quantization", None)

        @QuantizationManager.with_quantization_env
        def compile_model(*args, **kwargs):
            wrapped_model.phase = "prefill"
            compiled_prefill = RBLNModel.compile(
                wrapped_model,
                prefill_compile_config,
                example_inputs=prefill_example_inputs,
                compile_context=context,
            )

            wrapped_model.phase = "decode"
            compiled_decoder = RBLNModel.compile(
                wrapped_model,
                dec_compile_config,
                example_inputs=dec_example_inputs,
                compile_context=context,
            )
            return {"prefill": compiled_prefill, "decoder": compiled_decoder}

        return compile_model(quantize_config=quantize_config)

    @classmethod
    def _get_rbln_config(
        cls,
        preprocessors: Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"],
        model_config: "PretrainedConfig",
        rbln_kwargs: Dict[str, Any] = {},
    ) -> RBLNConfig:
        rbln_max_seq_len = rbln_kwargs.get("max_seq_len", None)
        rbln_batch_size = rbln_kwargs.get("batch_size", None)
        rbln_use_inputs_embeds = rbln_kwargs.get("use_inputs_embeds", None)
        rbln_attn_impl = rbln_kwargs.get("attn_impl", None)
        rbln_kvcache_partition_len = rbln_kwargs.get("kvcache_partition_len", None)
        rbln_quantization = QuantizationManager.validate_quantization_config(rbln_kwargs.get("quantization", None))

        prefill_chunk_size = 128
        if rbln_max_seq_len is None:
            rbln_max_seq_len = getattr(model_config, "max_position_embeddings", None) or getattr(
                model_config, "n_positions", None
            )
        if rbln_max_seq_len is None:
            raise ValueError("`rbln_max_seq_len` should be specified.")

        rbln_batch_size = 1 if rbln_batch_size is None else rbln_batch_size
        rbln_use_inputs_embeds = False if rbln_use_inputs_embeds is None else rbln_use_inputs_embeds

        rbln_attn_impl, rbln_kvcache_partition_len = validate_attention_method(
            rbln_attn_impl=rbln_attn_impl,
            rbln_kvcache_partition_len=rbln_kvcache_partition_len,
            rbln_max_seq_len=rbln_max_seq_len,
        )

        num_attention_heads = getattr(model_config, "n_head", None) or getattr(model_config, "num_attention_heads")
        num_key_value_heads = getattr(model_config, "num_key_value_heads", None) or num_attention_heads
        num_hidden_layers = getattr(model_config, "n_layer", None) or getattr(model_config, "num_hidden_layers")
        head_dim = getattr(model_config, "head_dim", None) or model_config.hidden_size // num_attention_heads
        hidden_size = getattr(model_config, "n_embd", None) or getattr(model_config, "hidden_size")

        def get_input_info(
            batch_size,
            query_length,
            use_inputs_embeds,
            hidden_size,
        ):
            if use_inputs_embeds:
                main_input = ("inputs_embeds", [batch_size, query_length, hidden_size], "float32")
            else:
                main_input = ("input_ids", [batch_size, query_length], "int64")

            input_info = [
                main_input,
                ("attention_mask", [batch_size, 1, query_length, rbln_max_seq_len], "float32"),
                (
                    "cache_position",
                    [batch_size, query_length],
                    "int32",
                ),
            ]
            if query_length > 1:
                input_info.extend(
                    [
                        ("batch_position", [], "int16"),
                        ("query_position", [], "int16"),
                    ]
                )

            input_info.extend(
                [
                    (
                        f"past_key_values_{i}",
                        [
                            rbln_batch_size,
                            num_key_value_heads,
                            rbln_max_seq_len,
                            head_dim,
                        ],
                        "float32",
                    )
                    for i in range(num_hidden_layers * 2)
                ]
            )

            return input_info

        prefill_input_info = get_input_info(
            batch_size=1,
            query_length=prefill_chunk_size,
            use_inputs_embeds=rbln_use_inputs_embeds,
            hidden_size=hidden_size,
        )
        dec_input_info = get_input_info(
            batch_size=rbln_batch_size,
            query_length=1,
            use_inputs_embeds=rbln_use_inputs_embeds,
            hidden_size=hidden_size,
        )

        prefill_compile_config = RBLNCompileConfig(compiled_model_name="prefill", input_info=prefill_input_info)
        dec_compile_config = RBLNCompileConfig(compiled_model_name="decoder", input_info=dec_input_info)

        rbln_config = RBLNConfig(
            rbln_cls=cls.__name__,
            compile_cfgs=[prefill_compile_config, dec_compile_config],
            rbln_kwargs=rbln_kwargs,
        )

        rbln_config.model_cfg.update(
            {
                "max_seq_len": rbln_max_seq_len,
                "batch_size": rbln_batch_size,
                "prefill_chunk_size": prefill_chunk_size,
                "use_inputs_embeds": rbln_use_inputs_embeds,
                "kvcache_partition_len": rbln_kvcache_partition_len,
                "attn_impl": rbln_attn_impl,
            }
        )

        if rbln_quantization is not None:
            rbln_config.model_cfg.update({"quantization": rbln_quantization})

        return rbln_config

    @classmethod
    def _create_runtimes(
        cls,
        compiled_models: List[rebel.RBLNCompiledModel],
        rbln_device_map: Dict[str, int],
        activate_profiler: Optional[bool] = None,
    ) -> List[rebel.Runtime]:
        if any(model_name not in rbln_device_map for model_name in ["prefill", "decoder"]):
            cls._raise_missing_compiled_file_error(["prefill", "decoder"])

        return [
            compiled_models[0].create_runtime(
                tensor_type="pt", device=rbln_device_map["prefill"], activate_profiler=activate_profiler
            ),
            compiled_models[1].create_runtime(
                tensor_type="pt", device=rbln_device_map["decoder"], activate_profiler=activate_profiler
            ),
        ]

    def get_decoder(self):
        return self.decoder

    def can_generate(self):
        return True

    def _reorder_cache(self, past_key_values, beam_idx):
        raise NotImplementedError

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        generate_idx: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        model_inputs = {}
        is_prefill_phase = generate_idx is None

        if is_prefill_phase:
            generate_idx = attention_mask.sum(dim=-1, keepdim=True).int()
            cache_position = None
        else:
            if inputs_embeds is not None:
                raise NotImplementedError("Specifying inputs_embeds in decoder phase is not supported.")

            input_ids = input_ids[:, -1:]
            cache_position = generate_idx
            generate_idx = generate_idx + 1
            model_inputs.update({"input_ids": input_ids})

        if inputs_embeds is not None:
            if self.rbln_config.model_cfg["use_inputs_embeds"]:
                model_inputs.update({"inputs_embeds": inputs_embeds})
            else:
                raise ValueError(
                    "The specifying inputs_embedst is only supported when using a compiled RBLN model with 'rbln_use_inputs_embeds' set to True."
                )
        else:
            model_inputs.update({"input_ids": input_ids})

        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "generate_idx": generate_idx,
            }
        )

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

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        generate_idx: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor]:
        # prefll
        if cache_position is None:
            logits = []
            input_tensors = inputs_embeds if inputs_embeds is not None else input_ids
            batch_size = input_tensors.shape[0]

            for b_idx in range(batch_size):
                # Transform inputs as vllm format
                if attention_mask is not None:
                    input_tensor = input_tensors[b_idx : b_idx + 1, attention_mask[b_idx].bool()]
                else:
                    input_tensor = input_tensors[b_idx : b_idx + 1]

                cache_position = torch.arange(0, generate_idx[b_idx].item(), dtype=torch.int32).unsqueeze(0)

                logit = self._forward_prefill(
                    input_ids=input_tensor if inputs_embeds is None else None,
                    inputs_embeds=input_tensor if inputs_embeds is not None else None,
                    cache_position=cache_position,
                    batch_idx=b_idx,
                )
                logits.append(logit)
            logits = torch.cat(logits, dim=0)
        # decoder
        else:
            logits = self._forward_decoder(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
            )

        return RBLNDecoderOnlyOutput(
            logits=logits,
            generate_idx=generate_idx,
        )

    def _forward_prefill(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: torch.Tensor = None,
        cache_position: torch.Tensor = None,
        batch_idx: int = None,
    ) -> torch.FloatTensor:
        if batch_idx is None or batch_idx >= self.batch_size:
            raise RuntimeError(
                f"Invalid batch_idx ({batch_idx}). It must be a non-null value less than the batch size ({self.batch_size})."
            )

        out_buffers = [
            torch.empty(
                size=[
                    1,
                    1,
                    self.config.vocab_size,
                ],
                dtype=torch.float32,
                device="cpu",
            )
        ]

        input_tensors = inputs_embeds if inputs_embeds is not None else input_ids
        query_length = input_tensors.shape[1]
        if query_length > self.max_seq_len:
            raise ValueError(
                f"Input length ({query_length}) exceeds the maximum allowed sequence length ({self.max_seq_len})."
            )

        _attention_mask = self.prefill_attention_mask.clone()

        for step in range(0, query_length, self.prefill_chunk_size):
            # pad input_tensors & cache_position for prefill_chunk
            if (step + self.prefill_chunk_size) > query_length:
                pad_to_chunk = step + self.prefill_chunk_size - query_length
                if inputs_embeds is not None:
                    input_tensors = torch.nn.functional.pad(input_tensors, (0, 0, 0, pad_to_chunk))
                else:
                    input_tensors = torch.nn.functional.pad(input_tensors, (0, pad_to_chunk))

                cache_position = torch.cat(
                    [
                        cache_position,
                        torch.arange(
                            query_length,
                            step + self.prefill_chunk_size,
                            dtype=torch.int32,
                        ).unsqueeze(0),
                    ],
                    dim=-1,
                )

            # slice input_tensor & cache_position with prefill_chunk_size
            _input_tensors = input_tensors[:, step : step + self.prefill_chunk_size]
            _cache_position = cache_position[:, step : step + self.prefill_chunk_size]

            # update attention_mask
            if step >= self.prefill_chunk_size:
                _attention_mask[:, :, :, step - self.prefill_chunk_size : step] = 1
            _attention_mask[:, :, :, step : step + self.prefill_chunk_size] = self.causal_mask

            query_position = (query_length - 1) % self.prefill_chunk_size

            logits = self.prefill_decoder(
                input_ids=_input_tensors.contiguous() if inputs_embeds is None else None,
                inputs_embeds=_input_tensors.contiguous() if inputs_embeds is not None else None,
                attention_mask=_attention_mask.contiguous(),
                cache_position=_cache_position.contiguous(),
                batch_position=torch.tensor(batch_idx, dtype=torch.int16),
                query_position=torch.tensor(query_position, dtype=torch.int16),
                out=out_buffers,
            )

        # update decoder_attn_mask with preprocessed kv-cache length in prefill phase
        self.dec_attn_mask[batch_idx] = self.dec_attn_mask_init.clone()
        self.dec_attn_mask[batch_idx, :, :, :query_length] = 1

        return logits

    def _forward_decoder(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: torch.Tensor = None,
        cache_position: torch.Tensor = None,
    ) -> torch.FloatTensor:
        input_tensors = inputs_embeds if inputs_embeds is not None else input_ids
        if input_tensors is None:
            raise ValueError("Either `input_ids` or `inputs_embeds` must be provided.")

        batch_size = input_tensors.shape[0]
        if batch_size != self.batch_size:
            raise RuntimeError(
                f"Batch size mismatch: got {batch_size}, expected {self.batch_size} (compiled batch size)."
            )

        if batch_size != cache_position.shape[0]:
            raise RuntimeError(f"Cache position size mismatch: got {cache_position.shape[0]}, expected {batch_size}.")

        for b_idx in range(batch_size):
            decoding_step = cache_position[b_idx].item()
            if not (0 <= decoding_step < self.dec_attn_mask.shape[-1]):
                raise ValueError(
                    f"Decoding step {decoding_step} out of bounds for attention mask with shape {self.dec_attn_mask.shape}."
                )
            self.dec_attn_mask[b_idx, :, :, decoding_step] = 1
        logits = self.decoder(
            input_ids=input_tensors.contiguous() if inputs_embeds is None else None,
            inputs_embeds=input_tensors.contiguous() if inputs_embeds is not None else None,
            attention_mask=self.dec_attn_mask.contiguous(),
            cache_position=cache_position.contiguous(),
        )

        return logits
