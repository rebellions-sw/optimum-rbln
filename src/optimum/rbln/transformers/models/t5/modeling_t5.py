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
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import rebel
import torch
from transformers import (
    AutoModelForTextEncoding,
    PretrainedConfig,
    T5EncoderModel,
    T5ForConditionalGeneration,
)
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput

from ....diffusers.modeling_diffusers import RBLNDiffusionMixin
from ....modeling import RBLNModel
from ....modeling_config import RBLNCompileConfig, RBLNConfig
from ....utils.logging import get_logger
from ....utils.runtime_utils import RBLNPytorchRuntime
from ...models.seq2seq import RBLNModelForSeq2SeqLM
from .t5_architecture import T5Wrapper


logger = get_logger()

if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer, PreTrainedModel


class RBLNRuntimeModel(RBLNPytorchRuntime):
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor,
        head_mask: torch.FloatTensor,
        inputs_embeds: torch.FloatTensor,
        **kwargs,
    ):
        return super().forward(
            input_ids,
            attention_mask,
            head_mask,
            inputs_embeds,
            **kwargs,
        )


class RBLNRuntimeEncoder(RBLNPytorchRuntime):
    mandatory_members = ["main_input_name"]

    def forward(self, *args: List[torch.Tensor], **kwargs: Dict[str, torch.Tensor]):
        _ = super().forward(*args, **kwargs)
        return BaseModelOutput(last_hidden_state=torch.tensor([1.0]))


class RBLNRuntimeDecoder(RBLNPytorchRuntime):
    mandatory_members = ["main_input_name"]

    def __init__(
        self,
        runtime: rebel.Runtime,
        batch_size: int,
        dec_max_seq_len: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(runtime, **kwargs)
        self.batch_size = batch_size
        self.dec_max_seq_len = dec_max_seq_len

    def forward(
        self,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor]:
        batch_size = decoder_input_ids.shape[0]
        if batch_size != self.batch_size:
            raise RuntimeError(
                f"Batch size mismatch: got {batch_size}, expected {self.batch_size} (compiled batch size)."
            )

        if batch_size != cache_position.shape[0]:
            raise RuntimeError(f"Cache position size mismatch: got {cache_position.shape[0]}, expected {batch_size}.")

        for b_idx in range(self.batch_size):
            decoding_step = cache_position[b_idx].item()
            if not (0 <= decoding_step < self.dec_max_seq_len):
                raise ValueError(
                    f"Decoding step {decoding_step} out of bounds for attention mask with shape {self.dec_attn_mask.shape}."
                )
            decoder_attention_mask[b_idx, : decoding_step + 1] = 1

        lm_logits = super().forward(
            decoder_input_ids,
            decoder_attention_mask,
            attention_mask,
            cache_position,
        )

        return Seq2SeqLMOutput(logits=lm_logits)


class T5EncoderWrapper(torch.nn.Module):
    def __init__(self, model: "T5EncoderModel") -> None:
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        kwargs.pop("return_dict", None)
        return self.model(*args, **kwargs, return_dict=False)


class RBLNT5EncoderModel(RBLNModel):
    auto_model_class = AutoModelForTextEncoding
    rbln_model_input_names = ["input_ids", "attention_mask"]

    def __post_init__(self, **kwargs):
        self.model = RBLNRuntimeModel(runtime=self.model[0])

    @classmethod
    def wrap_model_if_needed(self, model: "PreTrainedModel", rbln_config: "RBLNConfig"):
        return T5EncoderWrapper(model)

    @classmethod
    def update_rbln_config_using_pipe(cls, pipe: RBLNDiffusionMixin, rbln_config: Dict[str, Any]) -> Dict[str, Any]:
        batch_size = rbln_config.get("batch_size", 1)
        max_sequence_length = rbln_config.get("max_sequence_length", 256)
        model_input_names = ["input_ids"]

        rbln_config.update(
            {
                "batch_size": batch_size,
                "max_seq_len": max_sequence_length,
                "model_input_names": model_input_names,
            }
        )

        return rbln_config

    @classmethod
    def _get_rbln_config(
        cls,
        preprocessors: Optional[Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"]],
        model_config: Optional["PretrainedConfig"] = None,
        rbln_kwargs: Dict[str, Any] = {},
    ) -> RBLNConfig:
        rbln_max_seq_len = rbln_kwargs.get("max_seq_len", None)
        rbln_model_input_names = rbln_kwargs.get("model_input_names", None)
        rbln_batch_size = rbln_kwargs.get("batch_size", None)

        max_position_embeddings = getattr(model_config, "n_positions", None)

        if rbln_max_seq_len is None:
            rbln_max_seq_len = max_position_embeddings
            if rbln_max_seq_len is None:
                for tokenizer in preprocessors:
                    if hasattr(tokenizer, "model_max_length"):
                        rbln_max_seq_len = tokenizer.model_max_length
                        break
                if rbln_max_seq_len is None:
                    raise ValueError("`rbln_max_seq_len` should be specified!")

        if max_position_embeddings is not None and rbln_max_seq_len > max_position_embeddings:
            raise ValueError("`rbln_max_seq_len` should be less or equal than max_position_embeddings!")

        signature_params = inspect.signature(cls.get_hf_class().forward).parameters.keys()

        if rbln_model_input_names is None:
            for tokenizer in preprocessors:
                if hasattr(tokenizer, "model_input_names"):
                    rbln_model_input_names = [name for name in signature_params if name in tokenizer.model_input_names]

                    invalid_params = set(rbln_model_input_names) - set(signature_params)
                    if invalid_params:
                        raise ValueError(f"Invalid model input names: {invalid_params}")
                    break
            if rbln_model_input_names is None and hasattr(cls, "rbln_model_input_names"):
                rbln_model_input_names = cls.rbln_model_input_names
            elif rbln_model_input_names is None and hasattr(cls, "rbln_model_input_names") is False:
                raise ValueError(
                    "Specify the model input names obtained by the tokenizer via `rbln_model_input_names`, "
                    f"and be sure to make the order of the inputs same as T5EncoderModel forward() arguments like ({list(signature_params)})"
                )
        else:
            invalid_params = set(rbln_model_input_names) - set(signature_params)
            if invalid_params:
                raise ValueError(f"Invalid model input names: {invalid_params}")
            rbln_model_input_names = [name for name in signature_params if name in rbln_model_input_names]

        if rbln_batch_size is None:
            rbln_batch_size = 1

        input_info = [
            (model_input_name, [rbln_batch_size, rbln_max_seq_len], "int64")
            for model_input_name in rbln_model_input_names
        ]

        rbln_compile_config = RBLNCompileConfig(input_info=input_info)

        rbln_config = RBLNConfig(
            rbln_cls=cls.__name__,
            compile_cfgs=[rbln_compile_config],
            rbln_kwargs=rbln_kwargs,
        )

        rbln_config.model_cfg.update({"max_seq_len": rbln_max_seq_len})
        return rbln_config

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], BaseModelOutput]:
        encoder_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if not return_dict:
            return (encoder_outputs,)
        else:
            return BaseModelOutput(last_hidden_state=encoder_outputs)


class RBLNT5ForConditionalGeneration(RBLNModelForSeq2SeqLM):
    def __post_init__(self, **kwargs):
        batch_size = self.rbln_config.model_cfg["batch_size"]
        dec_max_seq_len = self.rbln_config.model_cfg["dec_max_seq_len"]

        self.encoder = RBLNRuntimeEncoder(
            runtime=self.model[0],
            main_input_name="input_ids",
        )
        self.decoder = RBLNRuntimeDecoder(
            runtime=self.model[1],
            main_input_name="input_ids",
            batch_size=batch_size,
            dec_max_seq_len=dec_max_seq_len,
        )

    @classmethod
    def wrap_model_if_needed(self, model: "PreTrainedModel", rbln_config: "RBLNConfig"):
        enc_max_seq_len = rbln_config.model_cfg["enc_max_seq_len"]
        dec_max_seq_len = rbln_config.model_cfg["dec_max_seq_len"]

        return T5Wrapper(model, enc_max_seq_len=enc_max_seq_len, dec_max_seq_len=dec_max_seq_len)

    def __getattr__(self, __name: str) -> Any:
        def redirect(func):
            return lambda *pargs, **kwargs: func(self, *pargs, **kwargs)

        val = getattr(T5ForConditionalGeneration, __name)

        if isinstance(val, Callable) and "self" in set(inspect.signature(val).parameters):
            return redirect(val)

        return val

    @classmethod
    def _get_rbln_config(
        cls,
        preprocessors: Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"],
        model_config: "PretrainedConfig",
        rbln_kwargs: Dict[str, Any] = {},
    ) -> RBLNConfig:
        rbln_enc_max_seq_len = rbln_kwargs.get("enc_max_seq_len", None)
        rbln_dec_max_seq_len = rbln_kwargs.get("dec_max_seq_len", None)
        rbln_batch_size = rbln_kwargs.get("batch_size", None)
        rbln_batch_size = 1 if rbln_batch_size is None else rbln_batch_size

        n_layer = getattr(model_config, "decoder_layers", None) or getattr(model_config, "num_layers")
        n_head = getattr(model_config, "decoder_attention_heads", None) or getattr(model_config, "num_heads")
        d_kv = (
            model_config.d_kv
            if hasattr(model_config, "d_kv")
            else model_config.d_model // model_config.encoder_attention_heads
        )

        max_position_embeddings = getattr(model_config, "n_positions", None) or getattr(
            model_config, "max_position_embeddings", None
        )

        rbln_pad_token_id = getattr(model_config, "pad_token_id", None)
        if rbln_pad_token_id is None:
            rbln_pad_token_id = getattr(model_config, "bos_token_id", None)
            if rbln_pad_token_id is None:
                rbln_pad_token_id = getattr(model_config, "eos_token_id", None)
                if rbln_pad_token_id is None:
                    rbln_pad_token_id = -1

        if rbln_enc_max_seq_len is None:
            rbln_enc_max_seq_len = max_position_embeddings
            if rbln_enc_max_seq_len is None:
                for tokenizer in preprocessors:
                    if hasattr(tokenizer, "model_max_length"):
                        rbln_enc_max_seq_len = tokenizer.model_max_length
                        break
                if rbln_enc_max_seq_len is None:
                    raise ValueError("`rbln_enc_max_seq_len` should be specified!")
        if max_position_embeddings is not None and rbln_enc_max_seq_len > max_position_embeddings:
            raise ValueError("`rbln_enc_max_seq_len` should be less or equal than max_position_embeddings!")

        if rbln_dec_max_seq_len is None:
            rbln_dec_max_seq_len = max_position_embeddings
            if rbln_dec_max_seq_len is None:
                for tokenizer in preprocessors:
                    if hasattr(tokenizer, "model_max_length"):
                        rbln_dec_max_seq_len = tokenizer.model_max_length
                        break
                if rbln_dec_max_seq_len is None:
                    raise ValueError("`rbln_dec_max_seq_len` should be specified!")

        if max_position_embeddings is not None and rbln_dec_max_seq_len > max_position_embeddings:
            raise ValueError("`rbln_dec_max_seq_len` should be less or equal than max_position_embeddings!")

        # model input info
        enc_input_info = [
            ("input_ids", [1, rbln_enc_max_seq_len], "int64"),
            ("attention_mask", [1, rbln_enc_max_seq_len], "float32"),
            (
                "cross_key_value_states",
                [
                    n_layer * 2,
                    rbln_batch_size,
                    n_head,
                    rbln_enc_max_seq_len,
                    d_kv,
                ],
                "float32",
            ),
            ("batch_position", [1], "int16"),
        ]

        dec_input_info = [
            ("input_ids", [rbln_batch_size, 1], "int64"),
            ("attention_mask", [rbln_batch_size, rbln_dec_max_seq_len], "float32"),
            ("encoder_attention_mask", [rbln_batch_size, rbln_enc_max_seq_len], "float32"),
            (
                "cache_position",
                [rbln_batch_size, 1],
                "int32",
            ),
        ]
        dec_input_info.extend(
            [
                (
                    "cross_key_value_states",
                    [
                        n_layer * 2,
                        rbln_batch_size,
                        n_head,
                        rbln_enc_max_seq_len,
                        d_kv,
                    ],
                    "float32",
                )
            ]
        )
        dec_input_info.extend(
            [
                (
                    f"self_key_value_states_{i}",
                    [
                        rbln_batch_size,
                        n_head,
                        rbln_dec_max_seq_len,
                        d_kv,
                    ],
                    "float32",
                )
                for i in range(n_layer * 2)
            ]
        )

        enc_compile_config = RBLNCompileConfig(compiled_model_name="encoder", input_info=enc_input_info)
        dec_compile_config = RBLNCompileConfig(compiled_model_name="decoder", input_info=dec_input_info)

        rbln_config = RBLNConfig(
            rbln_cls=cls.__name__,
            compile_cfgs=[enc_compile_config, dec_compile_config],
            rbln_kwargs=rbln_kwargs,
        )

        rbln_config.model_cfg.update(
            {
                "enc_max_seq_len": rbln_enc_max_seq_len,
                "dec_max_seq_len": rbln_dec_max_seq_len,
                "batch_size": rbln_batch_size,
                "pad_token_id": rbln_pad_token_id,
            }
        )

        return rbln_config
