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
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Union

from transformers import BartForConditionalGeneration, PretrainedConfig, PreTrainedModel

from ....modeling import RBLNModel
from ....modeling_config import RBLNCompileConfig, RBLNConfig
from ....utils.logging import get_logger
from ...models.seq2seq import RBLNModelForSeq2SeqLM
from .bart_architecture import BartWrapper


logger = get_logger()


if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer, PreTrainedModel


class RBLNBartModel(RBLNModel):
    @classmethod
    def _get_rbln_config(
        cls,
        preprocessors: Optional[Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"]],
        model_config: Optional["PretrainedConfig"] = None,
        rbln_kwargs: Dict[str, Any] = {},
    ) -> RBLNConfig:
        rbln_max_seq_len = rbln_kwargs.get("max_seq_len", None)
        rbln_batch_size = rbln_kwargs.get("batch_size", None)
        rbln_model_input_names = rbln_kwargs.get("model_input_names", None)

        max_position_embeddings = getattr(model_config, "max_position_embeddings", None)

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
                    # BartModel's forward() does not take token_type_ids as input.
                    # (Added because some of the tokenizers includes 'token_type_ids')
                    if "token_type_ids" in rbln_model_input_names:
                        rbln_model_input_names.remove("token_type_ids")

                    invalid_params = set(rbln_model_input_names) - set(signature_params)
                    if invalid_params:
                        raise ValueError(f"Invalid model input names: {invalid_params}")
                    break
            if rbln_model_input_names is None and hasattr(cls, "rbln_model_input_names"):
                rbln_model_input_names = cls.rbln_model_input_names
            elif rbln_model_input_names is None and hasattr(cls, "rbln_model_input_names") is False:
                raise ValueError(
                    "Specify the model input names obtained by the tokenizer via `rbln_model_input_names`, "
                    f"and be sure to make the order of the inputs same as BartModel forward() arguments like ({list(signature_params)})"
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

        enc_compile_config = RBLNCompileConfig(input_info=input_info, compiled_model_name="encoder")
        dec_compile_config = RBLNCompileConfig(input_info=input_info, compiled_model_name="decoder")

        rbln_config = RBLNConfig(
            rbln_cls=cls.__name__,
            compile_cfgs=[enc_compile_config, dec_compile_config],
            rbln_kwargs=rbln_kwargs,
        )

        rbln_config.model_cfg.update({"max_seq_len": rbln_max_seq_len})
        return rbln_config


class RBLNBartForConditionalGeneration(RBLNModelForSeq2SeqLM):
    support_paged_causal_attn = True

    @classmethod
    def wrap_model_if_needed(self, model: "PreTrainedModel", rbln_config: "RBLNConfig"):
        enc_max_seq_len = (
            rbln_config.model_cfg["enc_max_seq_len"] if "enc_max_seq_len" in rbln_config.model_cfg else 1024
        )
        use_attention_mask = rbln_config.model_cfg.get("use_attention_mask", False)

        return BartWrapper(model, enc_max_seq_len=enc_max_seq_len, use_attention_mask=use_attention_mask)

    def __getattr__(self, __name: str) -> Any:
        def redirect(func):
            return lambda *pargs, **kwargs: func(self, *pargs, **kwargs)

        val = getattr(BartForConditionalGeneration, __name)

        if isinstance(val, Callable) and "self" in set(inspect.signature(val).parameters):
            return redirect(val)

        return val
