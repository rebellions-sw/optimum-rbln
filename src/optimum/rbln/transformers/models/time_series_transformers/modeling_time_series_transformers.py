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
import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import rebel
import torch
from rebel.compile_context import CompileContext
from transformers import (
    PretrainedConfig,
    TimeSeriesTransformerForPrediction,
)
from transformers.modeling_outputs import SampleTSPredictionOutput

from ....modeling import RBLNModel
from ....modeling_config import RBLNCompileConfig, RBLNConfig
from ....utils.runtime_utils import RBLNPytorchRuntime
from .time_series_transformers_architecture import TimeSeriesTransformersWrapper

# from .generation_whisper import RBLNWhisperGenerationMixin
# from .whisper_architecture import WhisperWrapper


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel, AutoFeatureExtractor, AutoProcessor, AutoTokenizer


class RBLNRuntimeEncoder(RBLNPytorchRuntime):
    mandatory_members = ["main_input_name"]

    def forward(self, inputs_embeds: torch.Tensor = None, attention_mask: torch.Tensor = None):
        _ = super().forward(inputs_embeds)


class RBLNRuntimeDecoder(RBLNPytorchRuntime):
    mandatory_members = ["main_input_name"]

    def forward(
        self,
        inputs_embeds: torch.Tensor = None,
        # decoder_attention_mask: torch.Tensor = None,
        # encoder_attention_mask: torch.Tensor = None,
        cache_position: torch.Tensor = None,
    ):
        pass


class RBLNTimeSeriesTransformerForPrediction(RBLNModel):
    auto_model_class = None  # TODO
    main_input_name = "inputs_embeds"

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        pass
        self.batch_size = self.rbln_config.model_cfg["batch_size"]
        self.dec_max_seq_len = self.rbln_config.model_cfg["num_parallel_samples"]

        self.encoder = RBLNRuntimeEncoder(
            runtime=self.model[0],
            main_input_name="inputs_embeds",  # , batch_size=self.batch_size
        )
        self.decoder = RBLNRuntimeDecoder(
            runtime=self.model[1],
            main_input_name="inputs_embeds",  # , batch_size=self.batch_size
        )

    # def get_encoder(self):
    #     return self.encoder

    # def get_decoder(self):
    #     return self.decoder

    def __getattr__(self, __name: str) -> Any:
        """This is the key method to implement RBLN-TimeSeriesTransformersForPrediction.
        Returns:
            Any: Whisper's corresponding method
        """

        def redirect(func):
            return lambda *pargs, **kwargs: func(self, *pargs, **kwargs)

        val = getattr(TimeSeriesTransformerForPrediction, __name)
        if isinstance(val, Callable) and "self" in set(inspect.signature(val).parameters):
            return redirect(val)

    @classmethod
    def wrap_model_if_needed(self, model: "PreTrainedModel", rbln_config: "RBLNConfig"):
        return TimeSeriesTransformersWrapper(model)

    @classmethod
    @torch.inference_mode()
    def get_compiled_model(cls, model, rbln_config: RBLNConfig):
        wrapped_model = cls.wrap_model_if_needed(model, rbln_config)

        enc_compile_config = rbln_config.compile_cfgs[0]
        dec_compile_config = rbln_config.compile_cfgs[1]

        context = CompileContext(use_weight_sharing=False)

        enc_example_inputs = enc_compile_config.get_dummy_inputs(fill=0)

        # Mark encoder's static tensors (cross kv states)
        static_tensors = {}
        for (name, _, _), tensor in zip(enc_compile_config.input_info, enc_example_inputs):
            if "key_value_states" in name:
                static_tensors[name] = tensor
                context.mark_static_address(tensor)

        dec_example_inputs = dec_compile_config.get_dummy_inputs(fill=0, static_tensors=static_tensors)

        # Mark decoder's static tensors (self kv states)
        for (name, _, _), tensor in zip(dec_compile_config.input_info, dec_example_inputs):
            if "key_value_states" in name:
                context.mark_static_address(tensor)

        compiled_encoder = super().compile(
            wrapped_model.encoder,
            enc_compile_config,
            example_inputs=enc_example_inputs,
            compile_context=context,
        )
        compiled_decoder = super().compile(
            wrapped_model.decoder,
            dec_compile_config,
            example_inputs=dec_example_inputs,
            compile_context=context,
        )

        return {"encoder": compiled_encoder, "decoder": compiled_decoder}

    @classmethod
    def _get_rbln_config(
        cls,
        preprocessors: Optional[Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"]],
        model_config: "PretrainedConfig",
        rbln_kwargs: Dict[str, Any] = {},
    ) -> RBLNConfig:
        rbln_batch_size = rbln_kwargs.get("batch_size", None)
        rbln_num_parallel_samples = rbln_kwargs.get("num_parallel_samples", None)

        rbln_batch_size = 1 if rbln_batch_size is None else rbln_batch_size
        rbln_num_parallel_samples = (
            model_config.num_parallel_samples if rbln_num_parallel_samples is None else rbln_num_parallel_samples
        )

        if not isinstance(rbln_batch_size, int):
            raise TypeError(f"Expected rbln_batch_size to be an int, but got {type(rbln_batch_size)}")

        if not isinstance(rbln_num_parallel_samples, int):
            raise TypeError(f"Expected rbln_num_parallel_samples to be an int, but got {type(rbln_batch_size)}")

        context_length = model_config.context_length  # enc_max_seq_len
        predict_length = model_config.prediction_length  # dec_max_seq_len

        # d_model = model_config.d_model
        feature_size = model_config.feature_size

        # model input info
        enc_input_info = [
            ("inputs_embeds", [rbln_batch_size, context_length, feature_size], "float32"),
            # ("attention_mask", [rbln_batch_size, context_length], "float32"),
        ]
        enc_input_info.extend(
            [
                (
                    "cross_key_value_states",
                    [
                        model_config.decoder_layers * 2,
                        rbln_batch_size,
                        model_config.decoder_attention_heads,
                        context_length,
                        model_config.d_model // model_config.decoder_attention_heads,
                    ],
                    "float32",
                )
            ]
        )

        dec_input_info = [
            ("inputs_embeds", [rbln_batch_size * rbln_num_parallel_samples, predict_length, feature_size], "float32"),
            # ("attention_mask", [rbln_batch_size * rbln_num_parallel_samples, predict_length], "float32"),
            # ("encoder_attention_mask", [rbln_batch_size, context_length], "float32"),
            ("cache_position", [], "int32"),
        ]
        dec_input_info.extend(
            [
                (
                    "cross_key_value_states",
                    [
                        model_config.decoder_layers * 2,  # 4
                        rbln_batch_size,  # 64
                        model_config.decoder_attention_heads,  # 2
                        context_length,  # 24
                        model_config.d_model // model_config.decoder_attention_heads,  # 13
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
                        rbln_batch_size * rbln_num_parallel_samples,
                        model_config.decoder_attention_heads,
                        predict_length,
                        model_config.d_model // model_config.encoder_attention_heads,
                    ],
                    "float32",
                )
                for i in range(model_config.decoder_layers * 2)
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
                "batch_size": rbln_batch_size,
                "num_parallel_samples": rbln_num_parallel_samples,
            }
        )

        return rbln_config

    @classmethod
    def _create_runtimes(
        cls,
        compiled_models: List[rebel.RBLNCompiledModel],
        rbln_device_map: Dict[str, int],
        activate_profiler: Optional[bool] = None,
    ) -> List[rebel.Runtime]:
        if any(model_name not in rbln_device_map for model_name in ["encoder", "decoder"]):
            cls._raise_missing_compiled_file_error(["encoder", "decoder"])

        return [
            compiled_models[0].create_runtime(
                tensor_type="pt", device=rbln_device_map["encoder"], activate_profiler=activate_profiler
            ),
            compiled_models[1].create_runtime(
                tensor_type="pt", device=rbln_device_map["decoder"], activate_profiler=activate_profiler
            ),
        ]

    # def forward(
    #     self,
    #     input_ids: Optional[torch.LongTensor] = None,
    #     cache_position: Optional[torch.Tensor] = None,
    #     input_features: Optional[torch.Tensor] = None,
    #     decoder_input_ids: Optional[torch.Tensor] = None,
    #     encoder_outputs: Optional[Seq2SeqLMOutput] = None,
    #     **kwargs,
    # ) -> Seq2SeqLMOutput:
    #     pass

    @torch.no_grad()
    def generate(
        self,
        past_values: torch.Tensor,
        past_time_features: torch.Tensor,
        future_time_features: torch.Tensor,
        past_observed_mask: Optional[torch.Tensor] = None,
        static_categorical_features: Optional[torch.Tensor] = None,
        static_real_features: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> SampleTSPredictionOutput:
        outputs = self(
            static_categorical_features=static_categorical_features,
            static_real_features=static_real_features,
            past_time_features=past_time_features,
            past_values=past_values,
            past_observed_mask=past_observed_mask,
            future_time_features=future_time_features,
            future_values=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            use_cache=True,
        )

        decoder = self.model.get_decoder()
        enc_last_hidden = outputs.encoder_last_hidden_state
        loc = outputs.loc
        scale = outputs.scale
        static_feat = outputs.static_features

        num_parallel_samples = self.config.num_parallel_samples
        repeated_loc = loc.repeat_interleave(repeats=num_parallel_samples, dim=0)
        repeated_scale = scale.repeat_interleave(repeats=num_parallel_samples, dim=0)

        repeated_past_values = (
            past_values.repeat_interleave(repeats=num_parallel_samples, dim=0) - repeated_loc
        ) / repeated_scale

        expanded_static_feat = static_feat.unsqueeze(1).expand(-1, future_time_features.shape[1], -1)
        features = torch.cat((expanded_static_feat, future_time_features), dim=-1)
        repeated_features = features.repeat_interleave(repeats=num_parallel_samples, dim=0)

        repeated_enc_last_hidden = enc_last_hidden.repeat_interleave(repeats=num_parallel_samples, dim=0)

        future_samples = []

        # greedy decoding
        for k in range(self.config.prediction_length):
            lagged_sequence = self.model.get_lagged_subsequences(
                sequence=repeated_past_values,
                subsequences_length=1 + k,
                shift=1,
            )

            lags_shape = lagged_sequence.shape
            reshaped_lagged_sequence = lagged_sequence.reshape(lags_shape[0], lags_shape[1], -1)

            decoder_input = torch.cat((reshaped_lagged_sequence, repeated_features[:, : k + 1]), dim=-1)

            print(f"decoder_input_shape = {decoder_input.shape}")

            dec_output = decoder(inputs_embeds=decoder_input, encoder_hidden_states=repeated_enc_last_hidden)
            dec_last_hidden = dec_output.last_hidden_state

            params = self.parameter_projection(dec_last_hidden[:, -1:])
            distr = self.output_distribution(params, loc=repeated_loc, scale=repeated_scale)
            next_sample = distr.sample()

            repeated_past_values = torch.cat(
                (repeated_past_values, (next_sample - repeated_loc) / repeated_scale), dim=1
            )
            future_samples.append(next_sample)

        concat_future_samples = torch.cat(future_samples, dim=1)

        return SampleTSPredictionOutput(
            sequences=concat_future_samples.reshape(
                (-1, num_parallel_samples, self.config.prediction_length) + self.target_shape,
            )
        )
