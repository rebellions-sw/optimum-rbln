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
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import rebel
import torch
from rebel.compile_context import CompileContext
from transformers import (
    AutoModelForVision2Seq,
    Idefics3Config,
    Idefics3ForConditionalGeneration,
    Idefics3VisionConfig,
    Idefics3VisionTransformer,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import no_init_weights
from transformers.models.idefics3.modeling_idefics3 import (
    Idefics3CausalLMOutputWithPast,
)

from ....modeling import RBLNModel
from ....modeling_config import RBLNCompileConfig, RBLNConfig
from ....utils.logging import get_logger
from ....utils.runtime_utils import RBLNPytorchRuntime
from ..decoderonly.decoderonly_architecture import validate_attention_method
from ..decoderonly.modeling_decoderonly import (
    DecoderOnlyWrapper,
    RBLNDecoderOnlyModelForCausalLM,
    RBLNDecoderOnlyOutput,
    RBLNRuntimeModel,
)


logger = get_logger(__name__)

if TYPE_CHECKING:
    from transformers import (
        AutoFeatureExtractor,
        AutoProcessor,
        AutoTokenizer,
        PretrainedConfig,
    )


class RBLNRuntimeVisionModel(RBLNPytorchRuntime):
    mandatory_members = ["main_input_name"]

    def __init__(
        self,
        runtime: rebel.Runtime,
        model: Idefics3VisionTransformer,
        config: Idefics3VisionConfig,
        **kwargs: Any,
    ) -> None:
        super().__init__(runtime, **kwargs)
        self.base_model = model
        self.patch_size = config.patch_size
        # self._use_flash_attention_2 = config.text_config._attn_implementation == "flash_attention_2"

    def forward(
        self,
        pixel_values,
        patch_attention_mask: Optional[torch.BoolTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        batch_size = pixel_values.size(0)
        if patch_attention_mask is None:
            patch_size = self.patch_size
            patch_attention_mask = torch.ones(
                (
                    batch_size,
                    pixel_values.size(2) // patch_size,
                    pixel_values.size(3) // patch_size,
                )
            )
            patch_attention_mask = patch_attention_mask.to(dtype=torch.bool, device=pixel_values.device)

        hidden_states = self.base_model.embeddings(
            pixel_values=pixel_values, patch_attention_mask=patch_attention_mask
        )
        patch_attention_mask = patch_attention_mask.view(batch_size, -1)

        if not torch.any(~patch_attention_mask):
            patch_attention_mask = None
        # elif not self._use_flash_attention_2:
        #     patch_attention_mask = _prepare_4d_attention_mask(patch_attention_mask, hidden_states.dtype)

        return super().forward(hidden_states.contiguous())


class _Idefics3VisionTransformer(torch.nn.Module):
    def __init__(self, model: "Idefics3VisionTransformer"):
        super().__init__()
        self.encoder = model.encoder
        self.post_layernorm = model.post_layernorm

    def forward(self, hidden_states, patch_attention_mask: Optional[torch.BoolTensor] = None):
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=patch_attention_mask,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=False,
        )
        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)
        return last_hidden_state


class RBLNIdefics3VisionTransformer(RBLNModel):
    def __post_init__(self, **kwargs):
        artifacts = torch.load(self.model_save_dir / self.subfolder / "torch_artifacts.pth", weights_only=False)
        with no_init_weights():
            self.base_model = Idefics3VisionTransformer._from_config(self.config)
        self.base_model.embeddings.load_state_dict(artifacts["embeddings"])
        self.model = RBLNRuntimeVisionModel(
            self.model[0], main_input_name="pixel_values", model=self.base_model, config=self.config
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
        save_dict = {}
        save_dict["embeddings"] = model.get_input_embeddings().state_dict()
        torch.save(save_dict, save_dir_path / subfolder / "torch_artifacts.pth")

    def get_input_embeddings(self):
        return self.embeddings

    @classmethod
    def wrap_model_if_needed(cls, model: torch.nn.Module, rbln_config: RBLNConfig) -> torch.nn.Module:
        return _Idefics3VisionTransformer(model).eval()

    @classmethod
    def _get_rbln_config(
        cls,
        preprocessors: Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"],
        model_config: "Idefics3VisionConfig",
        rbln_kwargs: Dict[str, Any] = {},
        rbln_batch_size: Optional[int] = None,
    ) -> RBLNConfig:
        rbln_batch_size = rbln_kwargs.get("batch_size", None)
        if rbln_batch_size is None:
            rbln_batch_size = 1

        input_info = [
            (
                "hidden_states",
                [
                    rbln_batch_size * 30,  # batch_size * num_images
                    (model_config.image_size // model_config.patch_size) ** 2,
                    model_config.hidden_size,
                ],
                "float32",
            ),
        ]

        rbln_compile_config = RBLNCompileConfig(input_info=input_info)
        rbln_config = RBLNConfig(
            rbln_cls=cls.__name__,
            compile_cfgs=[rbln_compile_config],
            rbln_kwargs=rbln_kwargs,
        )
        return rbln_config

    def forward(
        self,
        pixel_values,
        patch_attention_mask: Optional[torch.BoolTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        last_hidden_state = self.model(
            pixel_values, patch_attention_mask, output_attentions, output_hidden_states, return_dict=False
        )

        return BaseModelOutput(last_hidden_state=last_hidden_state)


# class RBLNIdefics3Connector(RBLNModel):
class RBLNIdefics3Model(RBLNModel):
    @classmethod
    def wrap_model_if_needed(cls, model: torch.nn.Module, rbln_config: RBLNConfig) -> torch.nn.Module:
        return model.connector.eval()

    @classmethod
    def _get_rbln_config(
        cls,
        preprocessors: Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"],
        model_config: "Idefics3Config",
        rbln_kwargs: Dict[str, Any] = {},
        rbln_batch_size: Optional[int] = None,
    ) -> RBLNConfig:
        rbln_batch_size = rbln_kwargs.get("batch_size", None)
        if rbln_batch_size is None:
            rbln_batch_size = 1

        model_config.return_dict = False
        input_info = [
            (
                "image_hidden_states",
                [
                    rbln_batch_size * 30,  # batch_size * num_images
                    (model_config.vision_config.image_size // model_config.vision_config.patch_size) ** 2,
                    model_config.vision_config.hidden_size,
                ],
                "float32",
            ),
        ]

        rbln_compile_config = RBLNCompileConfig(input_info=input_info)
        rbln_config = RBLNConfig(
            rbln_cls=cls.__name__,
            compile_cfgs=[rbln_compile_config],
            rbln_kwargs=rbln_kwargs,
        )
        return rbln_config

    def forward(self, image_hidden_states: "torch.Tensor", **kwargs):
        connector_output = super().forward(image_hidden_states)
        return connector_output


class Idefics3ModelWrapper(DecoderOnlyWrapper):
    pass


class RBLNIdefics3ForConditionalGeneration(RBLNDecoderOnlyModelForCausalLM):
    auto_model_class = AutoModelForVision2Seq
    _rbln_submodules = [{"name": "model.vision_model"}, {"name": "model"}]
    _decoder_wrapper_cls = Idefics3ModelWrapper

    def __getattr__(self, __name: str) -> Any:
        def redirect(func):
            return lambda *pargs, **kwargs: func(self, *pargs, **kwargs)

        val = getattr(Idefics3ForConditionalGeneration, __name)

        if isinstance(val, Callable) and "self" in set(inspect.signature(val).parameters):
            return redirect(val)
        return val

    def can_generate(self):
        return True

    def __post_init__(self, **kwargs):
        self.vision_model = self.rbln_submodules[0]
        self.connector = self.rbln_submodules[1]

        self.batch_size = self.rbln_config.model_cfg["batch_size"]
        self.max_seq_len = self.rbln_config.model_cfg["max_seq_len"]
        self.prefill_chunk_size = self.rbln_config.model_cfg["prefill_chunk_size"]
        self.kvcache_block_size = self.rbln_config.model_cfg["kvcache_block_size"]
        # FIXME get kvcache_num_blocks from compiled results.
        self.kvcache_num_blocks = self.rbln_config.model_cfg["kvcache_num_blocks"]
        self.use_attention_mask = self.rbln_config.model_cfg["use_attention_mask"]
        attn_impl = self.rbln_config.model_cfg["attn_impl"]
        main_input_name = self.main_input_name

        if self.rbln_config.model_cfg["use_inputs_embeds"]:
            main_input_name = "inputs_embeds"
            artifacts = torch.load(self.model_save_dir / self.subfolder / "torch_artifacts.pth", weights_only=False)
            with no_init_weights():
                self.embed_tokens = torch.nn.Embedding(
                    self.config.text_config.vocab_size,
                    self.config.text_config.hidden_size,
                    self.config.text_config.pad_token_id,
                )
            self.embed_tokens.load_state_dict(artifacts["embed_tokens"])
        else:
            self.embed_tokens = None

        # Initialize shared resources to be used across Runtime instances (prefill and decode phases)
        dec_attn_mask = torch.zeros(self.batch_size, 1, 1, self.max_seq_len, dtype=torch.float32)
        block_tables = torch.zeros(
            self.batch_size, self.max_seq_len // self.kvcache_block_size, dtype=torch.int16
        ).fill_(-1)
        free_block_pool = deque(x for x in range(self.kvcache_num_blocks))

        self.prefill_decoder = RBLNRuntimeModel(
            runtime=self.model[0],
            main_input_name=main_input_name,
            embed_tokens=self.embed_tokens,
            phase="prefill",
            batch_size=self.batch_size,
            dec_attn_mask=dec_attn_mask,
            block_tables=block_tables,
            free_block_pool=free_block_pool,
            kvcache_block_size=self.kvcache_block_size,
            vocab_size=self.config.text_config.vocab_size,
            prefill_chunk_size=self.prefill_chunk_size,
            max_seq_len=self.max_seq_len,
            use_attention_mask=self.use_attention_mask,
            attn_impl=attn_impl,
        )
        self.decoder = RBLNRuntimeModel(
            runtime=self.model[1],
            main_input_name=main_input_name,
            embed_tokens=self.embed_tokens,
            phase="decode",
            batch_size=self.batch_size,
            dec_attn_mask=dec_attn_mask,
            block_tables=block_tables,
            free_block_pool=free_block_pool,
            kvcache_block_size=self.kvcache_block_size,
            use_attention_mask=self.use_attention_mask,
            attn_impl=attn_impl,
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
            save_dict["embed_tokens"] = model.model.text_model.get_input_embeddings().state_dict()
            torch.save(save_dict, save_dir_path / subfolder / "torch_artifacts.pth")

    def get_input_embeddings(self):
        return self.embed_tokens

    @classmethod
    @torch.inference_mode()
    def get_compiled_model(cls, model: "PreTrainedModel", rbln_config: RBLNConfig):
        wrapped_model = cls.language_wrap_model(model, rbln_config)
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

        def compile_model(*args, **kwargs):
            try:
                original_linear = torch.nn.functional.linear
                torch.nn.functional.linear = torch.ops.rbln_custom_ops.linear
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
            finally:
                torch.nn.functional.linear = original_linear

        return compile_model()

    @classmethod
    def language_wrap_model(cls, model: "PreTrainedModel", rbln_config: RBLNConfig):
        wrapper_cfg = {"max_seq_len": rbln_config.model_cfg["max_seq_len"]}
        wrapper_cfg["attn_impl"] = rbln_config.model_cfg.get("attn_impl")
        wrapper_cfg["kvcache_partition_len"] = rbln_config.model_cfg.get("kvcache_partition_len")
        wrapper_cfg["kvcache_block_size"] = rbln_config.model_cfg.get("kvcache_block_size")
        wrapper_cfg["use_rotary_emb"] = rbln_config.model_cfg.get("use_inputs_embeds")
        wrapper_cfg["use_attention_mask"] = rbln_config.model_cfg.get("use_attention_mask")

        wrapped_language_model = Idefics3ModelWrapper(
            causal_lm=model.model.text_model, lm_head=model.lm_head, **wrapper_cfg
        ).eval()

        return wrapped_language_model

    @classmethod
    def _get_rbln_config(
        cls,
        preprocessors: Optional[Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"]],
        model_config: Optional["PretrainedConfig"] = None,
        rbln_kwargs={},
    ) -> RBLNConfig:
        rbln_max_seq_len = rbln_kwargs.get("max_seq_len", None)
        rbln_batch_size = rbln_kwargs.get("batch_size", None)
        rbln_use_inputs_embeds = rbln_kwargs.get("use_inputs_embeds", None)
        rbln_use_attention_mask = rbln_kwargs.get("use_attention_mask", None)
        rbln_attn_impl = rbln_kwargs.get("attn_impl", None)
        rbln_kvcache_partition_len = rbln_kwargs.get("kvcache_partition_len", None)
        rbln_kvcache_block_size = rbln_kwargs.get("kvcache_block_size", None)
        rbln_prefill_chunk_size = rbln_kwargs.get("prefill_chunk_size", None)

        if rbln_use_attention_mask is None:
            rbln_use_attention_mask = False
            rbln_npu = rbln_kwargs.get("npu", None) or rebel.get_npu_name()
            if rbln_npu == "RBLN-CA02":
                rbln_use_attention_mask = True

        if rbln_prefill_chunk_size is None:
            rbln_prefill_chunk_size = 128
        elif rbln_prefill_chunk_size % 64 != 0 or rbln_prefill_chunk_size == 0:
            raise ValueError(
                f"Invalid rbln_prefill_chunk_size: {rbln_prefill_chunk_size}. It must be a nonzero multiple of 64."
            )

        if rbln_max_seq_len is None:
            rbln_max_seq_len = getattr(model_config, "max_position_embeddings", None) or getattr(
                model_config, "n_positions", None
            )
        if rbln_max_seq_len is None:
            raise ValueError("`rbln_max_seq_len` should be specified.")

        rbln_batch_size = 1 if rbln_batch_size is None else rbln_batch_size
        rbln_use_inputs_embeds = False if rbln_use_inputs_embeds is None else rbln_use_inputs_embeds

        model_config = model_config.text_config

        rbln_attn_impl, rbln_kvcache_partition_len, rbln_kvcache_block_size = validate_attention_method(
            rbln_attn_impl=rbln_attn_impl,
            rbln_kvcache_partition_len=rbln_kvcache_partition_len,
            rbln_kvcache_block_size=rbln_kvcache_block_size,
            rbln_max_seq_len=rbln_max_seq_len,
        )

        if rbln_kvcache_block_size is None:
            if rbln_attn_impl == "eager":
                rbln_kvcache_block_size = rbln_max_seq_len
            else:
                rbln_kvcache_block_size = rbln_kvcache_partition_len

        rbln_kvcache_num_blocks = (rbln_max_seq_len // rbln_kvcache_block_size) * rbln_batch_size
        if rbln_attn_impl == "flash_attn":
            max_num_blocks, _ = cls.get_maximum_num_blocks(
                config=model_config,
                tensor_parallel_size=rbln_kwargs.get("tensor_parallel_size", 1),
                kvcache_block_size=rbln_kvcache_block_size,
                nbits_per_param=16,
                n_model_params=rbln_kwargs["n_model_params"],
            )
            rbln_kvcache_num_blocks = min(rbln_kvcache_num_blocks, max_num_blocks)

            required_blocks = rbln_max_seq_len // rbln_kvcache_block_size + 1
            if rbln_kvcache_num_blocks < required_blocks:
                rbln_kvcache_num_blocks = required_blocks

            logger.info(f"[KVCache] Compiling with num_blocks: {rbln_kvcache_num_blocks}")

            if rbln_kvcache_num_blocks < rbln_batch_size:
                raise RuntimeError(
                    f"Batch size ({rbln_batch_size}) exceeds available KV cache blocks ({rbln_kvcache_num_blocks}). "
                    "Ensure the number of blocks is at least equal to the batch size."
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
                (
                    "cache_position",
                    [batch_size, query_length],
                    "int32",
                ),
            ]

            if rbln_use_attention_mask:
                input_info.extend(
                    [
                        ("attention_mask", [batch_size, 1, query_length, rbln_max_seq_len], "float32"),
                    ]
                )

            if query_length > 1:
                input_info.extend(
                    [
                        ("query_position", [], "int16"),
                    ]
                )

            max_block_cnt = rbln_max_seq_len // rbln_kvcache_block_size

            if query_length > 1:
                input_info.extend([("block_tables", [max_block_cnt], "int16")])
            else:
                input_info.extend([("block_tables", [batch_size, max_block_cnt], "int16")])

            input_info.extend(
                [
                    (
                        f"past_key_values_{i}",
                        [
                            rbln_kvcache_num_blocks,
                            num_key_value_heads,
                            rbln_kvcache_block_size,
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
            query_length=rbln_prefill_chunk_size,
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
                "prefill_chunk_size": rbln_prefill_chunk_size,
                "use_attention_mask": rbln_use_attention_mask,
                "use_inputs_embeds": rbln_use_inputs_embeds,
                "kvcache_partition_len": rbln_kvcache_partition_len,
                "kvcache_block_size": rbln_kvcache_block_size,
                "attn_impl": rbln_attn_impl,
                "kvcache_num_blocks": rbln_kvcache_num_blocks,
            }
        )

        return rbln_config

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.BoolTensor] = None,
        image_hidden_states: Optional[torch.FloatTensor] = None,
        cache_position: torch.Tensor = None,
        generate_idx: Optional[torch.Tensor] = None,
        batch_idx: Optional[int] = None,
        **kwargs,
    ) -> Union[Tuple, Idefics3CausalLMOutputWithPast]:
        # retrieve input_ids and inputs_embeds
        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if cache_position is None:
            past_seen_tokens = 0
        else:
            past_seen_tokens = generate_idx - 1

        if inputs_embeds is not None and input_ids is None and past_seen_tokens == 0:
            raise ValueError("When first calling the model, if input_embeds are passed, input_ids should not be None.")

        if inputs_embeds is None:
            # inputs_embeds = self.text_model.get_input_embeddings()(input_ids).to(self.device)
            inputs_embeds = self.get_input_embeddings()(input_ids).to(self.device)

        # START VISUAL INPUTS INTEGRATION
        if pixel_values is not None and image_hidden_states is not None:
            raise ValueError("You cannot specify both pixel_values and image_hidden_states at the same time")
        elif pixel_values is not None:
            batch_size, num_images, num_channels, height, width = pixel_values.shape
            pixel_values = pixel_values.to(dtype=self.dtype)  # fp16 compatibility
            pixel_values = pixel_values.view(batch_size * num_images, *pixel_values.shape[2:])

            # Remove padding images - padding images are full 0.
            nb_values_per_image = pixel_values.shape[1:].numel()
            real_images_inds = (pixel_values == 0.0).sum(dim=(-1, -2, -3)) != nb_values_per_image
            pixel_values = pixel_values[real_images_inds].contiguous()

            # Handle the vision attention mask
            if pixel_attention_mask is None:
                pixel_attention_mask = torch.ones(
                    size=(pixel_values.size(0), pixel_values.size(2), pixel_values.size(3)),
                    dtype=torch.bool,
                    device=pixel_values.device,
                )
            else:
                # Remove padding images from the mask
                pixel_attention_mask = pixel_attention_mask.view(
                    batch_size * num_images, *pixel_attention_mask.shape[2:]
                )
                pixel_attention_mask = pixel_attention_mask[real_images_inds].contiguous()

            patch_size = self.config.vision_config.patch_size
            patches_subgrid = pixel_attention_mask.unfold(dimension=1, size=patch_size, step=patch_size)
            patches_subgrid = patches_subgrid.unfold(dimension=2, size=patch_size, step=patch_size)
            patch_attention_mask = (patches_subgrid.sum(dim=(-1, -2)) > 0).bool()

            # Get sequence from the vision encoder
            image_hidden_states = self.vision_model(
                pixel_values=pixel_values,
                patch_attention_mask=patch_attention_mask,
            ).last_hidden_state

            # Modality projection & resampling
            image_hidden_states = self.connector(image_hidden_states)

        elif image_hidden_states is not None:
            image_hidden_states = image_hidden_states.to(dtype=self.dtype, device=input_ids.device)

        if past_seen_tokens == 0 and inputs_embeds is not None and image_hidden_states is not None:
            # When we generate, we don't want to replace the potential image_token_id that we generated by images
            # that simply don't exist
            inputs_embeds = self.inputs_merger(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_hidden_states=image_hidden_states,
            )

        # Prefll
        if cache_position is None:
            logits = []
            inputs = inputs_embeds if inputs_embeds is not None else input_ids
            batch_size = inputs.shape[0]

            for b_idx in range(batch_size):
                cache_position = torch.arange(0, generate_idx[b_idx].item(), dtype=torch.int32).unsqueeze(0)
                logit = self.prefill_decoder(
                    input_ids=inputs[b_idx : b_idx + 1] if inputs_embeds is None else None,
                    inputs_embeds=inputs[b_idx : b_idx + 1] if inputs_embeds is not None else None,
                    attention_mask=attention_mask[b_idx] if attention_mask is not None else None,
                    cache_position=cache_position,
                    batch_idx=b_idx,
                )
                logits.append(logit)

            logits = torch.cat(logits, dim=0)
        # Decoder
        else:
            logits = self.decoder(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
            )

        return RBLNDecoderOnlyOutput(
            logits=logits,
            generate_idx=generate_idx,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        pixel_values=None,
        pixel_attention_mask=None,
        image_hidden_states=None,
        generate_idx=None,
        **kwargs,
    ):
        is_prefill_phase = generate_idx is None
        model_inputs = {}

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if not is_prefill_phase:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        if is_prefill_phase:
            generate_idx = attention_mask.sum(dim=-1, keepdim=True).int()
            cache_position = None
            pixel_values = pixel_values
            pixel_attention_mask = pixel_attention_mask
        else:
            if inputs_embeds is not None:
                raise NotImplementedError("Specifying inputs_embeds in decoder phase is not supported.")

            pixel_values = None
            pixel_attention_mask = None
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
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "pixel_attention_mask": pixel_attention_mask,
                "image_hidden_states": image_hidden_states,
                "cache_position": cache_position,
                "generate_idx": generate_idx,
            }
        )
        return model_inputs

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder, **kwargs):
        model_kwargs["generate_idx"] = outputs.generate_idx
        return model_kwargs

    def inputs_merger(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: Optional[torch.Tensor],
        image_hidden_states: Optional[torch.Tensor],
    ):
        """
        This method aims at merging the token embeddings with the image hidden states into one single sequence of vectors that are fed to the transformer LM.
        The merging happens as follows:
        - The text token sequence is: `tok_1 tok_2 tok_3 <fake_token_around_image> <image> <image> ... <image> <fake_token_around_image> tok_4`.
        - We get the image hidden states for the image through the vision encoder and that hidden state, after a pixel shuffle operation, is then projected into the text embedding space.
        We thus have a sequence of image hidden states of size (1, image_seq_len, hidden_dim), where 1 is for batch_size of 1 image and hidden_dim is the hidden_dim of the LM transformer.
        - The merging happens so that we obtain the following sequence: `vector_tok_1 vector_tok_2 vector_tok_3 vector_fake_tok_around_image {sequence of image_seq_len image hidden states} vector_fake_toke_around_image vector_tok_4`. That sequence is fed to the LM.
        - To fit the format of that sequence, `input_ids`, `input_embeds`, `attention_mask` are all 3 adapted to insert the image hidden states.
        """
        num_images, _, vision_hidden_size = image_hidden_states.shape
        special_image_token_mask = input_ids == self.config.image_token_id
        #  Fixes RuntimeError: a leaf Variable that requires grad is being used in an in-place operation.
        new_inputs_embeds = inputs_embeds.clone()
        reshaped_image_hidden_states = image_hidden_states.view(-1, vision_hidden_size)
        # cast to the dtype of the input_embeds to support quantized models
        reshaped_image_hidden_states = reshaped_image_hidden_states.to(inputs_embeds.device, inputs_embeds.dtype)
        new_inputs_embeds[special_image_token_mask] = reshaped_image_hidden_states
        return new_inputs_embeds
