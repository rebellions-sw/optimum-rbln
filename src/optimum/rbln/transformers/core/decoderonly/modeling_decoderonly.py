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
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import rebel
import torch
from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_utils import no_init_weights

from ....modeling import RBLNModel
from ....utils.logging import get_logger
from .compile_utils import DecoderOnlyCompileUtils
from .generation_utils import DecoderOnlyGenerationUtils
from .runtime_utils import RBLNRuntimeModel


logger = get_logger()

if TYPE_CHECKING:
    pass


class RBLNDecoderOnlyModelForCausalLM(DecoderOnlyCompileUtils, DecoderOnlyGenerationUtils, RBLNModel):
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

    def __post_init__(self, **kwargs):
        self.batch_size = self.rbln_config.model_cfg["batch_size"]
        self.max_seq_len = self.rbln_config.model_cfg["max_seq_len"]
        self.prefill_chunk_size = self.rbln_config.model_cfg["prefill_chunk_size"]
        self.kvcache_block_size = self.rbln_config.model_cfg["kvcache_block_size"]
        self.kvcache_num_blocks = self.rbln_config.model_cfg["kvcache_num_blocks"]
        self.use_attention_mask = self.rbln_config.model_cfg["use_attention_mask"]
        attn_impl = self.rbln_config.model_cfg["attn_impl"]
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
            vocab_size=self.config.vocab_size,
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

        val = getattr(self.get_hf_class(), __name, None) or getattr(PreTrainedModel, __name)
        if isinstance(val, Callable) and "self" in set(inspect.signature(val).parameters):
            return redirect(val)
        return val

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
