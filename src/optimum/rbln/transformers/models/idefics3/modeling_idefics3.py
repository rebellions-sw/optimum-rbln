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
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import rebel
import torch
from transformers import (
    AutoModelForVision2Seq,
    Idefics3ForConditionalGeneration,
    Idefics3VisionTransformer,
    Idefics3VisionConfig,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import BaseModelOutputWithPooling, BaseModelOutput
from transformers.modeling_utils import no_init_weights
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_attention_mask,
)

from ....modeling import RBLNModel
from ....modeling_config import RBLNCompileConfig, RBLNConfig
from ....utils.runtime_utils import RBLNPytorchRuntime
from ....utils.logging import get_logger
from ..decoderonly.modeling_decoderonly import RBLNDecoderOnlyOutput


logger = get_logger(__name__)

if TYPE_CHECKING:
    from transformers import (
        AutoFeatureExtractor,
        AutoProcessor,
        AutoTokenizer,
        PretrainedConfig,
    )
    

class RBLNRuntimeModel(RBLNPytorchRuntime):
    mandatory_members = ["main_input_name"]
    
    def __init__(
        self,
        runtime: rebel.Runtime,
        model: Idefics3VisionTransformer,
        **kwargs: Any,
    ) -> None:
        super().__init__(runtime, **kwargs)
        self.base_model = model
        self.patch_size = model.config.patch_size
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

        hidden_states = self.base_model.embeddings(pixel_values=pixel_values, patch_attention_mask=patch_attention_mask)
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
        self.model = RBLNRuntimeModel(self.model[0], main_input_name="pixel_values", model=self.base_model)
        
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
        
        # pixel_values = batch_size, num_images, num_channels, height, width
        input_info = [
            (
                "hidden_states",
                [
                    rbln_batch_size * 30, # batch_size * num_images
                    # model_config.num_channels,
                    (model_config.image_size // model_config.patch_size) ** 2,
                    model_config.hidden_size
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

        last_hidden_state = self.model(pixel_values, patch_attention_mask, output_attentions, output_hidden_states, return_dict=False)

        return BaseModelOutput(last_hidden_state=last_hidden_state)