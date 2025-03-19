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

from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from diffusers.models.transformers.prior_transformer import PriorTransformer, PriorTransformerOutput
from transformers import PretrainedConfig, PreTrainedModel

from ....modeling import RBLNModel
from ....modeling_config import RBLNCompileConfig, RBLNConfig
from ....utils.logging import get_logger
from ....utils.runtime_utils import RBLNPytorchRuntime
from ...modeling_diffusers import RBLNDiffusionMixin


logger = get_logger(__name__)


class RBLNRuntimePriorTransformer(RBLNPytorchRuntime):
    def forward(
        self, hidden_states, timestep, proj_embedding, encoder_hidden_states, attention_mask, return_dict: bool = True
    ):
        predicted_image_embedding = super().forward(
            hidden_states,
            timestep,
            proj_embedding,
            encoder_hidden_states,
            attention_mask,
        )
        if return_dict:
            return PriorTransformerOutput(predicted_image_embedding=predicted_image_embedding)
        else:
            return (predicted_image_embedding,)


class _PriorTransformer(torch.nn.Module):
    def __init__(self, prior: PriorTransformer):
        super().__init__()
        self._prior = prior

    def forward(
        self,
        hidden_states,
        timestep,
        proj_embedding,
        encoder_hidden_states,
        attention_mask,
        return_dict=True,
    ):
        return self._prior.forward(
            hidden_states,
            timestep,
            proj_embedding,
            encoder_hidden_states,
            attention_mask,
            return_dict=False,
        )


class RBLNPriorTransformer(RBLNModel):
    hf_library_name = "diffusers"
    auto_model_class = PriorTransformer

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        self.runtime = RBLNRuntimePriorTransformer(runtime=self.model[0])
        artifacts = torch.load(self.model_save_dir / self.subfolder / "torch_artifacts.pth", weights_only=False)
        self.clip_mean = artifacts["clip_mean"]
        self.clip_std = artifacts["clip_std"]

    @classmethod
    def wrap_model_if_needed(cls, model: torch.nn.Module, rbln_config: RBLNConfig) -> torch.nn.Module:
        return _PriorTransformer(model).eval()

    @classmethod
    def update_rbln_config_using_pipe(cls, pipe: RBLNDiffusionMixin, rbln_config: Dict[str, Any]) -> Dict[str, Any]:
        batch_size = rbln_config.get("batch_size")
        if not batch_size:
            do_classifier_free_guidance = rbln_config.get("guidance_scale", 5.0) > 1.0
            batch_size = 2 if do_classifier_free_guidance else 1
        else:
            if rbln_config.get("guidance_scale"):
                logger.warning(
                    "guidance_scale is ignored because batch size is explicitly specified. "
                    "To ensure consistent behavior, consider removing the guidance scale or "
                    "adjusting the batch size configuration as needed."
                )
        embedding_dim = rbln_config.get("embedding_dim", pipe.prior.config.embedding_dim)
        num_embeddings = rbln_config.get("num_embeddings", pipe.prior.config.num_embeddings)

        rbln_config.update(
            {
                "batch_size": batch_size,
                "embedding_dim": embedding_dim,
                "num_embeddings": num_embeddings,
            }
        )

        return rbln_config

    @classmethod
    def save_torch_artifacts(
        cls,
        model: "PreTrainedModel",
        save_dir_path: Path,
        subfolder: str,
        rbln_config: RBLNConfig,
    ):
        save_dict = {}
        save_dict["clip_mean"] = model.clip_mean
        save_dict["clip_std"] = model.clip_std
        torch.save(save_dict, save_dir_path / subfolder / "torch_artifacts.pth")

    @classmethod
    def _get_rbln_config(
        cls,
        preprocessors,
        model_config: PretrainedConfig,
        rbln_kwargs,
    ) -> RBLNConfig:
        batch_size = rbln_kwargs.get("batch_size") or 1
        embedding_dim = rbln_kwargs.get("embedding_dim") or model_config.embedding_dim
        num_embeddings = rbln_kwargs.get("num_embeddings") or model_config.num_embeddings

        input_info = [
            ("hidden_states", [batch_size, embedding_dim], "float32"),
            ("timestep", [], "float32"),
            ("proj_embedding", [batch_size, embedding_dim], "float32"),
            ("encoder_hidden_states", [batch_size, num_embeddings, embedding_dim], "float32"),
            ("attention_mask", [batch_size, num_embeddings], "float32"),
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
        hidden_states,
        timestep: Union[torch.Tensor, float, int],
        proj_embedding: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        return_dict: bool = True,
    ):
        return self.runtime.forward(
            hidden_states.contiguous(),
            timestep.float(),
            proj_embedding,
            encoder_hidden_states,
            attention_mask.float(),
            return_dict,
        )

    def post_process_latents(self, prior_latents):
        prior_latents = (prior_latents * self.clip_std) + self.clip_mean
        return prior_latents
