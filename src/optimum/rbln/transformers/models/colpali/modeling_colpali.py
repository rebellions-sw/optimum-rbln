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

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple, Union

import torch
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_utils import no_init_weights
from transformers.models.colpali.modeling_colpali import ColPaliForRetrieval, ColPaliForRetrievalOutput

from ....configuration_utils import RBLNModelConfig
from ....modeling import RBLNModel


if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer, PretrainedConfig


class RBLNColPaliForRetrieval(RBLNModel):
    """
    The ColPali Model transformer for document retrieval using vision-language models.
    This model inherits from [`RBLNModel`]. Check the superclass documentation for the generic methods the library implements for all its models.

    A class to convert and run pre-trained transformers based `ColPaliForRetrieval` model on RBLN devices.
    It implements the methods to convert a pre-trained transformers `ColPaliForRetrieval` model into a RBLN transformer model by:

    - transferring the checkpoint weights of the original into an optimized RBLN graph,
    - compiling the resulting graph using the RBLN compiler.

    **Configuration:**
    This model uses [`RBLNColPaliForRetrievalConfig`] for configuration. When calling methods like `from_pretrained` or `from_model`,
    the `rbln_config` parameter should be an instance of [`RBLNColPaliForRetrievalConfig`] or a dictionary conforming to its structure.

    See the [`RBLNColPaliForRetrievalConfig`] class for all available configuration options.

    Examples:
        ```python
        from optimum.rbln import RBLNColPaliForRetrieval

        # Simple usage using rbln_* arguments
        # `max_seq_lens` is automatically inferred from the model config
        model = RBLNColPaliForRetrieval.from_pretrained(
            "vidore/colpali-v1.3-hf",
            export=True,
            rbln_max_seq_lens=1152,
        )

        # Using a config dictionary
        rbln_config = {
            "max_seq_lens": 1152,
            "output_hidden_states": False,
        }
        model = RBLNColPaliForRetrieval.from_pretrained(
            "vidore/colpali-v1.3-hf",
            export=True,
            rbln_config=rbln_config
        )

        # Using a RBLNColPaliForRetrievalConfig instance (recommended for type checking)
        from optimum.rbln import RBLNColPaliForRetrievalConfig

        config = RBLNColPaliForRetrievalConfig(
            max_seq_lens=1152,
            output_hidden_states=False,
            tensor_parallel_size=4
        )
        model = RBLNColPaliForRetrieval.from_pretrained(
            "vidore/colpali-v1.3-hf",
            export=True,
            rbln_config=config
        )
        ```
    """

    auto_model_class = None
    _rbln_submodule_postfix = "model"
    _rbln_submodules = [
        {"name": "vlm"},
    ]

    def __post_init__(self, **kwargs):
        self.vlm_model = self.rbln_submodules[0]
        artifacts = torch.load(self.model_save_dir / self.subfolder / "torch_artifacts.pth", weights_only=False)
        self.embedding_proj_layer = self._create_embedding_proj_layer()
        self.embedding_proj_layer.load_state_dict(artifacts["embedding_proj_layer"])
        return super().__post_init__(**kwargs)

    def _create_embedding_proj_layer(self):
        with no_init_weights():
            embedding_proj_layer = torch.nn.Linear(
                self.config.vlm_config.text_config.hidden_size, self.config.embedding_dim
            )
        return embedding_proj_layer

    @classmethod
    def save_torch_artifacts(
        cls,
        model: "ColPaliForRetrieval",
        save_dir_path: Path,
        subfolder: str,
        rbln_config: RBLNModelConfig,
    ):
        save_dict = {}
        save_dict["embedding_proj_layer"] = model.embedding_proj_layer.state_dict()
        torch.save(save_dict, save_dir_path / subfolder / "torch_artifacts.pth")

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Optional[Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"]],
        model: Optional["PreTrainedModel"] = None,
        model_config: Optional["PretrainedConfig"] = None,
        rbln_config: Optional[RBLNModelConfig] = None,
    ) -> RBLNModelConfig:
        return rbln_config

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, ColPaliForRetrievalOutput]:
        if pixel_values is not None:
            pixel_values = pixel_values.to(dtype=self.dtype)

        if output_attentions:
            raise ValueError("output_attentions is not supported for RBLNColPaliForRetrieval")

        if output_hidden_states is not None and output_hidden_states != self.rbln_config.output_hidden_states:
            raise ValueError(
                f"Variable output_hidden_states {output_hidden_states} is not equal to rbln_config.output_hidden_states {self.rbln_config.output_hidden_states} "
                f"Please compile again with the correct argument."
            )

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vlm_output = self.vlm_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            return_dict=True,
            **kwargs,
        )
        vlm_hidden_states = vlm_output.hidden_states if output_hidden_states else None
        # vlm_image_hidden_states = vlm_output.image_hidden_states if pixel_values is not None else None

        last_hidden_states = vlm_output[0]
        proj_dtype = self.embedding_proj_layer.weight.dtype
        embeddings = self.embedding_proj_layer(last_hidden_states.to(proj_dtype))

        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        if attention_mask is not None:
            embeddings = embeddings * attention_mask.unsqueeze(-1)

        return ColPaliForRetrievalOutput(
            embeddings=embeddings,
            hidden_states=vlm_hidden_states,
            # image_hidden_states=vlm_image_hidden_states,
        )
