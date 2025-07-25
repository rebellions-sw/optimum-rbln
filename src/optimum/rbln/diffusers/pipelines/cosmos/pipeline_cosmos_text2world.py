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


from typing import Any, Dict, Optional

from diffusers import CosmosTextToWorldPipeline
from diffusers.schedulers import EDMEulerScheduler
from transformers import T5TokenizerFast

from ....transformers.models.t5.modeling_t5 import RBLNT5EncoderModel
from ....utils.logging import get_logger
from ...modeling_diffusers import RBLNDiffusionMixin
from ...models.autoencoders.autoencoder_kl_cosmos import RBLNAutoencoderKLCosmos
from ...models.transformers.transformer_cosmos import RBLNCosmosTransformer3DModel
from .cosmos_guardrail import RBLNCosmosSafetyChecker


logger = get_logger(__name__)


class RBLNCosmosTextToWorldPipeline(RBLNDiffusionMixin, CosmosTextToWorldPipeline):
    """
    RBLN-accelerated implementation of Cosmos Text to World pipeline for text-to-video generation.

    This pipeline compiles Cosmos Text to World models to run efficiently on RBLN NPUs, enabling high-performance
    inference for generating videos with distinctive artistic style and enhanced visual quality.
    """

    original_class = CosmosTextToWorldPipeline
    _submodules = ["text_encoder", "transformer", "vae"]
    _optional_submodules = ["safety_checker"]

    def __init__(
        self,
        text_encoder: RBLNT5EncoderModel,
        tokenizer: T5TokenizerFast,
        transformer: RBLNCosmosTransformer3DModel,
        vae: RBLNAutoencoderKLCosmos,
        scheduler: EDMEulerScheduler,
        safety_checker: RBLNCosmosSafetyChecker = None,
    ):
        if safety_checker is None:
            safety_checker = RBLNCosmosSafetyChecker()

        super().__init__(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
            safety_checker=safety_checker,
        )

    def handle_additional_kwargs(self, **kwargs):
        if "num_frames" in kwargs and kwargs["num_frames"] != self.transformer.rbln_config.num_frames:
            logger.warning(
                f"The transformer in this pipeline is compiled with 'num_frames={self.transformer.rbln_config.num_frames}'. 'num_frames' set by the user will be ignored"
            )
            kwargs.pop("num_frames")
        if (
            "max_sequence_length" in kwargs
            and kwargs["max_sequence_length"] != self.transformer.rbln_config.max_seq_len
        ):
            logger.warning(
                f"The transformer in this pipeline is compiled with 'max_seq_len={self.transformer.rbln_config.max_seq_len}'. 'max_sequence_length' set by the user will be ignored"
            )
            kwargs.pop("max_sequence_length")
        return kwargs

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        export: bool = False,
        safety_checker: Optional[RBLNCosmosSafetyChecker] = None,
        rbln_config: Dict[str, Any] = {},
        **kwargs: Any,
    ):
        rbln_config, kwargs = cls.get_rbln_config_class().initialize_from_kwargs(rbln_config, **kwargs)
        if safety_checker is None and export:
            safety_checker = RBLNCosmosSafetyChecker(rbln_config=rbln_config.safety_checker)

        return super().from_pretrained(
            model_id, export=export, safety_checker=safety_checker, rbln_config=rbln_config, **kwargs
        )
