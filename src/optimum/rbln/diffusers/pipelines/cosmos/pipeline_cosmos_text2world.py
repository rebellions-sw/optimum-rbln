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


from diffusers import CosmosTextToWorldPipeline
from diffusers.models import AutoencoderKLCosmos, CosmosTransformer3DModel
from diffusers.schedulers import EDMEulerScheduler
from transformers import T5EncoderModel, T5TokenizerFast

from ....utils.logging import get_logger
from ...modeling_diffusers import RBLNDiffusionMixin
from .guardrail.cosmos_guardrail import RBLNCosmosSafetyChecker


try:
    from cosmos_guardrail import CosmosSafetyChecker
except ImportError:
    from .guardrail.cosmos_guardrail import CosmosSafetyChecker

logger = get_logger(__name__)


class RBLNCosmosTextToWorldPipeline(RBLNDiffusionMixin, CosmosTextToWorldPipeline):
    """
    RBLN-accelerated implementation of Cosmos Text to World pipeline for text-to-video generation.

    This pipeline compiles Cosmos Text to World models to run efficiently on RBLN NPUs, enabling high-performance
    inference for generating images with distinctive artistic style and enhanced visual quality.
    """

    original_class = CosmosTextToWorldPipeline
    _submodules = ["text_encoder", "transformer", "vae"]
    _optional_components = ["safety_checker"]

    def __init__(
        self,
        text_encoder: T5EncoderModel,
        tokenizer: T5TokenizerFast,
        transformer: CosmosTransformer3DModel,
        vae: AutoencoderKLCosmos,
        scheduler: EDMEulerScheduler,
        safety_checker: CosmosSafetyChecker = None,
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
                f"The tranformer in this pipeline is compiled with 'num_frames={self.transformer.rbln_config.num_frames}'. 'num_frames' set by the user will be ignored"
            )
            kwargs.pop("num_frames")
        return kwargs
