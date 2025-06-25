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

from ....utils.logging import get_logger
from ...modeling_diffusers import RBLNDiffusionMixin


logger = get_logger(__name__)


class RBLNCosmosTextToWorldPipeline(RBLNDiffusionMixin, CosmosTextToWorldPipeline):
    """
    RBLN-accelerated implementation of Cosmos Text to World pipeline for text-to-video generation.

    This pipeline compiles Cosmos Text to World models to run efficiently on RBLN NPUs, enabling high-performance
    inference for generating images with distinctive artistic style and enhanced visual quality.
    """

    original_class = CosmosTextToWorldPipeline
    # _submodules = ["text_encoder", "transformer", "vae"]
    _optional_components = ["safety_checker"]

    def handle_additional_kwargs(self, **kwargs):
        if "num_frames" in kwargs and kwargs["num_frames"] != self.transformer.rbln_config.num_frames:
            logger.warning(
                f"The tranformer in this pipeline is compiled with 'num_frames={self.transformer.rbln_config.num_frames}'. 'num_frames' set by the user will be ignored"
            )
            kwargs.pop("num_frames")
        if "max_seq_len" in kwargs and kwargs["max_seq_len"] != self.transformer.rbln_config.max_seq_len:
            logger.warning(
                f"The tranformer in this pipeline is compiled with 'max_seq_len={self.transformer.rbln_config.max_seq_len}'. 'max_seq_len' set by the user will be ignored"
            )
            kwargs.pop("max_seq_len")
        return kwargs
