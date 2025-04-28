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


from diffusers import CosmosPipeline

from ....utils.logging import get_logger
from ...modeling_diffusers import RBLNDiffusionMixin


logger = get_logger(__name__)


class RBLNCosmosPipeline(RBLNDiffusionMixin, CosmosPipeline):
    original_class = CosmosPipeline
    _submodules = ["text_encoder", "transformer", "vae"]
    _optional_components = ["safety_checker"]

    def handle_additional_kwargs(self, **kwargs):
        if "fps" in kwargs and kwargs["fps"] != self.transformer.rbln_config.fps:
            logger.warning(
                f"The tranformer in this pipeline is compiled with 'fps={self.transformer.rbln_config.fps}'. 'fps' set by the user will be ignored"
            )
            kwargs.pop("fps")
        return kwargs
