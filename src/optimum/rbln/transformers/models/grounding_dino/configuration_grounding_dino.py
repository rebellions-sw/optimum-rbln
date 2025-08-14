# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from json import encoder
from typing import Any, Optional

from ....configuration_utils import RBLNModelConfig
from ...configuration_generic import RBLNImageModelConfig


class RBLNGroundingDinoForObjectDetectionConfig(RBLNImageModelConfig):
    submodules = ["encoder"]

    def __init__(
        self,
        batch_size: Optional[int] = None,
        encoder: Optional["RBLNGroundingDinoEncoderConfig"] = None,
        **kwargs: Any,
    ):
        """
        Args:
            batch_size (Optional[int]): The batch size for text processing. Defaults to 1.
            **kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If batch_size is not a positive integer.
        """
        super().__init__(**kwargs)
        self.encoder = encoder
        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")


class RBLNGroundingDinoEncoderConfig(RBLNImageModelConfig):
    def __init__(self, batch_size: Optional[int] = None, **kwargs: Any):
        super().__init__(**kwargs)
        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")
