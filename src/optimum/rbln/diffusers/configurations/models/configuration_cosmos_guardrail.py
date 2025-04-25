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

from typing import Optional

from ....configuration_utils import RBLNModelConfig
from ....utils.logging import get_logger


logger = get_logger(__name__)


class RBLNRetinaFaceConfig(RBLNModelConfig):
    def __init__(
        self,
        batch_size: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        **kwargs,
    ):
        """
        Args:
            height (Optional[int]): The width of input for inference. Defaults to 704.
            width (Optional[int]): The width of input for inference. Defaults to 1280.
            **kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If batch_size is not a positive integer.
        """
        super().__init__(**kwargs)
        self.batch_size = 1  # hard coded
        self.height = height or 704
        self.width = width or 1280


class RBLNVideoSafetyModelConfig(RBLNModelConfig):
    def __init__(
        self,
        batch_size: Optional[int] = None,
        input_size: Optional[int] = None,
        **kwargs,
    ):
        """
        Args:
            batch_size (Optional[int]): The batch size for inference. Defaults to 1.
            **kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If batch_size is not a positive integer.
        """
        super().__init__(**kwargs)
        self.batch_size = batch_size or 1
        self.input_size = 1152  # hard coded


class RBLNSiglipVisionModelConfig(RBLNModelConfig):
    def __init__(
        self,
        batch_size: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        **kwargs,
    ):
        """
        Args:
            batch_size (Optional[int]): The batch size for inference. Defaults to 1.
            **kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If batch_size is not a positive integer.
        """
        super().__init__(**kwargs)
        self.batch_size = batch_size or 1
        self.height = 384  # hard coded
        self.width = 384  # hard coded
