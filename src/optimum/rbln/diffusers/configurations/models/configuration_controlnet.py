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

from typing import Optional, Tuple

from ....configuration_utils import RBLNModelConfig


class RBLNControlNetModelConfig(RBLNModelConfig):
    def __init__(
        self,
        batch_size: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        unet_sample_size: Optional[Tuple[int, int]] = None,
        vae_sample_size: Optional[Tuple[int, int]] = None,
        text_model_hidden_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.batch_size = batch_size or 1
        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")

        self.max_seq_len = max_seq_len
        self.unet_sample_size = unet_sample_size
        self.vae_sample_size = vae_sample_size
        self.text_model_hidden_size = text_model_hidden_size
