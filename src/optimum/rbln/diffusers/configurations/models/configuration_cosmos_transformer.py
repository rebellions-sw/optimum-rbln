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


class RBLNCosmosTransformer3DModelConfig(RBLNModelConfig):
    def __init__(
        self,
        batch_size: Optional[int] = None,
        num_frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        fps: Optional[int] = None,
        max_sequence_length: Optional[int] = None,
        num_channel_latents: Optional[int] = None,
        num_latent_frames: Optional[int] = None,
        latent_height: Optional[int] = None,
        latent_width: Optional[int] = None,
        hidden_size: Optional[int] = None,
        embedding_dim: Optional[int] = None,
        time_proj_num_channels: Optional[int] = None,
        **kwargs,
    ):
        """
        Args:
            batch_size (Optional[int]): The batch size for inference. Defaults to 1.
            num_frames (Optional[int]): The number of frames in the generated video.
            height (Optional[int]): The height in pixels of the generated image.
            width (Optional[int]): The width in pixels of the generated image.
            fps (Optional[int]): The frames per second of the generated video.
            **kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If batch_size is not a positive integer.
        """
        super().__init__(**kwargs)
        self.batch_size = batch_size or 1
        self.num_frames = num_frames or 121
        self.height = height or 704
        self.width = width or 1280
        self.fps = fps or 30
        self.max_seq_len = max_sequence_length or 512

        self.num_channel_latents = num_channel_latents
        self.num_latent_frames = num_latent_frames
        self.latent_height = latent_height
        self.latent_width = latent_width
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.time_proj_num_channels = time_proj_num_channels

        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")
