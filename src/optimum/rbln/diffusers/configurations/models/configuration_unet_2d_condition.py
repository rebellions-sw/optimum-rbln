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


class RBLNUNet2DConditionModelConfig(RBLNModelConfig):
    subclass_non_save_attributes = ["_batch_size_is_specified"]

    def __init__(
        self,
        batch_size: Optional[int] = None,
        sample_size: Optional[Tuple[int, int]] = None,
        in_channels: Optional[int] = None,
        cross_attention_dim: Optional[int] = None,
        use_additional_residuals: Optional[bool] = None,
        max_seq_len: Optional[int] = None,
        in_features: Optional[int] = None,
        text_model_hidden_size: Optional[int] = None,
        image_model_hidden_size: Optional[int] = None,
        **kwargs,
    ):
        """
        Args:
            batch_size (Optional[int]): The batch size for inference. Defaults to 1.
            sample_size (Optional[Tuple[int, int]]): The spatial dimensions (height, width) of the generated samples.
                If an integer is provided, it's used for both height and width.
            in_channels (Optional[int]): Number of input channels for the UNet.
            cross_attention_dim (Optional[int]): Dimension of the cross-attention features.
            use_additional_residuals (Optional[bool]): Whether to use additional residual connections in the model.
            max_seq_len (Optional[int]): Maximum sequence length for text inputs when used with cross-attention.
            in_features (Optional[int]): Number of input features for the model.
            text_model_hidden_size (Optional[int]): Hidden size of the text encoder model.
            image_model_hidden_size (Optional[int]): Hidden size of the image encoder model.
            **kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If batch_size is not a positive integer.
        """
        super().__init__(**kwargs)
        self._batch_size_is_specified = batch_size is not None

        self.batch_size = batch_size or 1
        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")

        self.in_channels = in_channels
        self.cross_attention_dim = cross_attention_dim
        self.use_additional_residuals = use_additional_residuals
        self.max_seq_len = max_seq_len
        self.in_features = in_features
        self.text_model_hidden_size = text_model_hidden_size
        self.image_model_hidden_size = image_model_hidden_size

        self.sample_size = sample_size
        if isinstance(sample_size, int):
            self.sample_size = (sample_size, sample_size)

    @property
    def batch_size_is_specified(self):
        return self._batch_size_is_specified
