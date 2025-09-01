# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, List, Optional, Tuple, Union

import torch

from ...configuration_generic import RBLNImageModelConfig, RBLNModelConfig


class RBLNGroundingDinoForObjectDetectionConfig(RBLNImageModelConfig):
    submodules = [
        "text_backbone",
        "backbone",
        "encoder",
        "decoder",
    ]

    def __init__(
        self,
        batch_size: Optional[int] = None,
        encoder: Optional["RBLNGroundingDinoEncoderConfig"] = None,
        decoder: Optional["RBLNGroundingDinoDecoderConfig"] = None,
        text_backbone: Optional["RBLNModelConfig"] = None,
        backbone: Optional["RBLNModelConfig"] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
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
        self.decoder = decoder
        self.text_backbone = text_backbone
        self.backbone = backbone
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states

        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")


class RBLNGroundingDinoComponentConfig(RBLNImageModelConfig):
    def __init__(
        self,
        image_size: Optional[Union[int, Tuple[int, int]]] = None,
        batch_size: Optional[int] = None,
        spatial_shapes_list: Optional[List[Tuple[int, int]]] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        **kwargs: Any,
    ):
        super().__init__(image_size=image_size, batch_size=batch_size, **kwargs)
        self.spatial_shapes_list = spatial_shapes_list
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states

    @property
    def spatial_shapes(self):
        if self.spatial_shapes_list is None:
            raise ValueError("Spatial shapes are not defined. Please set them before accessing.")
        return torch.tensor(self.spatial_shapes_list)


class RBLNGroundingDinoEncoderConfig(RBLNGroundingDinoComponentConfig):
    pass


class RBLNGroundingDinoDecoderConfig(RBLNGroundingDinoComponentConfig):
    pass
