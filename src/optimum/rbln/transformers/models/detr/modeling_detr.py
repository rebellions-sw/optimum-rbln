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

from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from transformers import AutoModelForObjectDetection, PretrainedConfig, PreTrainedModel
from transformers.models.detr.modeling_detr import DetrObjectDetectionOutput

from ....modeling import RBLNModel
from ....modeling_config import RBLNCompileConfig, RBLNConfig
from ....utils.logging import get_logger
from .detr_architecture import DetrForObjectDetectionWrapper


if TYPE_CHECKING:
    from transformers import AutoImageProcessor, AutoProcessor

logger = get_logger(__name__)


class RBLNDetrForObjectDetection(RBLNModel):
    auto_model_class = AutoModelForObjectDetection
    _decoder_wrapper_cls = DetrForObjectDetectionWrapper

    @classmethod
    def wrap_model_if_needed(cls, model: "PreTrainedModel", rbln_config: "RBLNConfig"):
        print("wrapping detr model for optimize..")
        return cls._decoder_wrapper_cls(model).eval()

    @classmethod
    def _get_rbln_config(
        cls,
        preprocessors: Optional[Union["AutoImageProcessor", "AutoProcessor"]],
        model_config: Optional["PretrainedConfig"] = None,
        rbln_kwargs: Dict[str, Any] = {},
    ) -> RBLNConfig:
        rbln_image_size = rbln_kwargs.get("image_size", None)
        rbln_batch_size = rbln_kwargs.get("batch_size", None)

        if rbln_image_size is None:
            for processor in preprocessors:
                if hasattr(processor, "size"):
                    if all(required_key in processor.size.keys() for required_key in ["height", "width"]):
                        rbln_image_size = (processor.size["height"], processor.size["width"])
                    elif "shortest_edge" in processor.size.keys():
                        rbln_image_size = (processor.size["shortest_edge"], processor.size["shortest_edge"])
                    elif "longest_edge" in processor.size.keys():
                        rbln_image_size = (processor.size["longest_edge"], processor.size["longest_edge"])
                    break

            if rbln_image_size is None:
                rbln_image_size = model_config.image_size

            if rbln_image_size is None:
                raise ValueError("`rbln_image_size` should be specified!")

        if rbln_batch_size is None:
            rbln_batch_size = 1

        if isinstance(rbln_image_size, int):
            rbln_image_height, rbln_image_width = rbln_image_size, rbln_image_size
        elif isinstance(rbln_image_size, (list, tuple)):
            rbln_image_height, rbln_image_width = rbln_image_size[0], rbln_image_size[1]
        elif isinstance(rbln_image_size, dict):
            rbln_image_height, rbln_image_width = rbln_image_size["height"], rbln_image_size["width"]
        else:
            raise ValueError(
                "`rbln_image_size` should be `int` (ex. 800), `tuple` (ex. 800, 800), `dict` (ex. {'height': 800, 'width': 800}) format"
            )

        input_info = [
            (
                "pixel_values",
                [rbln_batch_size, 3, rbln_image_height, rbln_image_width],
                "float32",
            ),
            (
                "pixel_mask",
                [rbln_batch_size, rbln_image_height, rbln_image_width],
                "int64",
            ),
        ]

        rbln_compile_config = RBLNCompileConfig(input_info=input_info)
        return RBLNConfig(rbln_cls=cls.__name__, compile_cfgs=[rbln_compile_config], rbln_kwargs=rbln_kwargs)

    def forward(self, *args, **kwargs):
        outputs = super().forward(*args, **kwargs)

        logits = outputs[0]
        pred_boxes = outputs[1]
        last_hidden_state = outputs[2]
        encoder_last_hidden_state = outputs[3]

        return DetrObjectDetectionOutput(
            logits=logits,
            pred_boxes=pred_boxes,
            last_hidden_state=last_hidden_state,
            encoder_last_hidden_state=encoder_last_hidden_state,
        )
