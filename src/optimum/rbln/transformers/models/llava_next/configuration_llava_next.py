from typing import Optional

from ....configuration_utils import RBLNModelConfig


class RBLNLlavaNextForConditionalGenerationConfig(RBLNModelConfig):
    submodules = ["vision_tower", "language_model"]

    def __init__(
        self,
        batch_size: Optional[int] = None,
        vision_tower: Optional[RBLNModelConfig] = None,
        language_model: Optional[RBLNModelConfig] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.batch_size = batch_size or 1

        self.vision_tower = vision_tower
        self.language_model = language_model
