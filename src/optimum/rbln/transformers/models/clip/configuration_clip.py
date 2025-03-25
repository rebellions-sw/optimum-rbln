from typing import Optional

from ....configuration_utils import RBLNModelConfig


class RBLNCLIPTextModelConfig(RBLNModelConfig):
    def __init__(self, batch_size: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size or 1


class RBLNCLIPTextModelWithProjectionConfig(RBLNCLIPTextModelConfig):
    pass


class RBLNCLIPVisionModelConfig(RBLNModelConfig):
    def __init__(self, batch_size: Optional[int] = None, image_size: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size or 1
        self.image_size = image_size


class RBLNCLIPVisionModelWithProjectionConfig(RBLNCLIPVisionModelConfig):
    pass
