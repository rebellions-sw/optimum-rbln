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

    @property
    def image_width(self):
        if isinstance(self.image_size, int):
            return self.image_size
        elif isinstance(self.image_size, (list, tuple)):
            return self.image_size[1]
        else:
            return self.image_size["width"]

    @property
    def image_height(self):
        if isinstance(self.image_size, int):
            return self.image_size
        elif isinstance(self.image_size, (list, tuple)):
            return self.image_size[0]
        else:
            return self.image_size["height"]


class RBLNCLIPVisionModelWithProjectionConfig(RBLNCLIPVisionModelConfig):
    pass
