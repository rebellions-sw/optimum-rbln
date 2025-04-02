from typing import Optional, Tuple

from ...configuration_utils import RBLNModelConfig


class RBLNUNet2DConditionModelConfig(RBLNModelConfig):
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
        image_size: Optional[Tuple[int, int]] = None,
        *,
        img_height: Optional[int] = None,
        img_width: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
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
        if isinstance(self.sample_size, int):
            self.sample_size = (self.sample_size, self.sample_size)

        if image_size is not None and (img_height is not None or img_width is not None):
            raise ValueError("image_size and img_height/img_width cannot both be provided")

        if img_height is not None and img_width is not None:
            self.image_size = (img_height, img_width)
        else:
            self.image_size = image_size
