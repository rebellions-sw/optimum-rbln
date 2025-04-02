from typing import List, Optional, Tuple, Union

from ..configuration_utils import RBLNModelConfig


class _RBLNTransformerEncoderConfig(RBLNModelConfig):
    rbln_model_input_names: Optional[List[str]] = None

    def __init__(
        self,
        max_seq_len: Optional[int] = None,
        batch_size: Optional[int] = None,
        model_input_names: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size or 1
        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")

        self.model_input_names = model_input_names or self.rbln_model_input_names


class _RBLNImageModelConfig(RBLNModelConfig):
    def __init__(
        self, image_size: Optional[Union[int, Tuple[int, int]]] = None, batch_size: Optional[int] = None, **kwargs
    ):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.batch_size = batch_size or 1
        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")

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


class RBLNModelForQuestionAnsweringConfig(_RBLNTransformerEncoderConfig):
    pass


class RBLNModelForSequenceClassificationConfig(_RBLNTransformerEncoderConfig):
    pass


class RBLNModelForMaskedLMConfig(_RBLNTransformerEncoderConfig):
    pass


class RBLNModelForTextEncodingConfig(_RBLNTransformerEncoderConfig):
    pass


class RBLNTransformerEncoderForFeatureExtractionConfig(_RBLNTransformerEncoderConfig):
    pass


class RBLNModelForImageClassificationConfig(_RBLNImageModelConfig):
    pass


class RBLNModelForDepthEstimationConfig(_RBLNImageModelConfig):
    pass


class RBLNModelForAudioClassificationConfig(RBLNModelConfig):
    def __init__(
        self,
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        num_mel_bins: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.batch_size = batch_size or 1
        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")

        self.max_length = max_length
        self.num_mel_bins = num_mel_bins
