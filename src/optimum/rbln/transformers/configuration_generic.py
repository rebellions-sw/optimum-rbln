from typing import List, Optional, Tuple, Union

from ..configuration_utils import RBLNModelConfig


class RBLNModelForQuestionAnsweringConfig(RBLNModelConfig):
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
        self.model_input_names = model_input_names


class RBLNModelForImageClassificationConfig(RBLNModelConfig):
    def __init__(
        self, image_size: Optional[Union[int, Tuple[int, int]]] = None, batch_size: Optional[int] = None, **kwargs
    ):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.batch_size = batch_size or 1

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
        self.max_length = max_length
        self.num_mel_bins = num_mel_bins


class RBLNModelForSequenceClassificationConfig(RBLNModelConfig):
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
        self.model_input_names = model_input_names


class RBLNModelForMaskedLMConfig(RBLNModelConfig):
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
        self.model_input_names = model_input_names
