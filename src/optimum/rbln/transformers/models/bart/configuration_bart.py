from ...configuration_generic import RBLNModelForTextEncodingConfig
from ..seq2seq import RBLNSeq2SeqModelConfig


class RBLNBartModelConfig(RBLNModelForTextEncodingConfig):
    pass


class RBLNBartForConditionalGenerationConfig(RBLNSeq2SeqModelConfig):
    pass
