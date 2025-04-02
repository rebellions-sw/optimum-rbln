from ...configuration_generic import RBLNTransformerEncoderForFeatureExtractionConfig
from ..seq2seq import RBLNSeq2SeqModelConfig


class RBLNBartModelConfig(RBLNTransformerEncoderForFeatureExtractionConfig):
    pass


class RBLNBartForConditionalGenerationConfig(RBLNSeq2SeqModelConfig):
    pass
