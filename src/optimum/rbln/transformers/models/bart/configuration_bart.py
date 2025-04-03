from ...configuration_generic import RBLNTransformerEncoderForFeatureExtractionConfig
from ..seq2seq import RBLNModelForSeq2SeqLMConfig


class RBLNBartModelConfig(RBLNTransformerEncoderForFeatureExtractionConfig):
    pass


class RBLNBartForConditionalGenerationConfig(RBLNModelForSeq2SeqLMConfig):
    pass
