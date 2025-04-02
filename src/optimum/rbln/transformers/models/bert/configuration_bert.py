from ...configuration_generic import (
    RBLNModelForMaskedLMConfig,
    RBLNModelForQuestionAnsweringConfig,
    RBLNTransformerEncoderForFeatureExtractionConfig,
)


class RBLNBertModelConfig(RBLNTransformerEncoderForFeatureExtractionConfig):
    pass


class RBLNBertForMaskedLMConfig(RBLNModelForMaskedLMConfig):
    pass


class RBLNBertForQuestionAnsweringConfig(RBLNModelForQuestionAnsweringConfig):
    pass
