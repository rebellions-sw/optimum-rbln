from .configuration_generic import (
    RBLNModelForAudioClassificationConfig,
    RBLNModelForImageClassificationConfig,
    RBLNModelForMaskedLMConfig,
    RBLNModelForQuestionAnsweringConfig,
    RBLNModelForSequenceClassificationConfig,
)


class RBLNASTForAudioClassificationConfig(RBLNModelForAudioClassificationConfig):
    pass


class RBLNDistilBertForQuestionAnsweringConfig(RBLNModelForQuestionAnsweringConfig):
    pass


class RBLNResNetForImageClassificationConfig(RBLNModelForImageClassificationConfig):
    pass


class RBLNXLMRobertaForSequenceClassificationConfig(RBLNModelForSequenceClassificationConfig):
    pass


class RBLNRobertaForSequenceClassificationConfig(RBLNModelForSequenceClassificationConfig):
    pass


class RBLNRobertaForMaskedLMConfig(RBLNModelForMaskedLMConfig):
    pass


class RBLNViTForImageClassificationConfig(RBLNModelForImageClassificationConfig):
    pass
