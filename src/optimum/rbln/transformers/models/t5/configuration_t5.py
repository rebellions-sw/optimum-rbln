from typing import Optional

from ...configuration_generic import RBLNTransformerEncoderForFeatureExtractionConfig
from ..seq2seq import RBLNModelForSeq2SeqLMConfig


class RBLNT5EncoderModelConfig(RBLNTransformerEncoderForFeatureExtractionConfig):
    def __init__(self, max_sequence_length: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)

        # FIXME: why need max_sequence_length ??
        self.max_seq_len = self.max_seq_len or max_sequence_length


class RBLNT5ForConditionalGenerationConfig(RBLNModelForSeq2SeqLMConfig):
    pass
