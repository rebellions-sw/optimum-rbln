from ....configuration_utils import RBLNModelConfig


class RBLNWhisperForConditionalGenerationConfig(RBLNModelConfig):
    def __init__(
        self,
        batch_size: int = None,
        token_timestamps: bool = None,
        enc_max_seq_len: int = None,
        dec_max_seq_len: int = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.batch_size = batch_size or 1
        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")

        self.token_timestamps = token_timestamps or False
        self.enc_max_seq_len = enc_max_seq_len
        self.dec_max_seq_len = dec_max_seq_len
