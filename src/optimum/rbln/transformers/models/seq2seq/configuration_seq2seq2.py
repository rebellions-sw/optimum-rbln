from typing import Optional

import rebel

from ....configuration_utils import RBLNModelConfig
from ....utils.logging import get_logger


logger = get_logger()


class RBLNSeq2SeqModelConfig(RBLNModelConfig):
    def __init__(
        self,
        batch_size: Optional[int] = None,
        enc_max_seq_len: Optional[int] = None,
        dec_max_seq_len: Optional[int] = None,
        use_attention_mask: Optional[bool] = None,
        pad_token_id: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.batch_size = batch_size or 1
        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")

        self.enc_max_seq_len = enc_max_seq_len
        self.dec_max_seq_len = dec_max_seq_len

        self.use_attention_mask = use_attention_mask
        npu = self.npu or rebel.get_npu_name()
        if npu == "RBLN-CA02":
            if self.use_attention_mask is False:
                logger.warning("Attention mask should be used with RBLN-CA02. Setting use_attention_mask to True.")
            self.use_attention_mask = True
        else:
            self.use_attention_mask = self.use_attention_mask or False

        self.pad_token_id = pad_token_id
