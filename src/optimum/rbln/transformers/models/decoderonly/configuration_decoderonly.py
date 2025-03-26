from typing import Optional

import rebel

from ....configuration_utils import RBLNModelConfig
from ....utils.logging import get_logger


logger = get_logger()


class RBLNDecoderOnlyModelForCausalLMConfig(RBLNModelConfig):
    def __init__(
        self,
        batch_size: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        use_inputs_embeds: Optional[bool] = None,
        use_attention_mask: Optional[bool] = None,
        attn_impl: Optional[str] = None,
        kvcache_partition_len: Optional[int] = None,
        kvcache_block_size: Optional[int] = None,
        quantization: Optional[str] = None,
        prefill_chunk_size: Optional[int] = None,
        kvcache_num_blocks: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.batch_size = batch_size or 1
        self.max_seq_len = max_seq_len
        self.use_inputs_embeds = use_inputs_embeds or False

        self.use_attention_mask = use_attention_mask or False
        npu = self.npu or rebel.get_npu_name()
        if npu == "RBLN-CA02":
            if self.use_attention_mask is False:
                logger.warning("Attention mask should be used with RBLN-CA02. Setting use_attention_mask to True.")
            self.use_attention_mask = True

        self.attn_impl = attn_impl
        self.kvcache_partition_len = kvcache_partition_len
        self.kvcache_block_size = kvcache_block_size
        self.quantization = quantization

        self.prefill_chunk_size = prefill_chunk_size or 128
        if self.prefill_chunk_size % 64 != 0 or self.prefill_chunk_size <= 0:
            raise ValueError("`prefill_chunk_size` must be a positive integer divisible by 64.")

        self.kvcache_num_blocks = kvcache_num_blocks
