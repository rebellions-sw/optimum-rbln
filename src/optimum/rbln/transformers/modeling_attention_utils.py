import math
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from optimum.rbln.transformers.models.decoderonly.configuration_decoderonly import (
    RBLNDecoderOnlyModelForCausalLMConfig,
)

from ..utils.logging import get_logger


logger = get_logger()

if TYPE_CHECKING:
    from rebel import RBLNCompiledModel
    from transformers import PretrainedConfig


DEFAULT_FLASH_ATTN_PARTITION_LENGTH = 16_384
DEFAULT_MAX_EAGER_ATTN_SEQUENCE_LENGTH = 32_768
MIN_FLASH_ATTN_MAX_SEQ_LEN = 8_192
MIN_FLASH_ATTN_PARTITION_LENGTH = 4_096
MAX_FLASH_ATTN_PARTITION_LENGTH = 32_768
MAX_SLIDING_WINDOW_SIZE = 32_768


def set_default_values(
    attn_impl: Optional[str] = None,
    kvcache_partition_len: Optional[int] = None,
    kvcache_block_size: Optional[int] = None,
    max_seq_len: Optional[int] = None,
) -> Tuple[str, int, int]:
    if attn_impl is None:
        attn_impl = "eager"

    if kvcache_partition_len is not None:
        if attn_impl == "eager":
            attn_impl = "flash_attn"
            logger.warning(
                "A non-null `kvcache_partition_len` was provided, but `attn_impl` was not explicitly set or "
                "set to 'eager'. Since KV cache partitioning is only supported with flash attention, "
                "`attn_impl` has been automatically switched to 'flash_attn'."
            )

    if kvcache_partition_len is None and attn_impl == "flash_attn":
        kvcache_partition_len = DEFAULT_FLASH_ATTN_PARTITION_LENGTH

    if kvcache_block_size is None:
        if attn_impl == "eager":
            kvcache_block_size = max_seq_len
        else:
            kvcache_block_size = kvcache_partition_len

    return attn_impl, kvcache_partition_len, kvcache_block_size


def validate_attention_method(attn_impl: str, kvcache_partition_len: int, kvcache_block_size: int, max_seq_len: int):
    if attn_impl not in ["eager", "flash_attn"]:
        raise ValueError(f"Unknown `attn_impl` : {attn_impl}. (Available : 'eager', 'flash_attn`)")

    ## Checking Constraints...
    # Constraint of eager attention:
    # - `max_seq_len` <= 32k

    # Constraints of flash attention:
    # 1. `max_seq_len` should be multiple of `partition_len`.
    # 2. 4k <= `partition_len` <= 32k.
    # 3. `max_seq_len` should be larger then 8k.
    if attn_impl == "eager" and max_seq_len > DEFAULT_MAX_EAGER_ATTN_SEQUENCE_LENGTH:
        raise ValueError(
            f"`max_seq_len` is set to {max_seq_len}, "
            f"which exceeds the limit of {DEFAULT_MAX_EAGER_ATTN_SEQUENCE_LENGTH} for 'eager' attention. "
            f"Please reduce the `max_seq_len` to {DEFAULT_MAX_EAGER_ATTN_SEQUENCE_LENGTH} or lower,"
            " or consider switching `attn_impl` to 'flash_attn' for larger sequence lengths."
        )

    if attn_impl == "flash_attn":
        if max_seq_len // kvcache_partition_len < 2 or max_seq_len % kvcache_partition_len != 0:
            raise ValueError(
                f"`max_seq_len` ({max_seq_len}) must be a multiple of `kvcache_partition_len` ({kvcache_partition_len}) "
                f"when using 'flash_attn'. Please adjust either value to meet this requirement."
            )
        elif not (MIN_FLASH_ATTN_PARTITION_LENGTH <= kvcache_partition_len <= MAX_FLASH_ATTN_PARTITION_LENGTH):
            raise ValueError(
                f"`kvcache_partition_len` ({kvcache_partition_len}) is out of the supported range for 'flash_attn' "
                f"({MIN_FLASH_ATTN_PARTITION_LENGTH} <= `kvcache_partition_len` <= {MAX_FLASH_ATTN_PARTITION_LENGTH}). "
                f"Please provide a valid value within this range."
            )
        elif max_seq_len < MIN_FLASH_ATTN_MAX_SEQ_LEN:
            raise ValueError(
                f"`max_seq_len` ({max_seq_len}) is too small for 'flash_attn'. The minimum "
                f"supported value is {MIN_FLASH_ATTN_MAX_SEQ_LEN}. Please increase `max_seq_len` to meet "
                "this requirement, or consider switching `attn_impl` to 'eager' for shorter lengths."
            )

    if kvcache_block_size is not None:
        if attn_impl == "flash_attn" and kvcache_partition_len != kvcache_block_size:
            raise ValueError(
                f" When using 'flash attention', the `kvcache_block_size` ({kvcache_block_size})  "
                f"must always be set equal to the `kvcache_partition_len` {kvcache_partition_len}."
            )
        elif attn_impl == "eager" and kvcache_block_size != max_seq_len:
            raise ValueError(
                f" When using 'eager attention', the `kvcache_block_size` ({kvcache_block_size})  "
                f"must always be set equal to the `max_seq_len` {max_seq_len}."
            )


def validate_sliding_window(rbln_config: RBLNDecoderOnlyModelForCausalLMConfig):
    if rbln_config.sliding_window > MAX_SLIDING_WINDOW_SIZE - rbln_config.prefill_chunk_size:
        raise ValueError(
            f"Sliding window size ({rbln_config.sliding_window}) must be less than 32768 - prefill_chunk_size ({32768 - rbln_config.prefill_chunk_size})"
        )

    if rbln_config.cache_impl == "sliding_window" and rbln_config.use_attention_mask:
        raise ValueError("`use_attention_mask` must be set to False when `cache_impl` is set to 'sliding_window'.")


class RBLNDecoderOnlyFlashAttentionMixin:
    @classmethod
    def get_maximum_num_blocks(
        cls,
        config: "PretrainedConfig",
        tensor_parallel_size: int,
        kvcache_block_size: int,
        nbits_per_param: Optional[int] = None,
        n_model_params: Optional[int] = None,
        kernel_size: Optional[int] = None,
        buffer: Optional[int] = None,
        num_runtimes: int = 2,
    ) -> int:
        # We are finding max_n_blocks(x) that satisfies the following equation:

        # available_dram - kernel_size - buffer
        #     - num_layers * 2 * tensor_parallel_size
        #     * align_2MB(
        #         x
        #         * block_size
        #         * align_64(head_dim)
        #         * math.ceil(num_key_value_heads / tensor_parallel_size)
        #         * 2
        #     ) > 0

        # This inequality can be rewritten as follows:

        # a - c * align_2MB(b * x) > 0
        # where
        #    a = available_dram - kernel_size - buffer
        #    b = block_size * align_64(head_dim) * math.ceil(num_key_value_heads / tensor_parallel_size) * 2
        #    c = num_layers * 2 * tensor_parallel_size

        # We can rewrite the inequality as follows:
        # k > align_2MB(b*x)
        # where
        #    k = a / c

        # After that, we can derive the following equation:
        # x = floor(2**21 / b * floor((k - 1) / 2**21))

        def align(x: int, nbytes: int) -> int:
            return int(math.ceil(x / nbytes) * nbytes)

        def align_2MB(x: int) -> int:
            return align(x, 2**21)

        num_attention_heads = getattr(config, "n_head", None) or getattr(config, "num_attention_heads")
        num_layers = getattr(config, "n_layer", None) or getattr(config, "num_hidden_layers")
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // num_attention_heads
        vocab_size = config.vocab_size
        hidden_size = getattr(config, "n_embd", None) or getattr(config, "hidden_size")
        num_key_value_heads = getattr(config, "num_key_value_heads", None) or num_attention_heads

        # TODO(jongho): Update if target npu is REBEL.
        ATOM_DRAM_NBYTES = 16 * 2**30
        ATOM_SYS_DRAM_NBYTES = 288 * 2**20
        available_dram = tensor_parallel_size * (ATOM_DRAM_NBYTES - ATOM_SYS_DRAM_NBYTES)

        if kernel_size is None:
            if n_model_params is None:
                raise ValueError("`n_model_params` should be specified to estimate the kernel memory.")
            # Get estimated kernel size (approximated)
            lm_heads_params = align(vocab_size, 64) * hidden_size
            lm_heads_nbytes = (
                align_2MB(lm_heads_params * nbits_per_param // 8 / tensor_parallel_size) * tensor_parallel_size
            )
            params = n_model_params - lm_heads_params
            layer_nbytes = (
                align_2MB(params * nbits_per_param // 8 / num_layers / tensor_parallel_size)
                * num_layers
                * tensor_parallel_size
            )
            kernel_size = layer_nbytes + lm_heads_nbytes
        elif n_model_params is not None:
            raise ValueError("Both `n_model_params` and `kernel_size` cannot be specified.")

        available_dram -= kernel_size

        if buffer is None:
            # TODO: Accurate buffer estimation
            buffer_per_runtime_per_core = 2**28  # 256MB per runtime
            buffer_per_core = buffer_per_runtime_per_core * num_runtimes  # 1 for prefill, 1 for decoder
            buffer = buffer_per_core * tensor_parallel_size
        available_dram -= buffer

        b = kvcache_block_size * align(head_dim, 64) * math.ceil(num_key_value_heads / tensor_parallel_size) * 2
        c = num_layers * 2 * tensor_parallel_size
        k = available_dram / c
        max_n_blocks = math.floor(2**21 / b * math.floor((k - 1) / 2**21))

        return max_n_blocks

    @classmethod
    def maybe_suggest_kvcache_num_blocks(
        cls,
        compiled_models: Dict[str, "RBLNCompiledModel"],
        model_config: "PretrainedConfig",
        rbln_config: RBLNDecoderOnlyModelForCausalLMConfig,
    ) -> None:
        # Get the actual memory allocation of each node by key
        alloc_memory_per_node_by_key: Dict[str, List[int]] = compiled_models["prefill"].get_alloc_per_node_by_key()
        alloc_memory_by_key: Dict[str, int] = {
            key: sum(memory_per_node) for key, memory_per_node in alloc_memory_per_node_by_key.items()
        }
        for batch_size in rbln_config.decoder_batch_sizes:
            for key, memory_per_node in (
                compiled_models[f"decoder_batch_{batch_size}"].get_alloc_per_node_by_key().items()
            ):
                alloc_memory_by_key[key] += sum(memory_per_node)
        alloc_memory_by_key.pop("PortRecur", None)  # Old compiler's kv-cache Key
        alloc_memory_by_key.pop("DramTensor", None)  # kv-cache
        kernel_size = alloc_memory_by_key.pop("Kernel")  # model weight

        # Get the maximum number of blocks that can be allocated
        buffer = sum(alloc_memory_by_key.values())
        max_num_blocks = cls.get_maximum_num_blocks(
            config=model_config,
            tensor_parallel_size=rbln_config.tensor_parallel_size,
            kvcache_block_size=rbln_config.kvcache_block_size,
            kernel_size=kernel_size,
            buffer=buffer,
        )

        # Since our estimation logic is not always accurate,
        # users can set `kvcache_num_blocks` to `max_num_blocks`.
        # If the memory is not enough, the model will fail to compile.
        if rbln_config.kvcache_num_blocks < max_num_blocks:
            logger.warning(
                f"Current `kvcache_num_blocks` setting is {rbln_config.kvcache_num_blocks}. "
                "Our analysis indicates that additional memory is available for more blocks. "
                f"Consider increasing `kvcache_num_blocks` to {max_num_blocks} for potentially improved performance. "
                "Please be advised that our memory estimation algorithm has limitations, "
                "and increasing this value may not guarantee successful model compilation."
            )
