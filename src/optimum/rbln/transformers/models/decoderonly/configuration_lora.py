from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ....configuration_utils import RBLNSerializableConfigProtocol
from ....utils.logging import get_logger


logger = get_logger(__name__)


class RBLNLoRAAdapterConfig(RBLNSerializableConfigProtocol):
    """
    Configuration class for individual LoRA adapter settings.

    This class represents a single LoRA adapter that will be compiled into the RBLN model.
    Since RBLN NPU requires all adapters to be determined at compile time, each adapter
    must be fully specified including its weights.
    """

    def __init__(
        self,
        adapter_id: str,
        adapter_name: str,
        adapter_path: Union[str, Path],
        r: Optional[int] = None,
        lora_alpha: Optional[float] = None,
        target_modules: Optional[List[str]] = None,
        bias: Optional[str] = None,
        use_rslora: Optional[bool] = None,
        use_dora: Optional[bool] = None,
        scaling_factor: Optional[float] = None,
    ):
        """
        Args:
            adapter_id (str): Unique identifier for this LoRA adapter (e.g., "0", "1", "adapter_0").
                This ID will be used during runtime to select which adapter to use.
            adapter_name (str): Human-readable name for this adapter (e.g., "math_tuned", "code_tuned").
            adapter_path (Union[str, Path]): Path to the LoRA adapter weights directory or file.
                Must be accessible at compile time to load the weights.
            r (Optional[int]): The rank of the LoRA approximation for this adapter. Defaults to 8.
            lora_alpha (Optional[float]): The LoRA scaling parameter for this adapter. Defaults to 8.0.
            target_modules (Optional[List[str]]): List of module names to apply LoRA to.
                If None, inherits from the parent RBLNLoRAConfig. Defaults to None.
            bias (Optional[str]): Bias handling strategy. Options: "none", "all", "lora_only". Defaults to "none".
            use_rslora (Optional[bool]): Whether to use Rank-Stabilized LoRA. Defaults to False.
            use_dora (Optional[bool]): Whether to use DoRA (Weight-Decomposed Low-Rank Adaptation). Defaults to False.
            scaling_factor (Optional[float]): Additional scaling factor for this adapter. Defaults to 1.0.
            **kwargs: Additional adapter-specific arguments.

        Raises:
            ValueError: If adapter_id is empty or None.
            ValueError: If adapter_path doesn't exist.
            ValueError: If r is not a positive integer.
            ValueError: If lora_alpha is not positive.
        """
        if not adapter_id:
            raise ValueError("adapter_id cannot be empty or None")

        # Set default values
        r = r if r is not None else 8
        lora_alpha = lora_alpha if lora_alpha is not None else 8.0
        bias = bias if bias is not None else "none"
        use_rslora = use_rslora if use_rslora is not None else False
        use_dora = use_dora if use_dora is not None else False
        scaling_factor = scaling_factor if scaling_factor is not None else 1.0

        if not isinstance(r, int) or r <= 0:
            raise ValueError(f"r must be a positive integer, got {r}")

        if lora_alpha <= 0:
            raise ValueError(f"lora_alpha must be positive, got {lora_alpha}")

        if bias not in ["none", "all", "lora_only"]:
            raise ValueError(f"bias must be one of ['none', 'all', 'lora_only'], got {bias}")

        self.adapter_id = adapter_id
        self.adapter_name = adapter_name
        self.adapter_path = Path(adapter_path)

        # Validate that the adapter path exists
        if not self.adapter_path.exists():
            raise ValueError(f"LoRA adapter path does not exist: {self.adapter_path}")

        self.r = r
        self.lora_alpha = lora_alpha
        self.target_modules = target_modules
        self.bias = bias
        self.use_rslora = use_rslora
        self.use_dora = use_dora
        self.scaling_factor = scaling_factor

    def _prepare_for_serialization(self) -> Dict[str, Any]:
        config_dict = {
            "adapter_id": self.adapter_id,
            "adapter_name": self.adapter_name,
            "adapter_path": str(self.adapter_path),
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "target_modules": self.target_modules,
            "bias": self.bias,
            "use_rslora": self.use_rslora,
            "use_dora": self.use_dora,
            "scaling_factor": self.scaling_factor,
        }
        return config_dict

    def to_dict(self) -> Dict[str, Any]:
        return self._prepare_for_serialization()

    def __repr__(self) -> str:
        return f"RBLNLoRAAdapterConfig(adapter_id={self.adapter_id}, adapter_name={self.adapter_name}, r={self.r}, lora_alpha={self.lora_alpha})"


class RBLNLoRAConfig(RBLNSerializableConfigProtocol):
    """
    Configuration class for multi-LoRA support in RBLN decoder-only models.

    This class manages all LoRA adapters that will be compiled into the RBLN model.
    Since RBLN NPU requires all adapters to be determined at compile time, this
    configuration must specify all adapters upfront with their weights.

    Key constraints for RBLN multi-LoRA:
    1. All LoRA adapters must be specified at compile time
    2. Adapter weights must be available during compilation
    3. The number of adapters is fixed after compilation
    4. Runtime can only switch between pre-compiled adapters
    """

    def __init__(
        self,
        adapters: List[Union[Dict[str, Any], RBLNLoRAAdapterConfig]],
        global_lora_dtype: Optional[str] = None,
        global_target_modules: Optional[List[str]] = None,
        max_lora_rank: Optional[int] = None,
        lora_extra_vocab_size: Optional[int] = None,
        fully_sharded_loras: Optional[bool] = None,
        enable_lora_bias: Optional[bool] = None,
        long_lora_scaling_factor: Optional[float] = None,
        lora_a_stack_dim: Optional[int] = None,
        lora_b_stack_dim: Optional[int] = None,
    ):
        """
        Args:
            adapters (List[Union[Dict[str, Any], RBLNLoRAAdapterConfig]]): List of LoRA adapters
                to be compiled into the model. Each adapter must be fully specified with weights
                accessible at compile time.
            global_lora_dtype (Optional[str]): Global data type for LoRA weights. Individual adapters can
                override this. Supported: "float16", "bfloat16", "float32". Defaults to "float16".
            global_target_modules (Optional[List[str]]): Global list of target modules for LoRA.
                Individual adapters can override this. Common values include
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"].
                Defaults to ["q_proj", "k_proj", "v_proj", "o_proj"].
            max_lora_rank (Optional[int]): Maximum rank across all adapters. If None, automatically
                determined from the provided adapters. Used for memory allocation optimization.
            lora_extra_vocab_size (Optional[int]): Additional vocabulary size for LoRA embeddings.
                Defaults to 256.
            fully_sharded_loras (Optional[bool]): Whether to use fully sharded LoRA weights across devices.
                Useful for very large LoRA adapters. Defaults to False.
            enable_lora_bias (Optional[bool]): Whether to enable bias terms for LoRA layers. Defaults to False.
            long_lora_scaling_factor (Optional[float]): Scaling factor for long sequence LoRA.
                Defaults to None.
            lora_a_stack_dim (Optional[int]): Dimension for stacking LoRA A matrices. Defaults to 2.
            lora_b_stack_dim (Optional[int]): Dimension for stacking LoRA B matrices. Defaults to 1.
            **kwargs: Additional arguments for future extensions.

        Raises:
            ValueError: If adapters list is empty.
            ValueError: If adapter IDs are not unique.
            ValueError: If global_lora_dtype is not supported.
            ValueError: If any adapter path doesn't exist.
        """
        if not adapters:
            raise ValueError("adapters list cannot be empty")

        # Set default values
        global_lora_dtype = global_lora_dtype if global_lora_dtype is not None else "float16"
        global_target_modules = (
            global_target_modules if global_target_modules is not None else ["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        lora_extra_vocab_size = lora_extra_vocab_size if lora_extra_vocab_size is not None else 256
        fully_sharded_loras = fully_sharded_loras if fully_sharded_loras is not None else False
        enable_lora_bias = enable_lora_bias if enable_lora_bias is not None else False
        lora_a_stack_dim = lora_a_stack_dim if lora_a_stack_dim is not None else 2
        lora_b_stack_dim = lora_b_stack_dim if lora_b_stack_dim is not None else 1

        if global_lora_dtype not in ["float16", "bfloat16", "float32"]:
            raise ValueError(
                f"global_lora_dtype must be one of ['float16', 'bfloat16', 'float32'], got {global_lora_dtype}"
            )

        # Convert dict adapters to RBLNLoRAAdapterConfig objects
        self.adapters: List[RBLNLoRAAdapterConfig] = []
        for adapter in adapters:
            if isinstance(adapter, dict):
                # Apply global settings as defaults
                if global_target_modules is not None and adapter.get("target_modules") is None:
                    adapter["target_modules"] = global_target_modules
                self.adapters.append(RBLNLoRAAdapterConfig(**adapter))
            elif isinstance(adapter, RBLNLoRAAdapterConfig):
                # Apply global target modules if adapter doesn't have them
                if global_target_modules is not None and adapter.target_modules is None:
                    adapter.target_modules = global_target_modules
                self.adapters.append(adapter)
            else:
                raise ValueError(f"Invalid adapter type: {type(adapter)}")

        # Validate unique adapter IDs
        adapter_ids = [adapter.adapter_id for adapter in self.adapters]
        if len(adapter_ids) != len(set(adapter_ids)):
            raise ValueError("All adapter IDs must be unique")

        self.global_lora_dtype = global_lora_dtype
        self.global_target_modules = global_target_modules
        self.lora_extra_vocab_size = lora_extra_vocab_size
        self.fully_sharded_loras = fully_sharded_loras
        self.enable_lora_bias = enable_lora_bias
        self.long_lora_scaling_factor = long_lora_scaling_factor
        self.lora_a_stack_dim = lora_a_stack_dim
        self.lora_b_stack_dim = lora_b_stack_dim

        # Calculate max_lora_rank if not provided
        if max_lora_rank is None:
            self.max_lora_rank = max(adapter.r for adapter in self.adapters)
        else:
            self.max_lora_rank = max_lora_rank
            # Validate that max_lora_rank is sufficient
            actual_max_rank = max(adapter.r for adapter in self.adapters)
            if self.max_lora_rank < actual_max_rank:
                raise ValueError(
                    f"max_lora_rank ({self.max_lora_rank}) must be >= actual max rank ({actual_max_rank})"
                )

    @property
    def num_adapters(self) -> int:
        """Get the number of LoRA adapters."""
        return len(self.adapters)

    @property
    def adapter_ids(self) -> List[str]:
        """Get list of all adapter IDs."""
        return [adapter.adapter_id for adapter in self.adapters]

    @property
    def adapter_names(self) -> List[str]:
        """Get list of all adapter names."""
        return [adapter.adapter_name for adapter in self.adapters]

    def get_adapter_by_id(self, adapter_id: str) -> Optional[RBLNLoRAAdapterConfig]:
        """Get an adapter configuration by its ID."""
        for adapter in self.adapters:
            if adapter.adapter_id == adapter_id:
                return adapter
        return None

    def get_adapter_by_name(self, adapter_name: str) -> Optional[RBLNLoRAAdapterConfig]:
        """Get an adapter configuration by its name."""
        for adapter in self.adapters:
            if adapter.adapter_name == adapter_name:
                return adapter
        return None

    def validate_adapter_weights(self) -> Dict[str, bool]:
        """Validate that all adapter weights are accessible at compile time."""
        validation_results = {}
        for adapter in self.adapters:
            try:
                # Check if adapter path exists and contains expected files
                adapter_path = adapter.adapter_path
                if adapter_path.is_file():
                    # Single file adapter (e.g., safetensors)
                    validation_results[adapter.adapter_id] = adapter_path.exists()
                else:
                    # Directory adapter - check for common LoRA files
                    expected_files = ["adapter_model.safetensors", "adapter_config.json"]
                    alternative_files = ["pytorch_model.bin", "adapter_model.bin"]

                    has_weights = any((adapter_path / f).exists() for f in expected_files + alternative_files)
                    has_config = (adapter_path / "adapter_config.json").exists()

                    validation_results[adapter.adapter_id] = has_weights and has_config
            except Exception as e:
                logger.warning(f"Failed to validate adapter {adapter.adapter_id}: {e}")
                validation_results[adapter.adapter_id] = False

        return validation_results

    def _prepare_for_serialization(self) -> Dict[str, Any]:
        """Prepare the LoRA configuration for serialization by converting nested objects to dictionaries."""
        serializable_map = {
            "adapters": [adapter._prepare_for_serialization() for adapter in self.adapters],
            "global_lora_dtype": self.global_lora_dtype,
            "global_target_modules": self.global_target_modules,
            "max_lora_rank": self.max_lora_rank,
            "lora_extra_vocab_size": self.lora_extra_vocab_size,
            "fully_sharded_loras": self.fully_sharded_loras,
            "enable_lora_bias": self.enable_lora_bias,
            "long_lora_scaling_factor": self.long_lora_scaling_factor,
            "lora_a_stack_dim": self.lora_a_stack_dim,
            "lora_b_stack_dim": self.lora_b_stack_dim,
        }
        return serializable_map

    def to_dict(self) -> Dict[str, Any]:
        """Convert the LoRA configuration to a dictionary."""
        return self._prepare_for_serialization()

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "RBLNLoRAConfig":
        """Create a LoRA configuration from a dictionary."""
        return cls(**config_dict)

    def __repr__(self) -> str:
        return f"RBLNLoRAConfig(num_adapters={self.num_adapters}, adapter_ids={self.adapter_ids}, max_lora_rank={self.max_lora_rank})"
