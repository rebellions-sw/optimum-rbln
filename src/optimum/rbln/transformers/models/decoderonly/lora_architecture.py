import math
from pathlib import Path
from typing import Optional

import safetensors.torch
import torch
from torch import nn

from .configuration_lora import RBLNLoRAConfig


class LoRALinear(nn.Module):
    """
    A linear layer that supports multiple LoRA adapters compiled at static time.

    This class replaces the original linear layer and handles both base weights
    and multiple LoRA adapters in a single forward pass using custom ops.
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        lora_config: RBLNLoRAConfig,
        projection_name: str = "",
        layer_idx: int = 0,
    ):
        """
        Args:
            original_linear: The original linear layer to be replaced
            lora_config: LoRA configuration containing all adapters
            projection_name: Name of the projection (e.g., "q_proj", "k_proj")
            layer_idx: Layer index for loading the correct LoRA weights
        """
        super().__init__()

        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.projection_name = projection_name
        self.layer_idx = layer_idx
        self.lora_config = lora_config

        # Store original linear weights and bias directly without cloning
        self.register_buffer("weight", original_linear.weight.data)
        if original_linear.bias is not None:
            self.register_buffer("bias", original_linear.bias.data)
        else:
            self.register_buffer("bias", torch.zeros(self.out_features))

        # Initialize LoRA weights
        self._init_lora_weights()

    def _should_apply_lora(self) -> bool:
        """Check if this projection should have LoRA applied."""
        # Check if any adapter targets this projection
        return any(self.projection_name in adapter.target_modules for adapter in self.lora_config.adapters)

    def _load_adapter_weights(self, adapter_path: Path):
        """
        Load adapter weights from local directory.

        Args:
            adapter_path: Path to local directory containing adapter weights

        Returns:
            Dictionary containing adapter weights

        Raises:
            FileNotFoundError: If no adapter weights are found in the directory
        """
        if not adapter_path.is_dir():
            raise ValueError(f"Adapter path must be a directory, got: {adapter_path}")

        # Try to load weights in order of preference
        weight_files = [
            ("adapter_model.safetensors", lambda p: safetensors.torch.load_file(p)),
            ("adapter_model.bin", lambda p: torch.load(p, map_location="cpu")),
            ("pytorch_model.bin", lambda p: torch.load(p, map_location="cpu")),
        ]

        for filename, load_fn in weight_files:
            weight_path = adapter_path / filename
            if weight_path.exists():
                return load_fn(weight_path)

        raise FileNotFoundError(
            f"No adapter weights found in {adapter_path}. "
            f"Expected one of: {', '.join(filename for filename, _ in weight_files)}"
        )

    def _init_lora_weights(self):
        """Initialize LoRA adapter weights by loading and stacking them."""

        lora_a_weights = []
        lora_b_weights = []
        scaling_factors = []

        for adapter in self.lora_config.adapters:
            if self.projection_name not in adapter.target_modules:
                # Create zero weights for adapters that don't target this projection
                lora_a_weights.append(torch.zeros(adapter.r, self.in_features))
                lora_b_weights.append(torch.zeros(self.out_features, adapter.r))
                scaling_factors.append(0.0)
                continue

            adapter_weights = self._load_adapter_weights(adapter.local_adapter_path)

            layer_key = f"model.layers.{self.layer_idx}.self_attn.{self.projection_name}"
            lora_a_key = f"{layer_key}.lora_A.default.weight"
            lora_b_key = f"{layer_key}.lora_B.default.weight"

            if lora_a_key in adapter_weights and lora_b_key in adapter_weights:
                lora_a_weights.append(adapter_weights[lora_a_key])
                lora_b_weights.append(adapter_weights[lora_b_key])

                # Calculate scaling factor
                scaling = adapter.lora_alpha / adapter.r
                if adapter.use_rslora:
                    scaling = scaling / math.sqrt(adapter.r)
                scaling_factors.append(scaling * adapter.scaling_factor)
            else:
                lora_a_weights.append(torch.zeros(adapter.r, self.in_features))
                lora_b_weights.append(torch.zeros(self.out_features, adapter.r))
                scaling_factors.append(0.0)

        # Stack weights along adapter dimension
        max_rank = self.lora_config.max_lora_rank

        # Pad smaller ranks to max_rank
        padded_lora_a = []
        padded_lora_b = []

        for i, (lora_a, lora_b) in enumerate(zip(lora_a_weights, lora_b_weights)):
            current_rank = lora_a.shape[0]
            if current_rank < max_rank:
                # Pad with zeros
                padded_a = torch.zeros(max_rank, self.in_features)
                padded_b = torch.zeros(self.out_features, max_rank)
                padded_a[:current_rank] = lora_a
                padded_b[:, :current_rank] = lora_b
                padded_lora_a.append(padded_a)
                padded_lora_b.append(padded_b)
            else:
                padded_lora_a.append(lora_a)
                padded_lora_b.append(lora_b)

        # Stack along adapter dimension
        self.register_buffer("lora_a_weights", torch.stack(padded_lora_a, dim=self.lora_config.lora_a_stack_dim))
        self.register_buffer("lora_b_weights", torch.stack(padded_lora_b, dim=self.lora_config.lora_b_stack_dim))
        self.register_buffer("scaling_factors", torch.tensor(scaling_factors, dtype=torch.float32))

    def forward(self, x: torch.Tensor, adapter_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass that combines base linear transformation with LoRA.

        Args:
            x: Input tensor [batch_size, seq_len, in_features]
            adapter_id: Adapter ID tensor [batch_size] indicating which adapter to use

        Returns:
            Output tensor [batch_size, seq_len, out_features]
        """
        if self._should_apply_lora() and adapter_id is not None:
            # Use custom op to apply base linear + LoRA in one operation
            output = torch.ops.rbln_custom_ops.lora_linear_fused(
                x,
                self.weight,
                self.bias,
                self.lora_a_weights,
                self.lora_b_weights,
                self.scaling_factors,
                adapter_id,
            )
        else:
            # Standard linear transformation
            output = torch.nn.functional.linear(x, self.weight, self.bias)

        return output
