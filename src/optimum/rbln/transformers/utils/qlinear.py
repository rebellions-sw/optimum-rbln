from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class QLinear(nn.Module):
    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        weight_scale: Optional[torch.Tensor] = None,
        input_scale: Optional[torch.Tensor] = None,
        # FIXME(jongho): Make it only holds k_scale or v_scale
        k_scale: Optional[torch.Tensor] = None,
        v_scale: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        self.weight = weight
        self.bias = bias
        self.weight_scale = weight_scale
        self.input_scale = input_scale
        self.k_scale = k_scale
        self.v_scale = v_scale

        if weight_scale is None:
            raise ValueError("weight_scale is required")

    def dtype(self) -> torch.dtype:
        return self.weight.dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class QIntLinear(QLinear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_scale:
            raise NotImplementedError("Input scale is currently not supported for int quantization")

        weight = self.weight * self.weight_scale
        return F.linear(x, weight, self.bias)


class QFloatLinear(QLinear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_scale:
            finfo = torch.finfo(self.dtype())
            x = (x / self.input_scale).clamp(min=finfo.min, max=finfo.max)

        weight = self.weight.to(self.weight_scale.dtype) * self.weight_scale

        return F.linear(x, weight, self.bias)
