"""Real-valued readout layers for complex-valued regression networks."""

import torch
import torch.nn as nn


class ComplexRealProjection(nn.Module):
    """Project each complex output onto a learnable real-valued axis.

    The projection starts as ``Re(z)`` so gradients remain well-defined when a
    preceding output layer is zero-initialized. During training, the model can
    also use the imaginary component. This supports signed standardized
    regression targets, unlike a magnitude readout.
    """

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        self.real_weight = nn.Parameter(torch.ones(output_dim))
        self.imag_weight = nn.Parameter(torch.zeros(output_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(z):
            raise TypeError("ComplexRealProjection expects a complex tensor")
        return z.real * self.real_weight + z.imag * self.imag_weight + self.bias
