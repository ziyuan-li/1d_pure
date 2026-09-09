"""Real-valued readout layers for complex-valued regression networks."""

import torch
import torch.nn as nn


class ComplexRealProjection(nn.Module):
    """Compute a * Re(z) + b * Im(z) + c for signed regression targets.

    Initialize (a, b, c) = (1, 0, 0). The real component passes gradients
    through a zero-initialized output layer.
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
