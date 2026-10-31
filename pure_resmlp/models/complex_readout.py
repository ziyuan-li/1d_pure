import torch
import torch.nn as nn

class ComplexRealProjection(nn.Module):

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        if output_dim <= 0:
            raise ValueError('output_dim must be positive')
        self.real_weight = nn.Parameter(torch.ones(output_dim, dtype=torch.float32))
        self.imag_weight = nn.Parameter(torch.zeros(output_dim, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(output_dim, dtype=torch.float32))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(z):
            raise TypeError('ComplexRealProjection expects a complex tensor')
        return z.real * self.real_weight + z.imag * self.imag_weight + self.bias
