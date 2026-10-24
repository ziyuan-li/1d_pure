import torch
from torch import nn


class ComplexRealProjection(nn.Module):
    def __init__(self, output_dim=1):
        super().__init__()
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        self.real_weight = nn.Parameter(torch.ones(output_dim, dtype=torch.float32))
        self.imag_weight = nn.Parameter(torch.zeros(output_dim, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(output_dim, dtype=torch.float32))

    def forward(self, z):
        if not torch.is_complex(z):
            raise TypeError("ComplexRealProjection expects a complex tensor")
        return self.real_weight * z.real + self.imag_weight * z.imag + self.bias
