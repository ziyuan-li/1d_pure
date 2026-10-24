import math

import torch
from torch import nn

from .complex_readout import ComplexRealProjection
from .initialization import make_linear


class ComplexProductUnit1D(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = make_linear(
            in_dim, out_dim, bias=False, dtype=torch.complex64
        )
        self.offset = 1 / math.sqrt(2)

    def forward(self, z):
        shifted = torch.complex(
            torch.relu(z.real) + self.offset,
            torch.relu(z.imag) + self.offset,
        )
        return torch.exp(self.linear(torch.log(shifted)))


class ComplexPUResidualBlock1D(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc1 = make_linear(hidden_dim, hidden_dim, dtype=torch.complex64)
        self.pu = ComplexProductUnit1D(hidden_dim, hidden_dim)
        self.fc2 = make_linear(hidden_dim, hidden_dim, dtype=torch.complex64)

    def forward(self, z):
        return z + self.fc2(self.pu(self.fc1(z)))


class ComplexPUResNet1D(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super().__init__()
        self.input_fc = make_linear(input_dim, hidden_dim, dtype=torch.complex64)
        self.block1 = ComplexPUResidualBlock1D(hidden_dim)
        self.block2 = ComplexPUResidualBlock1D(hidden_dim)
        self.out_fc = make_linear(hidden_dim, output_dim, dtype=torch.complex64)
        self.readout = ComplexRealProjection(output_dim)

    def forward(self, x):
        z = x.to(dtype=self.input_fc.weight.dtype)
        h = self.input_fc(z)
        h = self.block1(h)
        h = self.block2(h)
        return self.readout(self.out_fc(h))
