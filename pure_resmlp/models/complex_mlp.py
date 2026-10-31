import torch
import torch.nn as nn
from pure_resmlp.models.complex_readout import ComplexRealProjection

class ComplexReLU(nn.Module):

    def forward(self, z):
        return torch.complex(torch.relu(z.real), torch.relu(z.imag))

class ComplexMLPRegressor(nn.Module):

    def __init__(self, input_dim: int, output_dim: int=1, hidden_dim: int=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim, dtype=torch.complex64), ComplexReLU(), nn.Linear(hidden_dim, hidden_dim, dtype=torch.complex64), ComplexReLU(), nn.Linear(hidden_dim, hidden_dim, dtype=torch.complex64), ComplexReLU(), nn.Linear(hidden_dim, hidden_dim, dtype=torch.complex64), ComplexReLU(), nn.Linear(hidden_dim, hidden_dim, dtype=torch.complex64), ComplexReLU(), nn.Linear(hidden_dim, hidden_dim, dtype=torch.complex64), ComplexReLU(), nn.Linear(hidden_dim, hidden_dim, dtype=torch.complex64), ComplexReLU(), nn.Linear(hidden_dim, output_dim, dtype=torch.complex64))
        self.readout = ComplexRealProjection(output_dim)

    def forward(self, x):
        return self.readout(self.net(x))

def build_complex_mlp(input_dim: int, output_dim: int=1, hidden_dim: int=128) -> nn.Module:
    return ComplexMLPRegressor(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim)
