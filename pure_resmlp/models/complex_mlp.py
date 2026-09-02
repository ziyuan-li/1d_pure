"""Complex-valued MLP baseline."""
import torch
import torch.nn as nn

from pure_resmlp.models.complex_readout import ComplexRealProjection

class ComplexReLU(nn.Module):
    """
    复数 ReLU (CReLU): 分别对实部和虚部取 ReLU
    """
    def forward(self, z):
        return torch.complex(torch.relu(z.real), torch.relu(z.imag))

class ComplexMLP4Layer(nn.Module):
    """
    复数版本的 MLP：
    所有线性层均使用 torch.complex64
    """
    def __init__(self, input_dim: int, output_dim: int = 1, hidden_dim: int = 128):
        super().__init__()

        # Keep the same depth as the real-valued MLP baseline.
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, dtype=torch.complex64),
            ComplexReLU(),
            nn.Linear(hidden_dim, hidden_dim, dtype=torch.complex64),
            ComplexReLU(),
            nn.Linear(hidden_dim, hidden_dim, dtype=torch.complex64),
            ComplexReLU(),
            nn.Linear(hidden_dim, hidden_dim, dtype=torch.complex64),
            ComplexReLU(),
            nn.Linear(hidden_dim, hidden_dim, dtype=torch.complex64),
            ComplexReLU(),
            nn.Linear(hidden_dim, hidden_dim, dtype=torch.complex64),
            ComplexReLU(),
            nn.Linear(hidden_dim, hidden_dim, dtype=torch.complex64),
            ComplexReLU(),
            nn.Linear(hidden_dim, output_dim, dtype=torch.complex64),
        )

        self.readout = ComplexRealProjection(output_dim)

    def forward(self, x):
        return self.readout(self.net(x))

def build_complex_mlp(input_dim: int, output_dim: int = 1, hidden_dim: int = 128) -> nn.Module:
    return ComplexMLP4Layer(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim)
