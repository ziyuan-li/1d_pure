"""Real-valued MLP baseline."""
import torch
import torch.nn as nn


class MLP4Layer(nn.Module):
    """Seven ReLU hidden layers followed by a linear output.

    The historical class name is retained for checkpoint compatibility.
    """
    def __init__(self, input_dim: int, output_dim: int = 1,
                 hidden_dim: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def build_4layer_mlp(input_dim: int,
                     output_dim: int = 1,
                     hidden_dim: int = 128) -> nn.Module:
    """Build the seven-hidden-layer MLP baseline."""
    return MLP4Layer(input_dim=input_dim,
                     output_dim=output_dim,
                     hidden_dim=hidden_dim)
