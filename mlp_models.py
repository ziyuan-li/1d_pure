# mlp_models.py
import torch
import torch.nn as nn


class MLP4Layer(nn.Module):
    """
    一个简单的四层 MLP：
    input_dim -> hidden_dim -> hidden_dim -> hidden_dim -> hidden_dim -> output_dim
    中间每层都是 Linear + ReLU
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
    """
    返回一个四层 MLP 模型，方便在训练脚本中直接调用
    """
    return MLP4Layer(input_dim=input_dim,
                     output_dim=output_dim,
                     hidden_dim=hidden_dim)
