import torch
from torch import nn

from .initialization import make_linear


class ProductUnit1D(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = make_linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        return torch.exp(self.linear(torch.log(torch.relu(x) + 1)))


class PUResidualBlock1D(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc1 = make_linear(hidden_dim, hidden_dim)
        self.pu = ProductUnit1D(hidden_dim, hidden_dim)
        self.fc2 = make_linear(hidden_dim, hidden_dim)

    def forward(self, x):
        return x + self.fc2(self.pu(self.fc1(x)))


class RealPUResNet1D(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=1):
        super().__init__()
        self.input_fc = make_linear(input_dim, hidden_dim)
        self.block1 = PUResidualBlock1D(hidden_dim)
        self.block2 = PUResidualBlock1D(hidden_dim)
        self.out_fc = make_linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.relu(self.input_fc(x))
        h = self.block1(h)
        h = self.block2(h)
        return self.out_fc(h)


def build_real_pu_resnet1d(input_dim, hidden_dim=128, output_dim=1):
    return RealPUResNet1D(input_dim, hidden_dim, output_dim)
