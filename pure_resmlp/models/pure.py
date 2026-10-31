import torch
import torch.nn as nn

class ProductUnit1D(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(x) + 1
        x_log = torch.log(x)
        return torch.exp(self.linear(x_log))

class PUResidualBlock1D(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.pu = ProductUnit1D(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        identity = x
        x_fc1 = self.fc1(x)
        x_pu = self.pu(x_fc1)
        x_fc2 = self.fc2(x_pu)
        return identity + x_fc2

class RealPUResNet1D(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.block1 = PUResidualBlock1D(hidden_dim)
        self.block2 = PUResidualBlock1D(hidden_dim)
        self.out_fc = nn.Linear(hidden_dim, output_dim)
        self.out_fc.weight.data.zero_()
        self.out_fc.bias.data.zero_()
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.input_fc(x)
        h = self.relu(h)
        h = self.block1(h)
        h = self.block2(h)
        y = self.out_fc(h)
        return y

def build_real_pu_resnet1d(input_dim, hidden_dim, output_dim=1):
    return RealPUResNet1D(input_dim, hidden_dim, output_dim)
