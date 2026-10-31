import torch
import torch.nn as nn

class MLPResidualBlock(nn.Module):

    def __init__(self, hidden_dim, ini0=False):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        if ini0:
            self.fc3.weight.data.zero_()
            self.fc3.bias.data.zero_()

    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return identity + out

class MLPResNet(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.block1 = MLPResidualBlock(hidden_dim)
        self.block2 = MLPResidualBlock(hidden_dim, ini0=True)
        self.out_fc = nn.Linear(hidden_dim, output_dim)
        self.out_fc.weight.data.zero_()
        self.out_fc.bias.data.zero_()

    def forward(self, x):
        h = self.input_fc(x)
        h = self.relu(h)
        h = self.block1(h)
        h = self.block2(h)
        h = self.relu(h)
        y = self.out_fc(h)
        return y

def build_mlp_resnet(input_dim, hidden_dim, output_dim=1):
    return MLPResNet(input_dim, hidden_dim, output_dim)
