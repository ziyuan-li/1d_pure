import torch
import torch.nn as nn
import math
from pure_resmlp.models.complex_readout import ComplexRealProjection

class ComplexProductUnit1D(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False, dtype=torch.complex64)
        self.offset = math.sqrt(0.5)

    def forward(self, z):
        z_real = z.real
        z_imag = z.imag
        safe_real = torch.relu(z_real) + self.offset
        safe_imag = torch.relu(z_imag) + self.offset
        safe_z = torch.complex(safe_real, safe_imag)
        z_log = torch.log(safe_z)
        return torch.exp(self.linear(z_log))

class ComplexPUResidualBlock1D(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim, dtype=torch.complex64)
        self.pu = ComplexProductUnit1D(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, dtype=torch.complex64)

    def forward(self, z):
        identity = z
        z_out = self.fc1(z)
        z_out = self.pu(z_out)
        z_out = self.fc2(z_out)
        return identity + z_out

class ComplexPUResNet1D(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, hidden_dim, dtype=torch.complex64)
        self.block1 = ComplexPUResidualBlock1D(hidden_dim)
        self.block2 = ComplexPUResidualBlock1D(hidden_dim)
        self.out_fc = nn.Linear(hidden_dim, output_dim, dtype=torch.complex64)
        self.readout = ComplexRealProjection(output_dim)
        nn.init.zeros_(self.out_fc.weight)
        nn.init.zeros_(self.out_fc.bias)

    def forward(self, z):
        h = self.input_fc(z)
        h = self.block1(h)
        h = self.block2(h)
        return self.readout(self.out_fc(h))
