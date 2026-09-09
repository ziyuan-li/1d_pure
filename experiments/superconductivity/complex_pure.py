import torch
import torch.nn as nn
import math

from pure_resmlp.models.complex_readout import ComplexRealProjection


class ComplexProductUnit1D(nn.Module):
    """Complex product unit with a componentwise ReLU + sqrt(0.5) shift.

    Shifted inputs lie in the first quadrant and have modulus at least one.
    """

    def __init__(self, in_dim, out_dim, ini0=False):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False, dtype=torch.complex64)

        if ini0:
            nn.init.zeros_(self.linear.weight)

        self.offset = math.sqrt(0.5)

    def forward(self, z):
        z_real = z.real
        z_imag = z.imag

        # The componentwise shift keeps inputs in the first quadrant with |z| >= 1.
        safe_real = torch.relu(z_real) + self.offset
        safe_imag = torch.relu(z_imag) + self.offset

        # torch.log uses the principal complex logarithm.
        safe_z = torch.complex(safe_real, safe_imag)
        z_log = torch.log(safe_z)

        return torch.exp(self.linear(z_log))


class ComplexPUResidualBlock1D(nn.Module):
    def __init__(self, hidden_dim, ini0=False):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim, dtype=torch.complex64)
        self.pu = ComplexProductUnit1D(hidden_dim, hidden_dim, ini0)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, dtype=torch.complex64)

        if ini0:
            nn.init.zeros_(self.fc1.weight)
            nn.init.zeros_(self.fc1.bias)
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

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
        self.block2 = ComplexPUResidualBlock1D(hidden_dim, ini0=True)
        self.block3 = ComplexPUResidualBlock1D(hidden_dim, ini0=True)
        self.out_fc = nn.Linear(hidden_dim, output_dim, dtype=torch.complex64)
        self.readout = ComplexRealProjection(output_dim)

        nn.init.zeros_(self.out_fc.weight)
        nn.init.zeros_(self.out_fc.bias)

    def forward(self, z):
        h = self.input_fc(z)
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        return self.readout(self.out_fc(h))
