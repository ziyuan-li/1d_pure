import torch
import torch.nn as nn
import math


class ComplexProductUnit1D(nn.Module):
    """
    复数乘积单元 (修改版):
    通过分别对实部和虚部进行 ReLU + sqrt(0.5) 处理，
    确保 log(z) 运算的数值稳定性。
    """

    def __init__(self, in_dim, out_dim, ini0=False):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False, dtype=torch.complex64)

        if ini0:
            nn.init.zeros_(self.linear.weight)

        # 原有的 threshold 参数不再需要，改为固定常数偏移
        self.offset = math.sqrt(0.5)

    def forward(self, z):
        # 1. 分离实部和虚部
        z_real = z.real
        z_imag = z.imag

        # 2. 分别应用 ReLU 并加上 sqrt(0.5)
        # 这样确保了新的实部和虚部均 >= 0.707...
        safe_real = torch.relu(z_real) + self.offset
        safe_imag = torch.relu(z_imag) + self.offset

        # 3. 构造安全的复数并计算复数对数
        # torch.log(z) 对于复数会自动计算 ln|z| + i*arg(z)
        safe_z = torch.complex(safe_real, safe_imag)
        z_log = torch.log(safe_z)

        # 4. 线性变换后取指数：exp(W * log(z)) 相当于 z^W
        return torch.exp(self.linear(z_log))


# 其余 Block 和 ResNet 结构保持不变
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

        nn.init.zeros_(self.out_fc.weight)
        nn.init.zeros_(self.out_fc.bias)

    def forward(self, z):
        h = self.input_fc(z)
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        return self.out_fc(h)