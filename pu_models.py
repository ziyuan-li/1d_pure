# pu_models.py
import torch
import torch.nn as nn


class ProductUnit1D(nn.Module):
    """
    PU(x) = exp( Linear(log(|x| + eps)) )
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.eps = 1e-8
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(x) + 1
        x_log = torch.log(x)
        return torch.exp(self.linear(x_log))


class PUResidualBlock1D(nn.Module):
    """
    残差块 (hidden_dim → hidden_dim):
        identity = x
        x_fc = FC(x)
        x_pu = PU(x_fc)
        out = identity + x_pu
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.pu = ProductUnit1D(hidden_dim, hidden_dim)

    def forward(self, x):
        identity = x
        x_fc = self.fc(x)
        x_pu = self.pu(x_fc)
        return identity + x_pu


class RealPUResNet1D(nn.Module):
    """
    网络结构：
        x → Linear(input_dim → hidden_dim)
          → Block1
          → Block2
          → Linear(hidden_dim → output_dim)
    """
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super().__init__()

        # 输入映射到 hidden_dim
        self.input_fc = nn.Linear(input_dim, hidden_dim)

        # 两个残差块
        self.block1 = PUResidualBlock1D(hidden_dim)
        self.block2 = PUResidualBlock1D(hidden_dim)

        # 输出层
        self.out_fc = nn.Linear(hidden_dim, output_dim)

        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.input_fc(x)  # input_dim → hidden_dim
        h = self.relu(h)
        h = self.block1(h)
        h = self.block2(h)
        y = self.out_fc(h)    # hidden_dim → output_dim
        return y


def build_real_pu_resnet1d(input_dim, hidden_dim, output_dim=1):
    """
    工厂函数：供训练脚本调用
    """
    return RealPUResNet1D(input_dim, hidden_dim, output_dim)
