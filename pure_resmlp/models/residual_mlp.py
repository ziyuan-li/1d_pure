"""Residual MLP used by the ablation study."""

import torch
import torch.nn as nn


class MLPResidualBlock(nn.Module):
    """
    仿照 PUResidualBlock1D 设计的 MLP 残差块:
    out = x + FC2(ReLU(FC1(x)))
    """

    def __init__(self, hidden_dim, ini0=False):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)


        # 如果开启 ini0，将最后一层初始化为0，使残差分支初始输出为0
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
    """
    网络结构参考 RealPUResNet1D：
    x -> Linear(input_dim -> hidden_dim) -> ReLU
      -> Block1 (Residual)
      -> Block2 (Residual, Zero Initialized)
      -> Linear(hidden_dim -> output_dim)
    """

    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super().__init__()

        # 1. 输入投影层：将原始特征映射到隐藏层维度
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()

        # 2. 残差层：这里设置两个块，第二个块采用零初始化
        self.block1 = MLPResidualBlock(hidden_dim)
        self.block2 = MLPResidualBlock(hidden_dim, ini0=True)

        # 3. 输出层
        self.out_fc = nn.Linear(hidden_dim, output_dim)

        # 仿照 PU 模型，输出层也可以初始化为 0（可选）
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
    """
    工厂函数，方便训练脚本调用
    """
    return MLPResNet(input_dim, hidden_dim, output_dim)
