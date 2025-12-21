# test_zero_init.py
import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# 假设你已经按照上面的修改更新了 pu_models.py
from pu_models import RealPUResNet1D


def eval_model(model, loader, device):
    model.eval()
    criterion = torch.nn.MSELoss()
    total_loss = 0.0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            total_loss += criterion(pred, y).item() * X.size(0)
    return total_loss / len(loader.dataset)


def main():
    # 简单的测试流程：加载 California 数据集
    data = fetch_california_housing()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=0)

    scaler_x = StandardScaler()
    X_test_s = scaler_x.fit_transform(X_test)
    scaler_y = StandardScaler()
    y_test_s = scaler_y.fit_transform(y_test.reshape(-1, 1)).ravel()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型（应用了零初始化策略）
    model = RealPUResNet1D(input_dim=8, hidden_dim=128, output_dim=1).to(device)

    # 验证初始状态下的输出
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test_s).float(), torch.from_numpy(y_test_s).float().unsqueeze(-1)),
        batch_size=128
    )

    initial_loss = eval_model(model, test_loader, device)
    print(f"模型已完成‘零初始化’修改。")
    print(f"初始状态（未经训练）在测试集上的 MSE Loss: {initial_loss:.4f}")

    # 检查 gamma 参数是否生效
    print(f"Block1 Gamma: {model.block1.gamma.item():.4f}")
    print(f"Block2 Gamma: {model.block2.gamma.item():.4f}")


if __name__ == "__main__":
    main()