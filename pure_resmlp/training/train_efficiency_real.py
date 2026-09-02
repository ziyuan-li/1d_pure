"""Train real-valued models for sample-efficiency experiments."""
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_friedman1, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from pure_resmlp.models.mlp import build_4layer_mlp
from pure_resmlp.models.pure import RealPUResNet1D
from pure_resmlp.paths import DATA_DIR, EFFICIENCY_REAL_CHECKPOINT_DIR


def load_data_raw(dataset_name):
    """加载原始数据，不进行标准化"""
    if dataset_name == "friedman":
        X, y = make_friedman1(n_samples=10000, n_features=10, noise=0.0, random_state=0)
    elif dataset_name == "concrete":
        df = pd.read_excel(DATA_DIR / "concrete+compressive+strength" / "Concrete_Data.xls")
        X = df.iloc[:, :8].values
        y = df.iloc[:, 8].values
    else:  # california
        data = fetch_california_housing()
        X, y = data.data, data.target
    return X, y


def get_data_splits(X, y, fraction, seed):
    """
    1. 先划分出固定的 Test Set (保持 20% 不变，用于所有实验公平对比)
    2. 从剩余的 Train Set 中，根据 fraction 抽取子集
    """
    # 固定随机种子划分 Train/Test，保证所有实验用的是同一个测试集
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.20, random_state=0
    )

    # 划分 Validation (从 Full Train 里切 20% 作为 Val)
    X_train_full, X_val_full, y_train_full, y_val_full = train_test_split(
        X_train_full, y_train_full, test_size=0.20, random_state=0
    )

    # === 关键步骤：根据 fraction 对训练集采样 ===
    if fraction < 1.0:
        n_samples = int(X_train_full.shape[0] * fraction)
        # 使用传入的 seed 进行采样，保证同一个 run 内数据固定，不同 run 数据随机
        rng = np.random.RandomState(seed)
        indices = rng.choice(X_train_full.shape[0], n_samples, replace=False)
        X_train_sub = X_train_full[indices]
        y_train_sub = y_train_full[indices]
    else:
        X_train_sub = X_train_full
        y_train_sub = y_train_full

    return X_train_sub, X_val_full, X_test, y_train_sub, y_val_full, y_test


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    n = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        n += X.size(0)
    return total_loss / n


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            loss = criterion(model(X), y)
            total_loss += loss.item() * X.size(0)
            n += X.size(0)
    return total_loss / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="concrete", choices=["friedman", "concrete", "california"])
    parser.add_argument("--fraction", type=float, default=0.8, help="训练数据比例 (0.1 - 1.0)")
    parser.add_argument("--runs", type=int, default=5, help="重复训练次数")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--save_dir", type=str, default=str(EFFICIENCY_REAL_CHECKPOINT_DIR))
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Start Training: Dataset={args.dataset} | Fraction={args.fraction} | Runs={args.runs}")

    # 加载原始数据
    X_raw, y_raw = load_data_raw(args.dataset)
    input_dim = X_raw.shape[1]

    for run in range(args.runs):
        # 1. 获取数据子集
        (X_train, X_val, X_test,
         y_train, y_val, y_test) = get_data_splits(X_raw, y_raw, args.fraction, seed=run)

        # 2. 标准化 (fit on train, transform on val/test)
        x_scaler = StandardScaler().fit(X_train)
        y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))

        # 转换所有数据
        X_train_s = x_scaler.transform(X_train)
        X_val_s = x_scaler.transform(X_val)
        X_test_s = x_scaler.transform(X_test)  # === 新增: 转换测试集 ===

        y_train_s = y_scaler.transform(y_train.reshape(-1, 1)).ravel()
        y_val_s = y_scaler.transform(y_val.reshape(-1, 1)).ravel()
        y_test_s = y_scaler.transform(y_test.reshape(-1, 1)).ravel()  # === 新增: 转换测试集 ===

        # DataLoader
        train_ds = TensorDataset(torch.tensor(X_train_s).float(), torch.tensor(y_train_s).float().unsqueeze(1))
        val_ds = TensorDataset(torch.tensor(X_val_s).float(), torch.tensor(y_val_s).float().unsqueeze(1))
        # === 新增: Test DataLoader ===
        test_ds = TensorDataset(torch.tensor(X_test_s).float(), torch.tensor(y_test_s).float().unsqueeze(1))

        train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)  # === 新增 ===

        # 3. Train both real-valued models used by the paper.
        models_to_train = [
            ("mlp", "mlp4_h128", lambda: build_4layer_mlp(input_dim, 1, 128)),
            ("pu", "puresnet_h128", lambda: RealPUResNet1D(input_dim, 128, 1))
        ]

        for m_type, m_tag, m_builder in models_to_train:
            model = m_builder().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.MSELoss()

            best_val_loss = float('inf')
            best_state = None

            # 训练循环
            for epoch in range(args.epochs):
                train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
                val_loss = eval_one_epoch(model, val_loader, criterion, device)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    }

            # === 新增: 训练结束后，加载最佳模型并在测试集上评估 ===
            if best_state is not None:
                model.load_state_dict(best_state)

            test_loss = eval_one_epoch(model, test_loader, criterion, device)

            # 4. 保存模型 (包含 Scaler 参数 和 test_mse)
            save_name = f"{m_tag}_{args.dataset}_frac{args.fraction}_run{run}_best.pth"
            save_path = os.path.join(args.save_dir, save_name)

            torch.save({
                "model_state_dict": best_state,
                "input_dim": input_dim,
                "hidden_dim": 128,
                "fraction": args.fraction,
                "test_mse": test_loss,  # 保存测试结果供后续分析
                # 关键：保存标准化参数
                "x_scaler_mean": x_scaler.mean_,
                "x_scaler_scale": x_scaler.scale_,
                "y_scaler_mean": y_scaler.mean_,
                "y_scaler_scale": y_scaler.scale_
            }, save_path)

            print(f"  [Run {run}][{m_type}] Best Val: {best_val_loss:.4f} | Test MSE: {test_loss:.4f} -> Saved.")


if __name__ == "__main__":
    main()
