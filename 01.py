# train_mlp.py
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

from mlp_models import build_4layer_mlp


def load_friedman():
    """Friedman Synthetic: Train/Val/Test = 70/15/15"""
    X, y = make_friedman1(
        n_samples=10000,
        n_features=10,
        noise=0.0,
        random_state=0
    )

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=0
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=0
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def load_concrete():
    """Concrete Strength: Train/Val/Test = 60/20/20"""
    df = pd.read_excel(
        "./data/concrete+compressive+strength/Concrete_Data.xls"
    )
    X = df.iloc[:, :8].values
    y = df.iloc[:, 8].values

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=0
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=0
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def load_california():
    """California Housing: Train/Val/Test = 80/10/10"""
    data = fetch_california_housing()
    X, y = data.data, data.target

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.20, random_state=0
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=0
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def standardize_splits(X_train, X_val, X_test, y_train, y_val, y_test):
    """对 X, y 做标准化，只在训练集上 fit。"""
    x_scaler = StandardScaler()
    X_train_s = x_scaler.fit_transform(X_train)
    X_val_s = x_scaler.transform(X_val)
    X_test_s = x_scaler.transform(X_test)

    y_scaler = StandardScaler()
    y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_s = y_scaler.transform(y_val.reshape(-1, 1)).ravel()
    y_test_s = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

    return (X_train_s, X_val_s, X_test_s,
            y_train_s, y_val_s, y_test_s,
            x_scaler, y_scaler)


def make_dataloaders(X_train, X_val, X_test,
                     y_train, y_val, y_test,
                     batch_size=128):
    """将 numpy 数组转成 PyTorch DataLoader"""
    X_train_t = torch.from_numpy(X_train).float()
    X_val_t = torch.from_numpy(X_val).float()
    X_test_t = torch.from_numpy(X_test).float()

    y_train_t = torch.from_numpy(y_train).float().unsqueeze(-1)
    y_val_t = torch.from_numpy(y_val).float().unsqueeze(-1)
    y_test_t = torch.from_numpy(y_test).float().unsqueeze(-1)

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)
    test_ds = TensorDataset(X_test_t, y_test_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    n = 0
    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

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
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            loss = criterion(pred, y)
            total_loss += loss.item() * X.size(0)
            n += X.size(0)
    return total_loss / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="friedman",
        choices=["friedman", "concrete", "california"],
        help="选择数据集"
    )
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_dir", type=str, default="./checkpoints",
                        help="保存最佳模型的目录")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # 1. 加载 & 划分数据
    if args.dataset == "friedman":
        X_train, X_val, X_test, y_train, y_val, y_test = load_friedman()
    elif args.dataset == "concrete":
        X_train, X_val, X_test, y_train, y_val, y_test = load_concrete()
    else:
        X_train, X_val, X_test, y_train, y_val, y_test = load_california()

    # 2. 标准化
    (X_train_s, X_val_s, X_test_s,
     y_train_s, y_val_s, y_test_s,
     x_scaler, y_scaler) = standardize_splits(
        X_train, X_val, X_test,
        y_train, y_val, y_test
    )

    # 3. DataLoader
    train_loader, val_loader, test_loader = make_dataloaders(
        X_train_s, X_val_s, X_test_s,
        y_train_s, y_val_s, y_test_s,
        batch_size=args.batch_size
    )

    # 4. 模型、优化器、损失
    input_dim = X_train_s.shape[1]
    model = build_4layer_mlp(
        input_dim=input_dim,
        output_dim=1,
        hidden_dim=args.hidden_dim
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = np.inf
    best_state = None

    # 5. 训练循环
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss = eval_one_epoch(
            model, val_loader, criterion, device
        )

        print(
            f"[{args.dataset}] Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
        )

        # 记录最优验证集模型参数
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()

    # 6. 使用最优验证模型在测试集上评估
    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss = eval_one_epoch(
        model, test_loader, criterion, device
    )
    print(
        f"[{args.dataset}] Best val_loss={best_val_loss:.4f} | "
        f"test_loss={test_loss:.4f}"
    )

    # 7. 保存 checkpoint（包含结构必要信息）
    ckpt_path = os.path.join(
        args.save_dir,
        f"mlp4_{args.dataset}_best.pth"
    )
    torch.save(
        {
            "model_state_dict": best_state,
            "input_dim": input_dim,
            "hidden_dim": args.hidden_dim,
        },
        ckpt_path
    )
    print(f"已保存最佳模型到: {ckpt_path}")


if __name__ == "__main__":
    main()
