# test_mlp.py
import argparse

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


def make_test_loader(X_test, y_test, batch_size=128):
    X_test_t = torch.from_numpy(X_test).float()
    y_test_t = torch.from_numpy(y_test).float().unsqueeze(-1)

    test_ds = TensorDataset(X_test_t, y_test_t)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return test_loader


def eval_on_test(model, loader, device):
    criterion = nn.MSELoss()
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
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="保存的 MLP4 checkpoint 路径（默认 ./checkpoints/mlp4_{dataset}_best.pth）"
    )
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    # 1. 选择 checkpoint 路径
    if args.ckpt_path is None:
        ckpt_path = f"./checkpoints/mlp4_{args.dataset}_best.pth"
    else:
        ckpt_path = args.ckpt_path

    print(f"加载 checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    input_dim_ckpt = ckpt["input_dim"]
    hidden_dim_ckpt = ckpt["hidden_dim"]
    state_dict = ckpt["model_state_dict"]

    # 2. 重新加载数据并按相同方式划分 + 标准化
    if args.dataset == "friedman":
        X_train, X_val, X_test, y_train, y_val, y_test = load_friedman()
    elif args.dataset == "concrete":
        X_train, X_val, X_test, y_train, y_val, y_test = load_concrete()
    else:
        X_train, X_val, X_test, y_train, y_val, y_test = load_california()

    (X_train_s, X_val_s, X_test_s,
     y_train_s, y_val_s, y_test_s,
     x_scaler, y_scaler) = standardize_splits(
        X_train, X_val, X_test,
        y_train, y_val, y_test
    )

    # 检查 input_dim 一致性
    input_dim_data = X_train_s.shape[1]
    assert input_dim_data == input_dim_ckpt, \
        f"Input dim mismatch: data={input_dim_data}, ckpt={input_dim_ckpt}"

    # 3. 构建模型并加载参数
    model = build_4layer_mlp(
        input_dim=input_dim_ckpt,
        output_dim=1,
        hidden_dim=hidden_dim_ckpt
    )
    model.load_state_dict(state_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 4. 只在测试集上评估
    test_loader = make_test_loader(X_test_s, y_test_s, batch_size=args.batch_size)
    test_loss = eval_on_test(model, test_loader, device)

    print(f"[{args.dataset}] Loaded checkpoint test_loss={test_loss:.4f}")


if __name__ == "__main__":
    main()
