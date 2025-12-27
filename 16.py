# 16_complex_train_subsample.py
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

from complex_pu_model import ComplexPUResNet1D
from complex_mlp_models import build_complex_mlp


# ==========================================
# 1) 数据加载（raw，不标准化）
# ==========================================
def load_data_raw(dataset_name):
    if dataset_name == "friedman":
        X, y = make_friedman1(n_samples=10000, n_features=10, noise=0.0, random_state=0)
    elif dataset_name == "concrete":
        df = pd.read_excel("./data/concrete+compressive+strength/Concrete_Data.xls")
        X = df.iloc[:, :8].values
        y = df.iloc[:, 8].values
    else:  # california
        data = fetch_california_housing()
        X, y = data.data, data.target
    return X, y


def get_data_splits(X, y, fraction, seed):
    """
    1) 固定 Test Set: 20% (random_state=0) -> 所有实验公平
    2) 从剩余 Train 里切 Val: 20% (random_state=0)
    3) 对 Train 再按 fraction 采样（seed=run）
    """
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.20, random_state=0
    )
    X_train_full, X_val_full, y_train_full, y_val_full = train_test_split(
        X_train_full, y_train_full, test_size=0.20, random_state=0
    )

    if fraction < 1.0:
        n_samples = max(1, int(X_train_full.shape[0] * fraction))
        rng = np.random.RandomState(seed)
        idx = rng.choice(X_train_full.shape[0], n_samples, replace=False)
        X_train_sub = X_train_full[idx]
        y_train_sub = y_train_full[idx]
    else:
        X_train_sub = X_train_full
        y_train_sub = y_train_full

    return X_train_sub, X_val_full, X_test, y_train_sub, y_val_full, y_test


# ==========================================
# 2) 训练/评估（复数：输入转 complex，输出取 real）
# ==========================================
def forward_model(model, X_real):
    # X_real: float tensor [B, D]
    X_c = X_real.to(torch.complex64)  # imag=0
    y = model(X_c)
    return y.real  # 回归：取实部


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total = 0.0
    n = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = forward_model(model, X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total += loss.item() * X.size(0)
        n += X.size(0)
    return total / max(1, n)


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total = 0.0
    n = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred = forward_model(model, X)
        loss = criterion(pred, y)
        total += loss.item() * X.size(0)
        n += X.size(0)
    return total / max(1, n)


# ==========================================
# 3) main
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="concrete", choices=["friedman", "concrete", "california"])
    parser.add_argument("--fraction", type=float, default=0.6, help="训练数据比例 (0.4 - 1.0)")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=64)

    parser.add_argument("--save_dir", type=str, default="./checkpoints_efficiency_complex_init0")
    parser.add_argument("--train_mlp", action="store_true", default=True, help="同时训练 Complex MLP（默认只训练 PU）")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Start Complex Training | Dataset={args.dataset} | Fraction={args.fraction} | Runs={args.runs}")

    # raw data
    X_raw, y_raw = load_data_raw(args.dataset)
    input_dim = X_raw.shape[1]

    # 默认：只跑 PU；如需对比，带 --train_mlp
    models_to_train = [

    ]
    if args.train_mlp:
        models_to_train.insert(0, ("complex_mlp", f"complex_mlp_h{args.hidden_dim}",
                                   lambda: build_complex_mlp(input_dim=input_dim, output_dim=1, hidden_dim=args.hidden_dim)))

    for run in range(args.runs):
        # split (subsample train)
        X_train, X_val, X_test, y_train, y_val, y_test = get_data_splits(X_raw, y_raw, args.fraction, seed=run)

        # standardize (fit on train)
        x_scaler = StandardScaler().fit(X_train)
        y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))

        X_train_s = x_scaler.transform(X_train)
        X_val_s = x_scaler.transform(X_val)
        X_test_s = x_scaler.transform(X_test)

        y_train_s = y_scaler.transform(y_train.reshape(-1, 1)).ravel()
        y_val_s = y_scaler.transform(y_val.reshape(-1, 1)).ravel()
        y_test_s = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

        # dataloaders
        train_ds = TensorDataset(torch.tensor(X_train_s).float(), torch.tensor(y_train_s).float().unsqueeze(1))
        val_ds = TensorDataset(torch.tensor(X_val_s).float(), torch.tensor(y_val_s).float().unsqueeze(1))
        test_ds = TensorDataset(torch.tensor(X_test_s).float(), torch.tensor(y_test_s).float().unsqueeze(1))

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

        for m_key, m_tag, m_builder in models_to_train:
            model = m_builder().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.MSELoss()

            best_val = float("inf")
            best_state = None

            for _epoch in range(args.epochs):
                _ = train_one_epoch(model, train_loader, criterion, optimizer, device)
                val_loss = eval_one_epoch(model, val_loader, criterion, device)
                if val_loss < best_val:
                    best_val = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            if best_state is not None:
                model.load_state_dict(best_state)

            test_mse = eval_one_epoch(model, test_loader, criterion, device)

            save_name = f"{m_tag}_{args.dataset}_frac{args.fraction}_run{run}_best.pth"
            save_path = os.path.join(args.save_dir, save_name)

            torch.save({
                "model_state_dict": best_state,
                "input_dim": input_dim,
                "hidden_dim": args.hidden_dim,
                "fraction": args.fraction,
                "test_mse": float(test_mse),

                "x_scaler_mean": x_scaler.mean_,
                "x_scaler_scale": x_scaler.scale_,
                "y_scaler_mean": y_scaler.mean_,
                "y_scaler_scale": y_scaler.scale_,
            }, save_path)

            print(f"  [Run {run}][{m_key}] Best Val: {best_val:.6f} | Test MSE: {test_mse:.6f} -> {save_name}")


if __name__ == "__main__":
    main()
