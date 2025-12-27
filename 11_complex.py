# 11_complex.py
import argparse
import os
import random
import numpy as np
import pandas as pd
from sklearn.datasets import make_friedman1, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# 导入复数模型
from complex_pu_model import ComplexPUResNet1D
from complex_mlp_models import build_complex_mlp  # 新增导入


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --- 数据加载逻辑 (已对齐 01.py) ---
def load_friedman():
    X, y = make_friedman1(n_samples=10000, n_features=10, noise=0.0, random_state=0)
    return train_test_split_all(X, y, test_size=0.30)


def load_concrete():
    df = pd.read_excel("./data/concrete+compressive+strength/Concrete_Data.xls")
    X, y = df.iloc[:, :8].values, df.iloc[:, 8].values
    return train_test_split_all(X, y, test_size=0.40)


def load_california():
    data = fetch_california_housing()
    return train_test_split_all(data.data, data.target, test_size=0.20)


def train_test_split_all(X, y, test_size):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=0)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=0)
    return X_train, X_val, X_test, y_train, y_val, y_test


def standardize_splits(X_train, X_val, X_test, y_train, y_val, y_test):
    x_scaler, y_scaler = StandardScaler(), StandardScaler()
    X_train_s = x_scaler.fit_transform(X_train)
    X_val_s, X_test_s = x_scaler.transform(X_val), x_scaler.transform(X_test)
    y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_s, y_test_s = y_scaler.transform(y_val.reshape(-1, 1)).ravel(), y_scaler.transform(
        y_test.reshape(-1, 1)).ravel()
    return (X_train_s, X_val_s, X_test_s, y_train_s, y_val_s, y_test_s, x_scaler, y_scaler)


def make_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test, batch_size=128):
    X_train_t = torch.from_numpy(X_train).to(torch.complex64)
    X_val_t = torch.from_numpy(X_val).to(torch.complex64)
    X_test_t = torch.from_numpy(X_test).to(torch.complex64)
    y_train_t = torch.from_numpy(y_train).float().unsqueeze(-1)
    y_val_t = torch.from_numpy(y_val).float().unsqueeze(-1)
    y_test_t = torch.from_numpy(y_test).float().unsqueeze(-1)
    return (
        DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True),
        DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False),
        DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=batch_size, shuffle=False)
    )


# --- 训练与评估逻辑 ---
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X).real  # 仅使用实部与标签对比
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
    return total_loss / len(loader.dataset)


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X).real
            loss = criterion(pred, y)
            total_loss += loss.item() * X.size(0)
    return total_loss / len(loader.dataset)


# --- 主程序 ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="friedman", choices=["friedman", "concrete", "california"])
    parser.add_argument("--model", type=str, default="complex_mlp", choices=["complex_pu", "complex_mlp"])  # 增加模型选择
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=64)  # 对齐 01.py 默认值
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--save_dir", type=str, default="./checkpoints_complex_test")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == "friedman":
        data = load_friedman()
    elif args.dataset == "concrete":
        data = load_concrete()
    else:
        data = load_california()

    (X_tr, X_va, X_te, y_tr, y_va, y_te, x_s, y_s) = standardize_splits(*data)
    train_loader, val_loader, test_loader = make_dataloaders(X_tr, X_va, X_te, y_tr, y_va, y_te, args.batch_size)

    input_dim, criterion = X_tr.shape[1], nn.MSELoss()
    all_test_losses = []

    for run in range(args.runs):
        print(f"\n==== Run {run + 1}/{args.runs}, model={args.model}, dataset={args.dataset} ====")

        # 动态初始化模型逻辑
        if args.model == "complex_pu":
            model = ComplexPUResNet1D(input_dim=input_dim, hidden_dim=args.hidden_dim).to(device)
            model_tag = f"complex_pu_h{args.hidden_dim}"
        else:
            model = build_complex_mlp(input_dim=input_dim, hidden_dim=args.hidden_dim).to(device)
            model_tag = f"complex_mlp_h{args.hidden_dim}"

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        best_val_loss, best_state = float('inf'), None
        train_losses, val_losses = [], []

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss = eval_one_epoch(model, val_loader, criterion, device)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}

            if epoch % 10 == 0:
                print(f"Epoch {epoch:03d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        model.load_state_dict(best_state)
        test_loss = eval_one_epoch(model, test_loader, criterion, device)
        all_test_losses.append(test_loss)
        print(f"Best Val Loss: {best_val_loss:.4f} | Test Loss: {test_loss:.4f}")

        # 保存结果
        ckpt_path = os.path.join(args.save_dir, f"{model_tag}_{args.dataset}_run{run}_best.pth")
        torch.save({"model_type": args.model, "model_state_dict": best_state}, ckpt_path)
        np.savez(os.path.join(args.save_dir, f"{model_tag}_{args.dataset}_run{run}_losses.npz"),
                 train_losses=np.array(train_losses), val_losses=np.array(val_losses))

    all_test_losses = np.array(all_test_losses)
    print(f"\n==== Summary [{args.dataset}][{args.model}] over {args.runs} runs ====")
    print(f"Final Result: {all_test_losses.mean():.4f} ± {all_test_losses.std():.4f}")


if __name__ == "__main__":
    main()