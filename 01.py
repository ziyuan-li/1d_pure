# train_mlp.py
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

from mlp_models import build_4layer_mlp
from pu_models import RealPUResNet1D   # 你也可以改成从 build_real_pu_resnet1d 导入


def set_seed(seed: int):
    """为每次 run 固定随机种子，保证可复现（当前版本中不再使用）。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 保证 determinism（会稍微慢一点，可以按需关掉）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_friedman():
    """Friedman Synthetic: Train/Val/Test = 70/15/15（划分固定）"""
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
    """Concrete Strength: Train/Val/Test = 60/20/20（划分固定）"""
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
    """California Housing: Train/Val/Test = 80/10/10（划分固定）"""
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
        default="california",
        choices=["friedman", "concrete", "california"],
        help="选择数据集"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlp",
        choices=["mlp", "pu"],
        help="选择模型类型：普通 MLP 或 PU 残差网络"
    )
    parser.add_argument("--hidden_dim", type=int, default=128,
                        help="MLP 的隐藏维度（默认 128）")
    parser.add_argument("--pu_hidden_dim", type=int, default=128,
                        help="PU 网络的隐藏维度（默认 128）")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_dir", type=str, default="./checkpoints_test_ini0",
                        help="保存最佳模型的目录")
    parser.add_argument("--runs", type=int, default=5,
                        help="同一配置重复训练次数（默认 5 次）")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # 1. 加载 & 划分数据（只做一次，每个 run 共用同一划分；random_state 固定）
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

    input_dim = X_train_s.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()

    all_test_losses = []

    # 多次训练（不同随机种子）
    for run in range(args.runs):
        print(f"\n==== Run {run + 1}/{args.runs}, model={args.model}, dataset={args.dataset} ====")

        # 4. 构建模型
        if args.model == "mlp":
            hidden_dim = args.hidden_dim
            model = build_4layer_mlp(
                input_dim=input_dim,
                output_dim=1,
                hidden_dim=hidden_dim
            )
            model_tag = f"mlp4_h{hidden_dim}"
        else:  # PU 模型
            hidden_dim = args.pu_hidden_dim
            model = RealPUResNet1D(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=1
            )
            model_tag = f"puresnet_h{hidden_dim}"

        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        best_val_loss = np.inf
        best_state = None
        train_losses = []
        val_losses = []

        # 5. 训练循环
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss = eval_one_epoch(
                model, val_loader, criterion, device
            )

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(
                f"[{args.dataset}][{args.model}][run {run + 1}] "
                f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
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
        all_test_losses.append(test_loss)

        print(
            f"[{args.dataset}][{args.model}][run {run + 1}] "
            f"Best val_loss={best_val_loss:.4f} | test_loss={test_loss:.4f}"
        )

        # 7. 保存 checkpoint（每个 run 单独保存）
        ckpt_path = os.path.join(
            args.save_dir,
            f"{model_tag}_{args.dataset}_run{run}_best.pth"
        )
        torch.save(
            {
                "model_type": args.model,
                "model_state_dict": best_state,
                "input_dim": input_dim,
                "hidden_dim": hidden_dim,
                "seed": None,  # 当前版本不再使用固定 seed
            },
            ckpt_path
        )
        print(f"已保存最佳模型到: {ckpt_path}")

        # 8. 保存当前 run 的 loss 曲线数据
        loss_path = os.path.join(
            args.save_dir,
            f"{model_tag}_{args.dataset}_run{run}_losses.npz"
        )
        np.savez(
            loss_path,
            train_losses=np.array(train_losses),
            val_losses=np.array(val_losses),
        )
        print(f"已保存 loss 曲线到: {loss_path}")

    # 9. 打印多次训练的 test loss 统计
    all_test_losses = np.array(all_test_losses)
    print(f"\n==== Summary [{args.dataset}][{args.model}] over {args.runs} runs ====")
    print("Test losses:", all_test_losses)
    print("Mean test loss:", all_test_losses.mean())
    print("Std  test loss:", all_test_losses.std())


if __name__ == "__main__":
    main()
