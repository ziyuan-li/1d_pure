# test_mlp.py
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
import matplotlib.pyplot as plt

from mlp_models import build_4layer_mlp
from pu_models import RealPUResNet1D


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


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="concrete",
        choices=["friedman", "concrete", "california"],
        help="选择数据集"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./checkpoints",
        help="checkpoint 和 loss 文件所在目录"
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--runs", type=int, default=5,
                        help="每种模型的重复训练次数（默认 5）")
    args = parser.parse_args()

    # 1. 加载数据并划分 + 标准化（与训练保持完全一致）
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

    input_dim_data = X_train_s.shape[1]
    test_loader = make_test_loader(X_test_s, y_test_s, batch_size=args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 两种模型配置：和 train 脚本里的 model_tag 对应
    model_configs = [
        ("mlp", "mlp4_h128"),
        ("pu", "puresnet_h128"),
    ]

    # 用来存储画 loss 曲线的数据：model_type -> {'train': [runs], 'val': [runs]}
    loss_curves = {
        "mlp": {"train": [], "val": []},
        "pu": {"train": [], "val": []},
    }

    for model_type, model_tag in model_configs:
        test_losses = []
        param_count = None

        print(f"\n==== Evaluating [{model_type}] on dataset [{args.dataset}] over {args.runs} runs ====")

        for run in range(args.runs):
            ckpt_path = os.path.join(
                args.save_dir,
                f"{model_tag}_{args.dataset}_run{run}_best.pth"
            )
            loss_path = os.path.join(
                args.save_dir,
                f"{model_tag}_{args.dataset}_run{run}_losses.npz"
            )

            if not os.path.exists(ckpt_path):
                print(f"  [run {run}] 找不到 checkpoint: {ckpt_path}，跳过")
                continue

            ckpt = torch.load(ckpt_path, map_location="cpu")
            input_dim_ckpt = ckpt["input_dim"]
            hidden_dim_ckpt = ckpt["hidden_dim"]
            state_dict = ckpt["model_state_dict"]
            model_type_ckpt = ckpt.get("model_type", model_type)

            # 一致性检查
            assert input_dim_ckpt == input_dim_data, \
                f"Input dim mismatch: data={input_dim_data}, ckpt={input_dim_ckpt}"
            assert model_type_ckpt == model_type, \
                f"Model type mismatch in ckpt: expected {model_type}, got {model_type_ckpt}"

            # 构建模型
            if model_type == "mlp":
                model = build_4layer_mlp(
                    input_dim=input_dim_ckpt,
                    output_dim=1,
                    hidden_dim=hidden_dim_ckpt,
                )
            else:  # "pu"
                model = RealPUResNet1D(
                    input_dim=input_dim_ckpt,
                    hidden_dim=hidden_dim_ckpt,
                    output_dim=1,
                )

            model.load_state_dict(state_dict)
            model.to(device)

            # 第一次记录参数量
            if param_count is None:
                param_count = count_parameters(model)

            # 评估 test loss
            test_loss = eval_on_test(model, test_loader, device)
            test_losses.append(test_loss)
            print(f"  [run {run}] test_loss={test_loss:.4f}")

            # 读取当前 run 的 loss 曲线
            if os.path.exists(loss_path):
                loss_data = np.load(loss_path)
                train_losses = loss_data["train_losses"]
                val_losses = loss_data["val_losses"]
                loss_curves[model_type]["train"].append(train_losses)
                loss_curves[model_type]["val"].append(val_losses)
            else:
                print(f"  [run {run}] 未找到 loss 文件: {loss_path}")

        test_losses = np.array(test_losses)
        if len(test_losses) == 0:
            print(f"  没有找到任何 {model_type} 的 checkpoint，无法统计。")
            continue

        print(f"\n  >> {model_type} 参数数量: {param_count}")
        print(f"  >> {model_type} test_loss (over {len(test_losses)} runs):")
        print(f"     mean = {test_losses.mean():.4f}")
        print(f"     std  = {test_losses.std():.4f}")

    # ===== 画 MLP vs PU 的 loss 曲线对比（取各自 5 次的平均） =====
    # 只在两种模型都有至少一次 loss 曲线时画
    if loss_curves["mlp"]["train"] and loss_curves["pu"]["train"]:
        # 转成 np.array: shape = (runs, epochs)
        mlp_train = np.array(loss_curves["mlp"]["train"])
        mlp_val = np.array(loss_curves["mlp"]["val"])
        pu_train = np.array(loss_curves["pu"]["train"])
        pu_val = np.array(loss_curves["pu"]["val"])

        # 防止不同 run epoch 数稍有不同，统一取最短长度
        min_epochs = min(
            mlp_train.shape[1],
            mlp_val.shape[1],
            pu_train.shape[1],
            pu_val.shape[1],
        )
        mlp_train = mlp_train[:, :min_epochs]
        mlp_val = mlp_val[:, :min_epochs]
        pu_train = pu_train[:, :min_epochs]
        pu_val = pu_val[:, :min_epochs]

        epochs = np.arange(1, min_epochs + 1)

        mlp_train_mean = mlp_train.mean(axis=0)
        mlp_val_mean = mlp_val.mean(axis=0)
        pu_train_mean = pu_train.mean(axis=0)
        pu_val_mean = pu_val.mean(axis=0)

        # ===== 使用 log10(loss) 画 MLP vs PU loss 曲线 =====
        if loss_curves["mlp"]["train"] and loss_curves["pu"]["train"]:
            mlp_train = np.array(loss_curves["mlp"]["train"])
            mlp_val = np.array(loss_curves["mlp"]["val"])
            pu_train = np.array(loss_curves["pu"]["train"])
            pu_val = np.array(loss_curves["pu"]["val"])

            # 统一长度
            min_epochs = min(
                mlp_train.shape[1],
                mlp_val.shape[1],
                pu_train.shape[1],
                pu_val.shape[1],
            )
            mlp_train = mlp_train[:, :min_epochs]
            mlp_val = mlp_val[:, :min_epochs]
            pu_train = pu_train[:, :min_epochs]
            pu_val = pu_val[:, :min_epochs]

            epochs = np.arange(1, min_epochs + 1)

            # 计算 log10(loss)
            eps = 1e-12  # 防止 log(0)
            mlp_train_log = np.log10(mlp_train + eps)
            mlp_val_log = np.log10(mlp_val + eps)
            pu_train_log = np.log10(pu_train + eps)
            pu_val_log = np.log10(pu_val + eps)

            # 求平均
            mlp_train_mean = mlp_train_log.mean(axis=0)
            mlp_val_mean = mlp_val_log.mean(axis=0)
            pu_train_mean = pu_train_log.mean(axis=0)
            pu_val_mean = pu_val_log.mean(axis=0)

            plt.figure(figsize=(7, 5))
            plt.plot(epochs, mlp_train_mean, label="MLP Train (log10)")
            plt.plot(epochs, mlp_val_mean, label="MLP Val (log10)")
            plt.plot(epochs, pu_train_mean, label="PU Train (log10)")
            plt.plot(epochs, pu_val_mean, label="PU Val (log10)")

            plt.xlabel("Epoch")
            plt.ylabel("log10(MSE Loss)")
            plt.title(f"MLP vs PU Loss Curves (log10) on {args.dataset}")
            plt.legend()
            plt.tight_layout()

            fig_path = os.path.join(
                args.save_dir,
                f"loss_compare_{args.dataset}_mlp_vs_pu_log10.png"
            )
            plt.savefig(fig_path, dpi=200)
            print(f"\n已保存 MLP vs PU log10 loss 对比图到: {fig_path}")
            plt.show()

    else:
        print("\nloss 曲线数据不完整，无法画 MLP vs PU 对比图。")


if __name__ == "__main__":
    main()
