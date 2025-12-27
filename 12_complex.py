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

# 导入实数模型（保留以备查）
from mlp_models import build_4layer_mlp
from pu_models import RealPUResNet1D

# 导入复数模型
from complex_pu_model import ComplexPUResNet1D
from complex_mlp_models import build_complex_mlp


# --- 辅助工具函数 ---
def count_parameters(model: nn.Module) -> int:
    """ 计算模型的可训练参数总量 """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --- 数据加载函数 (保持不变) ---
def load_friedman():
    X, y = make_friedman1(n_samples=10000, n_features=10, noise=0.0, random_state=0)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=0)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=0)
    return X_train, X_val, X_test, y_train, y_val, y_test


def load_concrete():
    df = pd.read_excel("./data/concrete+compressive+strength/Concrete_Data.xls")
    X, y = df.iloc[:, :8].values, df.iloc[:, 8].values
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.40, random_state=0)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=0)
    return X_train, X_val, X_test, y_train, y_val, y_test


def load_california():
    data = fetch_california_housing()
    X, y = data.data, data.target
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.20, random_state=0)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=0)
    return X_train, X_val, X_test, y_train, y_val, y_test


def standardize_splits(X_train, X_val, X_test, y_train, y_val, y_test):
    x_scaler, y_scaler = StandardScaler(), StandardScaler()
    X_train_s = x_scaler.fit_transform(X_train)
    X_val_s = x_scaler.transform(X_val)
    X_test_s = x_scaler.transform(X_test)
    y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_s = y_scaler.transform(y_val.reshape(-1, 1)).ravel()
    y_test_s = y_scaler.transform(y_test.reshape(-1, 1)).ravel()
    return X_train_s, X_val_s, X_test_s, y_train_s, y_val_s, y_test_s


# --- 修改后的评估函数：适配复数模型 ---
def eval_on_test(model, loader, device):
    criterion = nn.MSELoss()
    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            # 如果是复数模型，输入需要转换
            if not X.is_complex():
                X = X.to(torch.complex64)

            # 复数模型输出后取实部进行 MSE 比较
            pred = model(X).real
            total_loss += criterion(pred, y).item() * X.size(0)
            n += X.size(0)
    return total_loss / n


def plot_loss_curves(loss_curves, dataset_name, save_dir):
    """ 为单个数据集绘制 Complex MLP vs Complex PU 的 Loss 对比图 """
    try:
        plt.figure(figsize=(8, 5))
        for m_type in ["complex_mlp", "complex_pu"]:
            train_data = np.array(loss_curves[m_type]["train"])
            val_data = np.array(loss_curves[m_type]["val"])
            if train_data.size == 0: continue

            min_len = min(train_data.shape[1], val_data.shape[1])
            epochs = np.arange(1, min_len + 1)

            ls = '-' if "mlp" in m_type else '--'
            plt.plot(epochs, np.log10(train_data[:, :min_len].mean(axis=0) + 1e-12),
                     label=f"{m_type.upper()} Train", linestyle=ls)
            plt.plot(epochs, np.log10(val_data[:, :min_len].mean(axis=0) + 1e-12),
                     label=f"{m_type.upper()} Val", linestyle=ls)

        plt.xlabel("Epoch")
        plt.ylabel("log10(MSE Loss)")
        plt.title(f"Complex Training History on {dataset_name.upper()}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"loss_complex_{dataset_name}.png"), dpi=200)
        plt.close()
    except Exception as e:
        print(f"  [!] 绘图失败 ({dataset_name}): {e}")


def main():
    parser = argparse.ArgumentParser()
    # 默认路径改为复数模型保存路径
    parser.add_argument("--save_dir", type=str, default="./checkpoints_complex_test")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--noise_stds", type=float, nargs="+", default=[0.0, 0.05])
    parser.add_argument("--hidden_dim", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = ["friedman", "concrete", "california"]
    noise_stds = sorted(set(args.noise_stds))
    if 0.0 not in noise_stds: noise_stds = [0.0] + noise_stds

    final_results = []

    for dname in datasets:
        print(f"\n>>> Loading dataset: {dname.upper()}...")
        try:
            if dname == "friedman":
                data_splits = load_friedman()
            elif dname == "concrete":
                data_splits = load_concrete()
            else:
                data_splits = load_california()
        except Exception as e:
            print(f"    无法加载 {dname}: {e}")
            continue

        X_train_s, X_val_s, X_test_s, y_train_s, y_val_s, y_test_s = standardize_splits(*data_splits)
        input_dim = X_train_s.shape[1]

        rng = np.random.RandomState(123)
        test_noises = {std: (np.zeros_like(X_test_s) if std == 0.0 else rng.normal(0, std, X_test_s.shape)) for std in
                       noise_stds}

        # 更新模型对比列表为复数版本
        current_loss_curves = {"complex_mlp": {"train": [], "val": []}, "complex_pu": {"train": [], "val": []}}

        for model_type, model_tag in [("complex_mlp", f"complex_mlp_h{args.hidden_dim}"),
                                      ("complex_pu", f"complex_pu_h{args.hidden_dim}")]:
            run_metrics = {std: [] for std in noise_stds}
            param_count = 0

            for run in range(args.runs):
                ckpt_path = os.path.join(args.save_dir, f"{model_tag}_{dname}_run{run}_best.pth")
                loss_path = os.path.join(args.save_dir, f"{model_tag}_{dname}_run{run}_losses.npz")

                if not os.path.exists(ckpt_path): continue

                # 加载复数模型权重
                ckpt = torch.load(ckpt_path, map_location="cpu")
                h_dim = ckpt.get("hidden_dim", args.hidden_dim)

                if model_type == "complex_mlp":
                    model = build_complex_mlp(input_dim, 1, h_dim)
                else:
                    model = ComplexPUResNet1D(input_dim, h_dim, 1)

                model.load_state_dict(ckpt["model_state_dict"])
                model.to(device)

                if param_count == 0: param_count = count_parameters(model)

                for std in noise_stds:
                    # 准备测试数据，保持 float 类型，由 eval_on_test 处理转换
                    X_noisy = torch.from_numpy(X_test_s + test_noises[std]).float()
                    y_true = torch.from_numpy(y_test_s).float().unsqueeze(-1)
                    loader = DataLoader(TensorDataset(X_noisy, y_true), batch_size=128)
                    run_metrics[std].append(eval_on_test(model, loader, device))

                if os.path.exists(loss_path):
                    ld = np.load(loss_path)
                    current_loss_curves[model_type]["train"].append(ld["train_losses"])
                    current_loss_curves[model_type]["val"].append(ld["val_losses"])

            for std in noise_stds:
                vals = np.array(run_metrics[std])
                if len(vals) > 0:
                    final_results.append({
                        "Dataset": dname, "Model": model_type.upper(), "Params": param_count,
                        "Noise": std, "MSE_Mean": vals.mean(), "MSE_Std": vals.std()
                    })

        plot_loss_curves(current_loss_curves, dname, args.save_dir)

    # --- 输出表格 ---
    print("\n" + "=" * 85)
    header = f"{'DATASET':<12} | {'MODEL':<12} | {'PARAMS':<10} | {'NOISE':<6} | {'MSE MEAN':<12} | {'MSE STD':<10}"
    print(header)
    print("-" * 85)

    df = pd.DataFrame(final_results)
    for _, row in df.iterrows():
        print(
            f"{row['Dataset']:<12} | {row['Model']:<12} | {row['Params']:<10,} | {row['Noise']:<6.2f} | {row['MSE_Mean']:<12.6f} | {row['MSE_Std']:<10.6f}")
    print("=" * 85)
    print(f"Images path: {args.save_dir}")


if __name__ == "__main__":
    main()