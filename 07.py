# 07_test_sample_efficiency.py
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_friedman1, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # 仅用于导入类，具体参数从checkpoint加载

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from mlp_models import build_4layer_mlp
from pu_models import RealPUResNet1D


def load_raw_test_data(dataset_name):
    """
    必须使用与训练脚本完全一致的逻辑来获取 Test Set
    """
    if dataset_name == "friedman":
        X, y = make_friedman1(n_samples=10000, n_features=10, noise=0.0, random_state=0)
    elif dataset_name == "concrete":
        df = pd.read_excel("./data/concrete+compressive+strength/Concrete_Data.xls")
        X = df.iloc[:, :8].values
        y = df.iloc[:, 8].values
    else:
        data = fetch_california_housing()
        X, y = data.data, data.target

    # 只需提取 Test Set
    _, _, _, y_temp = train_test_split(X, y, test_size=0.20, random_state=0)
    # 这里的 X_temp, y_temp 实际上包含了 Val+Test，需要再切一次，这必须跟 06.py 里的逻辑完全对齐
    # 06.py: X_train_full, X_test... = split(..., test_size=0.20)
    # 所以上面的 X_temp 其实已经是 Test Set 了？不对，让我们仔细看 06.py 的第一步 split

    # 修正逻辑：必须严格复制 06.py 的第一次 split
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.20, random_state=0
    )
    return X_test, y_test


def evaluate_model(model, X_test, y_test, scaler_params, device):
    """
    加载模型和对应的 scaler 参数，对 Test Set 进行评估
    返回：Test MSE (Scaled Space) 或者 Original Space 均可，这里为了稳定通常看 Scaled MSE
    """
    # 1. 恢复 Scaler
    # 手动应用标准化 (x - mean) / scale
    x_mean = scaler_params["x_scaler_mean"]
    x_scale = scaler_params["x_scaler_scale"]
    y_mean = scaler_params["y_scaler_mean"]
    y_scale = scaler_params["y_scaler_scale"]

    X_test_s = (X_test - x_mean) / x_scale
    y_test_s = (y_test - y_mean) / y_scale

    # 转 Tensor
    X_t = torch.tensor(X_test_s).float().to(device)
    y_t = torch.tensor(y_test_s).float().unsqueeze(1).to(device)

    model.eval()
    with torch.no_grad():
        pred = model(X_t)
        loss = nn.MSELoss()(pred, y_t)

    return loss.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="friedman")
    parser.add_argument("--runs", type=int, default=5, help="需要读取的 Run 数量")
    parser.add_argument("--save_dir", type=str, default="./checkpoints_efficiency")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fractions = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

    print(f"Loading Test Data for {args.dataset}...")
    X_test, y_test = load_raw_test_data(args.dataset)

    # 结果容器
    results = {
        "mlp": {"means": [], "stds": []},
        "pu": {"means": [], "stds": []}
    }

    # 遍历所有比例
    for frac in fractions:
        print(f"\nProcessing Fraction: {frac}")

        loss_buffer = {"mlp": [], "pu": []}

        for run in range(args.runs):
            # 遍历两种模型
            for m_key, m_tag, m_cls in [
                ("mlp", "mlp4_h128", build_4layer_mlp),
                ("pu", "puresnet_h128", RealPUResNet1D)
            ]:
                ckpt_path = os.path.join(args.save_dir, f"{m_tag}_{args.dataset}_frac{frac}_run{run}_best.pth")

                if not os.path.exists(ckpt_path):
                    print(f"  [Warning] Missing: {ckpt_path}")
                    continue

                # 加载 Checkpoint
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

                # 构建模型
                model = m_cls(input_dim=X_test.shape[1], hidden_dim=ckpt["hidden_dim"], output_dim=1)
                if m_key == "mlp":  # build_4layer_mlp参数略有不同，需要适配
                    model = build_4layer_mlp(input_dim=X_test.shape[1], output_dim=1, hidden_dim=ckpt["hidden_dim"])
                else:
                    model = RealPUResNet1D(input_dim=X_test.shape[1], hidden_dim=ckpt["hidden_dim"], output_dim=1)

                model.load_state_dict(ckpt["model_state_dict"])
                model.to(device)

                # 提取 Scaler 参数
                scaler_params = {
                    "x_scaler_mean": ckpt["x_scaler_mean"],
                    "x_scaler_scale": ckpt["x_scaler_scale"],
                    "y_scaler_mean": ckpt["y_scaler_mean"],
                    "y_scaler_scale": ckpt["y_scaler_scale"],
                }

                # 评估
                mse = evaluate_model(model, X_test, y_test, scaler_params, device)
                loss_buffer[m_key].append(mse)

        # 统计当前 Fraction 的均值和方差
        for m_key in ["mlp", "pu"]:
            data = np.array(loss_buffer[m_key])
            if len(data) > 0:
                mean_val = np.mean(data)
                std_val = np.std(data)
                results[m_key]["means"].append(mean_val)
                results[m_key]["stds"].append(std_val)
                print(f"  [{m_key.upper()}] Mean MSE: {mean_val:.4f} ± {std_val:.4f}")
            else:
                results[m_key]["means"].append(np.nan)
                results[m_key]["stds"].append(np.nan)

    # === 绘图 ===
    plt.figure(figsize=(8, 6))
    x_axis = np.array(fractions) * 100  # 转百分比

    # 绘制 MLP
    mlp_m = np.array(results["mlp"]["means"])
    mlp_s = np.array(results["mlp"]["stds"])
    plt.plot(x_axis, mlp_m, 'o--', label="Standard MLP", color="tab:blue", linewidth=2)
    plt.fill_between(x_axis, mlp_m - mlp_s, mlp_m + mlp_s, color="tab:blue", alpha=0.15)

    # 绘制 PU
    pu_m = np.array(results["pu"]["means"])
    pu_s = np.array(results["pu"]["stds"])
    plt.plot(x_axis, pu_m, 's-', label="PU-ResNet (Ours)", color="tab:red", linewidth=2)
    plt.fill_between(x_axis, pu_m - pu_s, pu_m + pu_s, color="tab:red", alpha=0.15)

    plt.xlabel("Training Data Percentage (%)", fontsize=12, fontweight='bold')
    plt.ylabel("Test MSE (Standardized)", fontsize=12, fontweight='bold')
    plt.title(f"Sample Efficiency Analysis ({args.dataset})", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # 保存
    save_fig_path = os.path.join(args.save_dir, f"sample_efficiency_{args.dataset}.png")
    plt.savefig(save_fig_path, dpi=300)
    print(f"\nResult plot saved to: {save_fig_path}")
    plt.show()


if __name__ == "__main__":
    main()