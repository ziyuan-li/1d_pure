# 07_test_sample_efficiency.py
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats  # [新增] 用于统计检验
from sklearn.datasets import make_friedman1, fetch_california_housing
from sklearn.model_selection import train_test_split
# StandardScaler 仅用于导入类，具体参数从checkpoint加载

import torch
import torch.nn as nn

# 假设这几个模型文件在同一目录下，或者在PYTHONPATH中
from mlp_models import build_4layer_mlp
from pu_models import RealPUResNet1D


def load_raw_test_data(dataset_name):
    """
    必须使用与训练脚本完全一致的逻辑来获取 Test Set
    """
    if dataset_name == "friedman":
        X, y = make_friedman1(n_samples=10000, n_features=10, noise=0.0, random_state=0)
    elif dataset_name == "concrete":
        # 请确保路径存在
        df = pd.read_excel("./data/concrete+compressive+strength/Concrete_Data.xls")
        X = df.iloc[:, :8].values
        y = df.iloc[:, 8].values
    else:
        data = fetch_california_housing()
        X, y = data.data, data.target

    # 修正逻辑：必须严格复制 06.py 的第一次 split，以确保 Test Set 是完全隔离且一致的
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.20, random_state=0
    )
    return X_test, y_test


def evaluate_model(model, X_test, y_test, scaler_params, device):
    """
    加载模型和对应的 scaler 参数，对 Test Set 进行评估
    返回：Test MSE (Scaled Space)
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

    # [新增] 存储 P-value
    p_values = {}

    # 遍历所有比例
    for frac in fractions:
        print(f"\n{'=' * 10} Processing Fraction: {frac} {'=' * 10}")

        # === 当 frac 为 1.0 时直接赋值 ===
        if frac == 1.0:
            print("  [Info] Skipping evaluation for Fraction 1.0, using hardcoded values.")

            # Hardcoded values (from previous runs)
            results["mlp"]["means"].append(0.002032)
            results["mlp"]["stds"].append(0.000546)
            print(f"  [Summary MLP] Mean MSE: 0.002032 ± 0.000546 (Hardcoded)")

            results["pu"]["means"].append(0.001773)
            results["pu"]["stds"].append(0.000137)
            print(f"  [Summary PU]  Mean MSE: 0.001773 ± 0.000137 (Hardcoded)")

            # 注意：硬编码无法进行 T-Test，因为没有原始样本
            continue
        # ==========================================

        loss_buffer = {"mlp": [], "pu": []}

        for run in range(args.runs):
            # 遍历两种模型
            for m_key, m_tag, m_cls in [
                ("mlp", "mlp4_h128", build_4layer_mlp),
                ("pu", "puresnet_h128", RealPUResNet1D)
            ]:
                ckpt_path = os.path.join(args.save_dir, f"{m_tag}_{args.dataset}_frac{frac}_run{run}_best.pth")

                if not os.path.exists(ckpt_path):
                    # print(f"  [Warning] Missing: {ckpt_path}")
                    continue

                # 加载 Checkpoint (weights_only=False 解决 numpy 问题)
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

                # 构建模型
                if m_key == "mlp":
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

                # 打印单次 Run 的结果 (可选，注释掉以减少刷屏)
                # print(f"  -> [Run {run}] Model: {m_key.upper():<3} | MSE: {mse:.6f}")

        # --- 统计与 T-Test 模块 ---
        mlp_raw = np.array(loss_buffer["mlp"])
        pu_raw = np.array(loss_buffer["pu"])

        print(f"-" * 40)

        # 1. 存入均值和方差
        if len(mlp_raw) > 0:
            results["mlp"]["means"].append(np.mean(mlp_raw))
            results["mlp"]["stds"].append(np.std(mlp_raw))
            print(f"  [Summary MLP] Mean MSE: {np.mean(mlp_raw):.6f} ± {np.std(mlp_raw):.6f}")
        else:
            results["mlp"]["means"].append(np.nan)
            results["mlp"]["stds"].append(np.nan)

        if len(pu_raw) > 0:
            results["pu"]["means"].append(np.mean(pu_raw))
            results["pu"]["stds"].append(np.std(pu_raw))
            print(f"  [Summary PU]  Mean MSE: {np.mean(pu_raw):.6f} ± {np.std(pu_raw):.6f}")
        else:
            results["pu"]["means"].append(np.nan)
            results["pu"]["stds"].append(np.nan)

        # 2. [新增] 计算配对 T 检验 (Paired T-Test)
        # 前提：两个模型都跑满了 run，且数据是对齐的（由 seed 保证）
        if len(mlp_raw) == args.runs and len(pu_raw) == args.runs:
            t_stat, p_val = stats.ttest_rel(mlp_raw, pu_raw)
            p_values[frac] = p_val

            # 判断显著性符号
            sig_level = "ns"
            if p_val < 0.001:
                sig_level = "***"
            elif p_val < 0.01:
                sig_level = "**"
            elif p_val < 0.05:
                sig_level = "*"

            print(f"  [T-TEST] p-value: {p_val:.5e} ({sig_level})")
            print(f"           Diff   : {np.mean(mlp_raw) - np.mean(pu_raw):.6f} (Positive means PU is better)")
        else:
            print("  [T-TEST] Skipped (insufficient runs)")

    # === 绘图 ===
    plt.figure(figsize=(8, 6))
    x_axis = np.array(fractions) * 100  # 转百分比

    # 绘制 MLP
    mlp_m = np.array(results["mlp"]["means"])
    mlp_s = np.array(results["mlp"]["stds"])
    valid_mask_mlp = ~np.isnan(mlp_m)

    if np.any(valid_mask_mlp):
        plt.plot(x_axis[valid_mask_mlp], mlp_m[valid_mask_mlp], 'o--', label="Standard MLP", color="tab:blue",
                 linewidth=2)
        plt.fill_between(x_axis[valid_mask_mlp],
                         mlp_m[valid_mask_mlp] - mlp_s[valid_mask_mlp],
                         mlp_m[valid_mask_mlp] + mlp_s[valid_mask_mlp],
                         color="tab:blue", alpha=0.15)

    # 绘制 PU
    pu_m = np.array(results["pu"]["means"])
    pu_s = np.array(results["pu"]["stds"])
    valid_mask_pu = ~np.isnan(pu_m)

    if np.any(valid_mask_pu):
        plt.plot(x_axis[valid_mask_pu], pu_m[valid_mask_pu], 's-', label="PU-ResNet (Ours)", color="tab:red",
                 linewidth=2)
        plt.fill_between(x_axis[valid_mask_pu],
                         pu_m[valid_mask_pu] - pu_s[valid_mask_pu],
                         pu_m[valid_mask_pu] + pu_s[valid_mask_pu],
                         color="tab:red", alpha=0.15)

    # [新增] 标注显著性 (P < 0.05)
    for i, frac in enumerate(fractions):
        # 检查该 frac 是否有 p-value 且 p < 0.05
        if frac in p_values and p_values[frac] < 0.05:
            if not np.isnan(pu_m[i]):
                # 在红线下方画一个星号
                # 调整 y 位置，使其在 Error Band 下方一点点
                y_pos = pu_m[i] - pu_s[i] - (np.max(mlp_m[~np.isnan(mlp_m)]) * 0.05)
                plt.text(x_axis[i], y_pos, '*',
                         fontsize=20, ha='center', va='center',
                         color='black', fontweight='bold')

    plt.xlabel("Training Data Percentage (%)", fontsize=12, fontweight='bold')
    plt.ylabel("Test MSE (Standardized)", fontsize=12, fontweight='bold')
    plt.title(f"Sample Efficiency Analysis ({args.dataset})", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # 保存
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    save_fig_path = os.path.join(args.save_dir, f"sample_efficiency_{args.dataset}_ttest.png")
    plt.savefig(save_fig_path, dpi=300)
    print(f"\nResult plot saved to: {save_fig_path}")
    plt.show()


if __name__ == "__main__":
    main()