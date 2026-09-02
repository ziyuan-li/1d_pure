# 07_test_sample_efficiency.py
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.datasets import make_friedman1, fetch_california_housing
from sklearn.model_selection import train_test_split
from matplotlib.lines import Line2D

import torch
import torch.nn as nn

# 假设 mlp_models 和 pu_models 在当前路径
from pure_resmlp.models.mlp import build_4layer_mlp
from pure_resmlp.models.pure import RealPUResNet1D


def load_raw_test_data(dataset_name):
    if dataset_name == "friedman":
        X, y = make_friedman1(n_samples=10000, n_features=10, noise=0.0, random_state=0)
    elif dataset_name == "concrete":
        df = pd.read_excel("./data/concrete+compressive+strength/Concrete_Data.xls")
        X = df.iloc[:, :8].values
        y = df.iloc[:, 8].values
    else:
        data = fetch_california_housing()
        X, y = data.data, data.target

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.20, random_state=0
    )
    return X_test, y_test


def evaluate_model(model, X_test, y_test, scaler_params, device):
    x_mean = scaler_params["x_scaler_mean"]
    x_scale = scaler_params["x_scaler_scale"]
    y_mean = scaler_params["y_scaler_mean"]
    y_scale = scaler_params["y_scaler_scale"]

    X_test_s = (X_test - x_mean) / x_scale
    y_test_s = (y_test - y_mean) / y_scale

    X_t = torch.tensor(X_test_s).float().to(device)
    y_t = torch.tensor(y_test_s).float().unsqueeze(1).to(device)

    model.eval()
    with torch.no_grad():
        pred = model(X_t)
        loss = nn.MSELoss()(pred, y_t)
    return loss.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="concrete")
    parser.add_argument("--runs", type=int, default=5, help="需要读取的 Run 数量")
    parser.add_argument("--save_dir", type=str, default="./checkpoints_efficiency_test_init0")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fractions = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8]

    print(f"Loading Test Data for {args.dataset}...")
    X_test, y_test = load_raw_test_data(args.dataset)

    results = {"mlp": {"means": [], "stds": []}, "pu": {"means": [], "stds": []}}
    p_values_dict = {}
    summary_data = []

    for frac in fractions:
        print(f"\n{'=' * 10} Processing Fraction: {frac * 100:.0f}% {'=' * 10}")
        loss_buffer = {"mlp": [], "pu": []}

        for run in range(args.runs):
            for m_key, m_tag in [("mlp", "mlp4_h128"), ("pu", "puresnet_h128")]:
                ckpt_path = os.path.join(args.save_dir, f"{m_tag}_{args.dataset}_frac{frac}_run{run}_best.pth")
                if not os.path.exists(ckpt_path):
                    continue

                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                if m_key == "mlp":
                    model = build_4layer_mlp(input_dim=X_test.shape[1], output_dim=1, hidden_dim=ckpt["hidden_dim"])
                else:
                    model = RealPUResNet1D(input_dim=X_test.shape[1], hidden_dim=ckpt["hidden_dim"], output_dim=1)

                model.load_state_dict(ckpt["model_state_dict"])
                model.to(device)

                scaler_params = {
                    "x_scaler_mean": ckpt["x_scaler_mean"], "x_scaler_scale": ckpt["x_scaler_scale"],
                    "y_scaler_mean": ckpt["y_scaler_mean"], "y_scaler_scale": ckpt["y_scaler_scale"],
                }
                mse = evaluate_model(model, X_test, y_test, scaler_params, device)
                loss_buffer[m_key].append(mse)

        # 统计计算
        mlp_raw = np.array(loss_buffer["mlp"])
        pu_raw = np.array(loss_buffer["pu"])

        m_mlp, s_mlp = (np.mean(mlp_raw), np.std(mlp_raw)) if len(mlp_raw) > 0 else (np.nan, np.nan)
        m_pu, s_pu = (np.mean(pu_raw), np.std(pu_raw)) if len(pu_raw) > 0 else (np.nan, np.nan)

        results["mlp"]["means"].append(m_mlp)
        results["mlp"]["stds"].append(s_mlp)
        results["pu"]["means"].append(m_pu)
        results["pu"]["stds"].append(s_pu)

        # [修改] 计算 Difference 和 P-Value
        diff = m_mlp - m_pu if not np.isnan(m_mlp) and not np.isnan(m_pu) else np.nan
        p_val_val = np.nan
        sig_str = "N/A"

        if len(mlp_raw) == args.runs and len(pu_raw) == args.runs:
            _, p_val_val = stats.ttest_rel(mlp_raw, pu_raw)
            p_values_dict[frac] = p_val_val
            sig_str = "***" if p_val_val < 0.001 else "**" if p_val_val < 0.01 else "*" if p_val_val < 0.05 else "ns"

        summary_data.append({
            "Fraction": f"{frac * 100:.0f}%",
            "MLP Mean": f"{m_mlp:.6f}",
            "MLP Std": f"{s_mlp:.6f}",
            "PU Mean": f"{m_pu:.6f}",
            "PU Std": f"{s_pu:.6f}",
            "Diff(M-P)": f"{diff:.6f}",
            "P-Value": f"{p_val_val:.2e}" if not np.isnan(p_val_val) else "N/A",
            "Sig.": sig_str
        })

    # --- 打印增强版汇总表格 ---
    print("\n" + "=" * 30 + " FINAL STATISTICAL SUMMARY " + "=" * 30)
    df_summary = pd.DataFrame(summary_data)
    # 设置显示选项确保不被截断
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df_summary.to_string(index=False))
    print("=" * 87)

    # === 绘图逻辑 ===
    plt.figure(figsize=(10, 6))
    x_axis = np.array(fractions) * 100
    mlp_m, mlp_s = np.array(results["mlp"]["means"]), np.array(results["mlp"]["stds"])
    pu_m, pu_s = np.array(results["pu"]["means"]), np.array(results["pu"]["stds"])

    plt.plot(x_axis, mlp_m, 'o--', label="Standard MLP", color="tab:blue", lw=2)
    plt.fill_between(x_axis, mlp_m - mlp_s, mlp_m + mlp_s, color="tab:blue", alpha=0.15)

    plt.plot(x_axis, pu_m, 's-', label="PU-ResNet (Ours)", color="tab:red", lw=2)
    plt.fill_between(x_axis, pu_m - pu_s, pu_m + pu_s, color="tab:red", alpha=0.15)

    # 标注星号
    has_star = False
    for i, frac in enumerate(fractions):
        if frac in p_values_dict and p_values_dict[frac] < 0.05:
            if not np.isnan(pu_m[i]):
                has_star = True
                y_pos = pu_m[i] - pu_s[i] - (np.nanmax(mlp_m) * 0.04)
                plt.text(x_axis[i], y_pos, '*', fontsize=20, ha='center', color='black', fontweight='bold')

    handles, labels = plt.gca().get_legend_handles_labels()
    if has_star:
        star_proxy = Line2D([0], [0], marker='*', color='none', markerfacecolor='black',
                            markersize=12, markeredgecolor='black', label='p < 0.05 (Paired T-test)')
        handles.append(star_proxy)

    plt.legend(handles=handles, fontsize=10, loc='upper right')
    plt.xlabel("Training Data Percentage (%)", fontsize=12, fontweight='bold')
    plt.ylabel("Test MSE (Standardized)", fontsize=12, fontweight='bold')
    plt.title(f"Sample Efficiency Analysis: {args.dataset.upper()}", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)

    save_fig_path = os.path.join(args.save_dir, f"sample_efficiency_{args.dataset}_final.png")
    plt.savefig(save_fig_path, dpi=300, bbox_inches='tight')
    print(f"\n[Success] Plot saved to: {save_fig_path}")
    plt.show()


if __name__ == "__main__":
    main()
