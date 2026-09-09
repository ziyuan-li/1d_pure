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

from pure_resmlp.models.complex_pure import ComplexPUResNet1D
from pure_resmlp.models.complex_mlp import build_complex_mlp


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

    # Use the same split seed as the sample-efficiency trainer.
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    return X_test, y_test


@torch.no_grad()
def evaluate_model_complex(model, X_test, y_test, scaler_params, device):
    x_mean = scaler_params["x_scaler_mean"]
    x_scale = scaler_params["x_scaler_scale"]
    y_mean = scaler_params["y_scaler_mean"]
    y_scale = scaler_params["y_scaler_scale"]

    X_test_s = (X_test - x_mean) / x_scale
    y_test_s = (y_test - y_mean) / y_scale

    X_t = torch.tensor(X_test_s).float().to(device)
    y_t = torch.tensor(y_test_s).float().unsqueeze(1).to(device)

    pred = model(X_t.to(torch.complex64))
    loss = nn.MSELoss()(pred, y_t)
    return float(loss.item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="concrete", choices=["friedman", "concrete", "california"])
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--save_dir", type=str, default="./checkpoints_efficiency_complex_init0")
    parser.add_argument("--hidden_dim", type=int, default=64)

    parser.add_argument("--fractions", type=str, default="0.05,0.1,0.2,0.4,0.6,0.8",
                        help="逗号分隔，比如 0.05,0.1,0.2,...")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fractions = [float(x) for x in args.fractions.split(",")]

    print(f"Loading fixed test set for {args.dataset} ...")
    X_test, y_test = load_raw_test_data(args.dataset)

    results = {
        "complex_mlp": {"means": [], "stds": []},
        "complex_pu": {"means": [], "stds": []},
    }
    p_values_dict = {}
    summary_rows = []

    for frac in fractions:
        print(f"\n{'='*10} Fraction: {frac*100:.0f}% {'='*10}")
        buf = {"complex_mlp": [], "complex_pu": []}

        for run in range(args.runs):
            for m_key, m_tag in [
                ("complex_mlp", f"complex_mlp_h{args.hidden_dim}"),
                ("complex_pu", f"complex_pu_h{args.hidden_dim}")
            ]:
                ckpt_path = os.path.join(args.save_dir, f"{m_tag}_{args.dataset}_frac{frac}_run{run}_best.pth")
                if not os.path.exists(ckpt_path):
                    continue

                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

                if m_key == "complex_mlp":
                    model = build_complex_mlp(input_dim=X_test.shape[1], output_dim=1, hidden_dim=ckpt["hidden_dim"])
                else:
                    model = ComplexPUResNet1D(input_dim=X_test.shape[1], hidden_dim=ckpt["hidden_dim"], output_dim=1)

                model.load_state_dict(ckpt["model_state_dict"])
                model.to(device).eval()

                scaler_params = {
                    "x_scaler_mean": ckpt["x_scaler_mean"],
                    "x_scaler_scale": ckpt["x_scaler_scale"],
                    "y_scaler_mean": ckpt["y_scaler_mean"],
                    "y_scaler_scale": ckpt["y_scaler_scale"],
                }
                mse = evaluate_model_complex(model, X_test, y_test, scaler_params, device)
                buf[m_key].append(mse)

        mlp_raw = np.array(buf["complex_mlp"])
        pu_raw = np.array(buf["complex_pu"])

        m_mlp, s_mlp = (np.mean(mlp_raw), np.std(mlp_raw)) if len(mlp_raw) > 0 else (np.nan, np.nan)
        m_pu, s_pu = (np.mean(pu_raw), np.std(pu_raw)) if len(pu_raw) > 0 else (np.nan, np.nan)

        results["complex_mlp"]["means"].append(m_mlp)
        results["complex_mlp"]["stds"].append(s_mlp)
        results["complex_pu"]["means"].append(m_pu)
        results["complex_pu"]["stds"].append(s_pu)

        diff = (m_mlp - m_pu) if (not np.isnan(m_mlp) and not np.isnan(m_pu)) else np.nan
        p_val = np.nan
        sig = "N/A"

        # Pair errors by training run when both models have all runs.
        if len(mlp_raw) == args.runs and len(pu_raw) == args.runs:
            _, p_val = stats.ttest_rel(mlp_raw, pu_raw)
            p_values_dict[frac] = p_val
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"

        summary_rows.append({
            "Fraction": f"{frac*100:.0f}%",
            "Complex MLP Mean": f"{m_mlp:.6f}",
            "Complex MLP Std": f"{s_mlp:.6f}",
            "Complex PU Mean": f"{m_pu:.6f}",
            "Complex PU Std": f"{s_pu:.6f}",
            "Diff(MLP-PU)": f"{diff:.6f}",
            "P-Value": f"{p_val:.2e}" if not np.isnan(p_val) else "N/A",
            "Sig.": sig
        })

    print("\n" + "=" * 30 + " FINAL STATISTICAL SUMMARY (COMPLEX) " + "=" * 30)
    df = pd.DataFrame(summary_rows)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1200)
    print(df.to_string(index=False))
    print("=" * 95)

    plt.figure(figsize=(10, 6))
    x_axis = np.array(fractions) * 100

    mlp_m = np.array(results["complex_mlp"]["means"])
    mlp_s = np.array(results["complex_mlp"]["stds"])
    pu_m = np.array(results["complex_pu"]["means"])
    pu_s = np.array(results["complex_pu"]["stds"])

    plt.plot(x_axis, mlp_m, "o--", label="Complex MLP", lw=2)
    plt.fill_between(x_axis, mlp_m - mlp_s, mlp_m + mlp_s, alpha=0.15)

    plt.plot(x_axis, pu_m, "s-", label="Complex PU-ResNet (Ours)", lw=2)
    plt.fill_between(x_axis, pu_m - pu_s, pu_m + pu_s, alpha=0.15)

    # Asterisks mark p < 0.05.
    has_star = False
    for i, frac in enumerate(fractions):
        if frac in p_values_dict and p_values_dict[frac] < 0.05 and not np.isnan(pu_m[i]):
            has_star = True
            y_pos = pu_m[i] - pu_s[i] - (np.nanmax(mlp_m) * 0.04 if np.nanmax(mlp_m) > 0 else 0.02)
            plt.text(x_axis[i], y_pos, "*", fontsize=20, ha="center", fontweight="bold")

    handles, labels = plt.gca().get_legend_handles_labels()
    if has_star:
        star_proxy = Line2D([0], [0], marker="*", color="none", markerfacecolor="black",
                            markersize=12, markeredgecolor="black",
                            label="p < 0.05 (Paired T-test)")
        handles.append(star_proxy)

    plt.legend(handles=handles, fontsize=10, loc="upper right")
    plt.xlabel("Training Data Percentage (%)", fontsize=12, fontweight="bold")
    plt.ylabel("Test MSE (Standardized)", fontsize=12, fontweight="bold")
    plt.title(f"Sample Efficiency (Complex): {args.dataset.upper()}", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.6)

    save_fig_path = os.path.join(args.save_dir, f"sample_efficiency_complex_{args.dataset}_h{args.hidden_dim}.png")
    plt.savefig(save_fig_path, dpi=300, bbox_inches="tight")
    print(f"\n[Success] Plot saved to: {save_fig_path}")
    plt.show()


if __name__ == "__main__":
    main()
