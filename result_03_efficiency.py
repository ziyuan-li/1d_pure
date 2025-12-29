#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sample_efficiency_all.py

Merged from:
- 07_test_sample_efficiency.py (real-valued)
- 17_complex_test_sample_efficiency.py (complex-valued)

Purpose:
- Unified sample efficiency analysis (Real + Complex)
- Output publication-ready plot + tables (CSV + LaTeX)

Models:
- RV-MLP, RV-PU
- CV-MLP, CV-PU

Outputs (to --out_dir):
- sample_efficiency_all_<dataset>.png
- sample_efficiency_raw_<dataset>.csv          (per-run raw)
- sample_efficiency_table_<dataset>.csv        (mean/std summary)
- sample_efficiency_table_<dataset>.tex        (LaTeX table, booktabs, bold best per fraction)
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from sklearn.datasets import make_friedman1, fetch_california_housing
from sklearn.model_selection import train_test_split

# ===== real =====
from mlp_models import build_4layer_mlp
from pu_models import RealPUResNet1D

# ===== complex =====
from complex_mlp_models import build_complex_mlp
from complex_pu_model import ComplexPUResNet1D


# ============================================================
# Data
# ============================================================
def load_raw_test_data(dataset: str):
    if dataset == "friedman":
        X, y = make_friedman1(n_samples=10000, n_features=10, noise=0.0, random_state=0)
    elif dataset == "concrete":
        df = pd.read_excel("./data/concrete+compressive+strength/Concrete_Data.xls")
        X = df.iloc[:, :8].values
        y = df.iloc[:, 8].values
    elif dataset == "california":
        data = fetch_california_housing()
        X, y = data.data, data.target
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    return X_test, y_test


# ============================================================
# Evaluation
# ============================================================
def eval_real(model, X_test, y_test, sp, device) -> float:
    Xs = (X_test - sp["x_mean"]) / sp["x_scale"]
    ys = (y_test - sp["y_mean"]) / sp["y_scale"]
    X_t = torch.tensor(Xs).float().to(device)
    y_t = torch.tensor(ys).float().unsqueeze(1).to(device)
    with torch.no_grad():
        pred = model(X_t)
        return float(nn.MSELoss()(pred, y_t).item())


def eval_complex(model, X_test, y_test, sp, device) -> float:
    Xs = (X_test - sp["x_mean"]) / sp["x_scale"]
    ys = (y_test - sp["y_mean"]) / sp["y_scale"]
    X_t = torch.tensor(Xs).float().to(device)
    y_t = torch.tensor(ys).float().unsqueeze(1).to(device)
    with torch.no_grad():
        pred = model(X_t.to(torch.complex64)).real
        return float(nn.MSELoss()(pred, y_t).item())


# ============================================================
# LaTeX table (booktabs + bold best per fraction)
# ============================================================
def write_latex_table(df_sum: pd.DataFrame, out_path: str) -> None:
    """
    df_sum columns:
      Dataset, Fraction, Model, MSE_Mean, MSE_Std, RunsUsed
    """
    if df_sum.empty:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("% Empty table.\n")
        return

    d = df_sum.copy()

    # format cell
    def _fmt(r):
        std = r["MSE_Std"]
        if pd.isna(std):
            std = 0.0
        return f"{r['MSE_Mean']:.4f} $\\pm$ {std:.4f}"

    d["Cell"] = d.apply(_fmt, axis=1)

    # bold best (min mean) per fraction
    best_idx = d.groupby(["Dataset", "Fraction"])["MSE_Mean"].idxmin()
    d.loc[best_idx, "Cell"] = d.loc[best_idx, "Cell"].apply(lambda s: f"\\textbf{{{s}}}")

    pivot = d.pivot_table(index=["Dataset", "Fraction"], columns="Model", values="Cell", aggfunc="first")

    # stable column order
    col_order = [c for c in ["RV-MLP", "RV-PURe", "CV-MLP", "CV-PURe"] if c in pivot.columns]
    pivot = pivot[col_order]

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\setlength{\\tabcolsep}{6pt}")
    lines.append("\\renewcommand{\\arraystretch}{1.15}")
    lines.append("\\caption{Sample efficiency results (test MSE; mean $\\pm$ std over runs). "
                 "Best per fraction is in bold.}")
    lines.append("\\begin{tabular}{ll" + "c" * len(col_order) + "}")
    lines.append("\\toprule")
    header = ["Dataset", "Frac."] + col_order
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")

    for (ds, frac), row in pivot.iterrows():
        row_cells = [ds, f"{float(frac):.2f}"] + [row.get(m, "--") for m in col_order]
        lines.append(" & ".join(row_cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="concrete", choices=["friedman", "concrete", "california"])
    parser.add_argument("--runs", type=int, default=5)

    parser.add_argument("--real_dir", default="./checkpoints_efficiency_test_init0")
    parser.add_argument("--complex_dir", default="./checkpoints_efficiency_complex_init0")
    parser.add_argument("--hidden_dim", type=int, default=64)

    parser.add_argument("--fractions", default="0.05,0.1,0.2,0.4,0.6,0.8")
    parser.add_argument("--out_dir", default="./pub_efficiency")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fractions = [float(x) for x in args.fractions.split(",")]
    X_test, y_test = load_raw_test_data(args.dataset)

    # for plotting (mean/std per fraction)
    results = {
        "RV-MLP": [], "RV-PURe": [],
        "CV-MLP": [], "CV-PURe": []
    }

    # for tables (raw per run)
    rows = []

    for frac in fractions:
        buf = {k: [] for k in results}

        for run in range(args.runs):
            # ---------- real ----------
            for key, tag, builder in [
                ("RV-MLP", "mlp4_h128", lambda d: build_4layer_mlp(d, 1, 128)),
                ("RV-PURe", "puresnet_h128", lambda d: RealPUResNet1D(d, 128, 1)),
            ]:
                ckpt = os.path.join(
                    args.real_dir,
                    f"{tag}_{args.dataset}_frac{frac}_run{run}_best.pth"
                )
                if not os.path.exists(ckpt):
                    continue

                # PyTorch 2.6+ compatibility
                c = torch.load(ckpt, map_location="cpu", weights_only=False)

                m = builder(X_test.shape[1])
                m.load_state_dict(c["model_state_dict"])
                m.to(device).eval()

                sp = {
                    "x_mean": c["x_scaler_mean"], "x_scale": c["x_scaler_scale"],
                    "y_mean": c["y_scaler_mean"], "y_scale": c["y_scaler_scale"],
                }

                val = eval_real(m, X_test, y_test, sp, device)
                buf[key].append(val)
                rows.append({
                    "Dataset": args.dataset,
                    "Fraction": float(frac),
                    "Run": int(run),
                    "Model": key,
                    "TestMSE": float(val),
                })

            # ---------- complex ----------
            for key, tag, builder in [
                ("CV-MLP", f"complex_mlp_h{args.hidden_dim}",
                 lambda d: build_complex_mlp(d, 1, args.hidden_dim)),
                ("CV-PURe", f"complex_pu_h{args.hidden_dim}",
                 lambda d: ComplexPUResNet1D(d, args.hidden_dim, 1)),
            ]:
                ckpt = os.path.join(
                    args.complex_dir,
                    f"{tag}_{args.dataset}_frac{frac}_run{run}_best.pth"
                )
                if not os.path.exists(ckpt):
                    continue

                # PyTorch 2.6+ compatibility
                c = torch.load(ckpt, map_location="cpu", weights_only=False)

                m = builder(X_test.shape[1])
                m.load_state_dict(c["model_state_dict"])
                m.to(device).eval()

                sp = {
                    "x_mean": c["x_scaler_mean"], "x_scale": c["x_scaler_scale"],
                    "y_mean": c["y_scaler_mean"], "y_scale": c["y_scaler_scale"],
                }

                val = eval_complex(m, X_test, y_test, sp, device)
                buf[key].append(val)
                rows.append({
                    "Dataset": args.dataset,
                    "Fraction": float(frac),
                    "Run": int(run),
                    "Model": key,
                    "TestMSE": float(val),
                })

        # aggregate for plotting
        for k in results:
            arr = np.array(buf[k], dtype=float)
            if arr.size == 0:
                results[k].append((np.nan, np.nan))
            else:
                results[k].append((float(np.mean(arr)), float(np.std(arr, ddof=0))))

    # ========================================================
    # Tables: raw + summary
    # ========================================================
    df_raw = pd.DataFrame(rows)
    raw_path = os.path.join(args.out_dir, f"sample_efficiency_raw_{args.dataset}.csv")
    df_raw.to_csv(raw_path, index=False)

    df_sum = (
        df_raw.groupby(["Dataset", "Fraction", "Model"])["TestMSE"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "MSE_Mean", "std": "MSE_Std", "count": "RunsUsed"})
    )

    sum_csv_path = os.path.join(args.out_dir, f"sample_efficiency_table_{args.dataset}.csv")
    df_sum.to_csv(sum_csv_path, index=False)

    tex_path = os.path.join(args.out_dir, f"sample_efficiency_table_{args.dataset}.tex")
    write_latex_table(df_sum, tex_path)

    # ========================================================
    # Plot (PNG)
    # ========================================================
    plt.figure(figsize=(10, 6))
    x = np.array(fractions) * 100

    # Match merge_eval_pub.py default color cycle order (C0,C1,C2,C3)
    style = {
        "RV-MLP": dict(color="C0", ls="--", marker="o"),  # model 1
        "RV-PURe": dict(color="C1", ls="-", marker="s"),  # model 2
        "CV-MLP": dict(color="C2", ls="--", marker="o"),  # model 3
        "CV-PURe": dict(color="C3", ls="-", marker="s"),  # model 4
    }

    for k in results:
        m = np.array([v[0] for v in results[k]], dtype=float)
        s = np.array([v[1] for v in results[k]], dtype=float)

        # skip if everything is nan
        if np.all(np.isnan(m)):
            continue

        plt.plot(x, m, lw=2, label=k, **style[k])

        # optional band only if std is available
        ok = ~np.isnan(m) & ~np.isnan(s)
        if np.any(ok):
            plt.fill_between(x[ok], (m - s)[ok], (m + s)[ok], alpha=0.12, color=style[k]["color"])

    plt.xlabel("Training Data Percentage (%)", fontsize=13)
    plt.ylabel("Test MSE (Standardized)", fontsize=13)
    # plt.title(f"Sample Efficiency Comparison ({args.dataset.upper()})", fontsize=15)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)

    save_path = os.path.join(args.out_dir, f"sample_efficiency_all_{args.dataset}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[OK] Saved plot:   {save_path}")
    print(f"[OK] Saved raw:    {raw_path}")
    print(f"[OK] Saved table:  {sum_csv_path}")
    print(f"[OK] Saved LaTeX:  {tex_path}")


if __name__ == "__main__":
    main()
