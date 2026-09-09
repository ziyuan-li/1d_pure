#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare sample efficiency of real- and complex-valued MLP and PURe models.

Write sample_efficiency_all_<dataset>.png, per-run
sample_efficiency_raw_<dataset>.csv, and mean/SD summaries in
sample_efficiency_table_<dataset>.csv and .tex under --out_dir.
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

from pure_resmlp.models.mlp import build_4layer_mlp
from pure_resmlp.models.pure import RealPUResNet1D

from pure_resmlp.models.complex_mlp import build_complex_mlp
from pure_resmlp.models.complex_pure import ComplexPUResNet1D
from pure_resmlp.paths import (
    DATA_DIR,
    EFFICIENCY_COMPLEX_CHECKPOINT_DIR,
    EFFICIENCY_REAL_CHECKPOINT_DIR,
    EFFICIENCY_RESULTS_DIR,
)


def load_raw_test_data(dataset: str):
    if dataset == "friedman":
        X, y = make_friedman1(n_samples=10000, n_features=10, noise=0.0, random_state=0)
    elif dataset == "concrete":
        df = pd.read_excel(DATA_DIR / "concrete+compressive+strength" / "Concrete_Data.xls")
        X = df.iloc[:, :8].values
        y = df.iloc[:, 8].values
    elif dataset == "california":
        data = fetch_california_housing()
        X, y = data.data, data.target
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    return X_test, y_test


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
        pred = model(X_t.to(torch.complex64))
        return float(nn.MSELoss()(pred, y_t).item())


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

    def _fmt(r):
        std = r["MSE_Std"]
        if pd.isna(std):
            std = 0.0
        return f"{r['MSE_Mean']:.4f} $\\pm$ {std:.4f}"

    d["Cell"] = d.apply(_fmt, axis=1)

    best_idx = d.groupby(["Dataset", "Fraction"])["MSE_Mean"].idxmin()
    d.loc[best_idx, "Cell"] = d.loc[best_idx, "Cell"].apply(lambda s: f"\\textbf{{{s}}}")

    pivot = d.pivot_table(index=["Dataset", "Fraction"], columns="Model", values="Cell", aggfunc="first")

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="concrete", choices=["friedman", "concrete", "california"])
    parser.add_argument("--runs", type=int, default=5)

    parser.add_argument("--real_dir", default=str(EFFICIENCY_REAL_CHECKPOINT_DIR))
    parser.add_argument("--complex_dir", default=str(EFFICIENCY_COMPLEX_CHECKPOINT_DIR))
    parser.add_argument("--hidden_dim", type=int, default=64)

    parser.add_argument("--fractions", default="0.05,0.1,0.2,0.4,0.6,0.8")
    parser.add_argument("--out_dir", default=str(EFFICIENCY_RESULTS_DIR))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fractions = [float(x) for x in args.fractions.split(",")]
    X_test, y_test = load_raw_test_data(args.dataset)

    results = {
        "RV-MLP": [], "RV-PURe": [],
        "CV-MLP": [], "CV-PURe": []
    }

    rows = []

    for frac in fractions:
        buf = {k: [] for k in results}

        for run in range(args.runs):
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

        for k in results:
            arr = np.array(buf[k], dtype=float)
            if arr.size == 0:
                results[k].append((np.nan, np.nan))
            else:
                results[k].append((float(np.mean(arr)), float(np.std(arr, ddof=0))))

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

    plt.figure(figsize=(10, 6))
    x = np.array(fractions) * 100

    # Use the same model colors as performance_robustness.py.
    style = {
        "RV-MLP": dict(color="C0", ls="--", marker="o"),
        "RV-PURe": dict(color="C1", ls="-", marker="s"),
        "CV-MLP": dict(color="C2", ls="--", marker="o"),
        "CV-PURe": dict(color="C3", ls="-", marker="s"),
    }

    for k in results:
        m = np.array([v[0] for v in results[k]], dtype=float)
        s = np.array([v[1] for v in results[k]], dtype=float)

        if np.all(np.isnan(m)):
            continue

        plt.plot(x, m, lw=2, label=k, **style[k])

        ok = ~np.isnan(m) & ~np.isnan(s)
        if np.any(ok):
            plt.fill_between(x[ok], (m - s)[ok], (m + s)[ok], alpha=0.12, color=style[k]["color"])

    plt.xlabel("Training Data Percentage (%)", fontsize=13)
    plt.ylabel("Test MSE (Standardized)", fontsize=13)
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
