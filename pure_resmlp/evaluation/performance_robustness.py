#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate prediction error and input-noise sensitivity for four neural models.

Write per-run results.csv, results_table.tex, loss_<dataset>.png, and
mse_<dataset>.png under --out_dir. Loss curves show mean training and
validation error; noise curves show test MSE with error bars.

Run with python -m pure_resmlp.evaluation.performance_robustness.
"""

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

from sklearn.datasets import make_friedman1, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pure_resmlp.paths import (
    DATA_DIR,
    MAIN_COMPLEX_CHECKPOINT_DIR,
    MAIN_REAL_CHECKPOINT_DIR,
    PERFORMANCE_RESULTS_DIR,
)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def standardize_splits(X_train, X_val, X_test, y_train, y_val, y_test):
    x_scaler, y_scaler = StandardScaler(), StandardScaler()
    X_train_s = x_scaler.fit_transform(X_train)
    X_val_s = x_scaler.transform(X_val)
    X_test_s = x_scaler.transform(X_test)
    y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_s = y_scaler.transform(y_val.reshape(-1, 1)).ravel()
    y_test_s = y_scaler.transform(y_test.reshape(-1, 1)).ravel()
    return X_train_s, X_val_s, X_test_s, y_train_s, y_val_s, y_test_s


def load_friedman():
    X, y = make_friedman1(n_samples=10000, n_features=10, noise=0.0, random_state=0)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=0)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=0)
    return X_train, X_val, X_test, y_train, y_val, y_test


def load_concrete():
    df = pd.read_excel(DATA_DIR / "concrete+compressive+strength" / "Concrete_Data.xls")
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


def eval_on_test(model: nn.Module, loader: DataLoader, device: torch.device, *, is_complex: bool) -> float:
    """Return mean MSE; complex backbones expose a real projection output."""
    criterion = nn.MSELoss()
    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            if is_complex and not X.is_complex():
                X = X.to(torch.complex64)
            pred = model(X)
            total_loss += criterion(pred, y).item() * X.size(0)
            n += X.size(0)
    return total_loss / max(n, 1)


def read_loss_npz(loss_path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    try:
        ld = np.load(loss_path)
        return ld["train_losses"], ld["val_losses"]
    except Exception:
        return None


def _stack_runs(curves: List[np.ndarray]) -> Optional[np.ndarray]:
    """Stack run curves to shape (runs, epochs) after trimming to min length."""
    if not curves:
        return None
    series = []
    for a in curves:
        a = np.asarray(a)
        if a.ndim == 2:  # sometimes stored as (k, epochs)
            a = a[0]
        series.append(a)
    min_len = min(len(s) for s in series)
    if min_len <= 0:
        return None
    return np.stack([s[:min_len] for s in series], axis=0)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    key: str
    family: str
    tag_template: str
    is_complex: bool


def get_model_specs(hidden_dim: int) -> List[ModelSpec]:
    return [
        ModelSpec(name="RV-MLP", key="rv_mlp", family="real", tag_template="mlp4_h128", is_complex=False),
        ModelSpec(name="RV-PURe", key="rv_pu", family="real", tag_template="puresnet_h128", is_complex=False),
        ModelSpec(name="CV-MLP", key="cv_mlp", family="complex", tag_template=f"complex_mlp_h{hidden_dim}", is_complex=True),
        ModelSpec(name="CV-PURe", key="cv_pu", family="complex", tag_template=f"complex_pu_h{hidden_dim}", is_complex=True),
    ]


def build_model(spec: ModelSpec, input_dim: int, out_dim: int, h_dim: int) -> nn.Module:
    """Build the model selected by its registry name."""
    if spec.family == "real":
        from pure_resmlp.models.mlp import build_4layer_mlp
        from pure_resmlp.models.pure import RealPUResNet1D
        if spec.key == "rv_mlp":
            return build_4layer_mlp(input_dim, out_dim, h_dim)
        return RealPUResNet1D(input_dim, h_dim, out_dim)

    from pure_resmlp.models.complex_mlp import build_complex_mlp
    from pure_resmlp.models.complex_pure import ComplexPUResNet1D
    if spec.key == "cv_mlp":
        return build_complex_mlp(input_dim, out_dim, h_dim)
    return ComplexPUResNet1D(input_dim, h_dim, out_dim)


def ckpt_paths(base_dir: str, tag: str, dataset: str, run: int) -> Tuple[str, str]:
    ckpt = os.path.join(base_dir, f"{tag}_{dataset}_run{run}_best.pth")
    loss = os.path.join(base_dir, f"{tag}_{dataset}_run{run}_losses.npz")
    return ckpt, loss


def plot_loss_curves(loss_curves: Dict[str, Dict[str, List[np.ndarray]]],
                     dataset: str,
                     out_dir: str) -> None:
    """Plot mean log10 loss with solid validation and dashed training curves."""

    LABEL_MAP = {
        "rv_mlp": "RV-MLP",
        "rv_pu": "RV-PURe",
        "cv_mlp": "CV-MLP",
        "cv_pu": "CV-PURe",
    }

    plt.figure(figsize=(6.5, 4.0))
    plotted = 0

    for mkey, tv in loss_curves.items():
        label_base = LABEL_MAP.get(mkey, mkey.upper())

        S_val = _stack_runs(tv.get("val", []))
        S_tr  = _stack_runs(tv.get("train", []))

        color = None

        if S_val is not None:
            mean_val = np.mean(S_val, axis=0)
            x = np.arange(1, len(mean_val) + 1)
            y = np.log10(mean_val + 1e-12)

            (line_val,) = plt.plot(x, y, linestyle="-", label=f"{label_base} (val)")
            color = line_val.get_color()
            plotted += 1

        if S_tr is not None:
            mean_tr = np.mean(S_tr, axis=0)
            x = np.arange(1, len(mean_tr) + 1)
            y = np.log10(mean_tr + 1e-12)

            if color is None:
                # if val is missing but train exists, let matplotlib pick a new color
                plt.plot(x, y, linestyle="--", linewidth=0.75, label=f"{label_base} (train)")
            else:
                plt.plot(x, y, linestyle="--", linewidth=0.75, color=color, label=f"{label_base} (train)")
            plotted += 1

    if plotted == 0:
        plt.close()
        return

    plt.xlabel("Epoch")
    plt.ylabel(r"$\log_{10}$ (MSE)")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()

    plt.savefig(os.path.join(out_dir, f"loss_{dataset}.png"), dpi=300)
    plt.close()



def plot_test_mse_vs_noise(df: pd.DataFrame, dataset: str, out_dir: str) -> None:
    """Line plot: noise std vs test MSE mean, with error bars (PNG only)."""
    sub = df[df["Dataset"] == dataset].copy()
    if sub.empty:
        return

    plt.figure(figsize=(6.5, 4.0))
    for model in sub["Model"].unique():
        ss = sub[sub["Model"] == model].sort_values("Noise")
        plt.errorbar(
            ss["Noise"].values,
            ss["MSE_Mean"].values,
            yerr=ss["MSE_Std"].values,
            marker="o",
            capsize=3,
            label=model,
        )

    plt.xlabel("Input noise std")
    plt.ylabel("Test MSE")
    plt.title(f"Robustness on {dataset.upper()}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(out_dir, f"mse_{dataset}.png"), dpi=300)
    plt.close()


def make_latex_table(df: pd.DataFrame, out_path: str) -> None:
    if df.empty:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("% Empty results.\n")
        return

    d = df.copy()
    d["Cell"] = d.apply(lambda r: f"{r['MSE_Mean']:.4f} $\\pm$ {r['MSE_Std']:.4f}", axis=1)

    best_idx = d.groupby(["Dataset", "Noise"])["MSE_Mean"].idxmin()
    d.loc[best_idx, "Cell"] = d.loc[best_idx, "Cell"].apply(lambda s: f"\\textbf{{{s}}}")

    pivot = d.pivot_table(index=["Dataset", "Noise"], columns="Model", values="Cell", aggfunc="first")

    params = df.groupby("Model")["Params"].max().to_dict()
    col_order = [c for c in ["RV-MLP", "RV-PURe", "CV-MLP", "CV-PURe"] if c in pivot.columns]
    pivot = pivot[col_order]

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\setlength{\\tabcolsep}{6pt}")
    lines.append("\\renewcommand{\\arraystretch}{1.15}")

    param_str = ", ".join(
        [f"{m}: {params.get(m, 'NA'):,}" if isinstance(params.get(m, None), (int, np.integer)) else f"{m}: NA"
         for m in col_order]
    )

    lines.append(
        "\\caption{Test MSE (mean $\\pm$ std over runs). Best per dataset/noise is in bold. "
        f"Parameter counts: {param_str}.}}"
    )

    lines.append("\\begin{tabular}{ll" + "c" * len(col_order) + "}")
    lines.append("\\toprule")
    header = ["Dataset", "Noise"] + col_order
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")
    for (ds, nz), row in pivot.iterrows():
        row_cells = [ds, f"{nz:.2f}"] + [row.get(m, "--") for m in col_order]
        lines.append(" & ".join(row_cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_dir", type=str, default=str(MAIN_REAL_CHECKPOINT_DIR))
    parser.add_argument("--complex_dir", type=str, default=str(MAIN_COMPLEX_CHECKPOINT_DIR))
    parser.add_argument("--out_dir", type=str, default=str(PERFORMANCE_RESULTS_DIR))
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--noise_stds", type=float, nargs="+", default=[0.0, 0.05])
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dir(args.out_dir)

    datasets = ["friedman", "concrete", "california"]
    noise_stds = sorted(set(args.noise_stds))
    if 0.0 not in noise_stds:
        noise_stds = [0.0] + noise_stds

    specs = get_model_specs(args.hidden_dim)

    final_rows = []
    all_loss_curves: Dict[str, Dict[str, Dict[str, List[np.ndarray]]]] = {
        d: {s.key: {"train": [], "val": []} for s in specs} for d in datasets
    }

    for dname in datasets:
        try:
            if dname == "friedman":
                data_splits = load_friedman()
            elif dname == "concrete":
                data_splits = load_concrete()
            else:
                data_splits = load_california()
        except Exception as e:
            print(f"[WARN] cannot load dataset {dname}: {e}")
            continue

        X_train_s, X_val_s, X_test_s, y_train_s, y_val_s, y_test_s = standardize_splits(*data_splits)
        input_dim = X_train_s.shape[1]

        # Reuse the same noise samples across models.
        rng = np.random.RandomState(123)
        test_noises = {
            std: (np.zeros_like(X_test_s) if std == 0.0 else rng.normal(0, std, X_test_s.shape))
            for std in noise_stds
        }

        for spec in specs:
            base_dir = args.real_dir if spec.family == "real" else args.complex_dir
            run_metrics = {std: [] for std in noise_stds}
            param_count: Optional[int] = None

            for run in range(args.runs):
                tag = spec.tag_template
                ckpt_path, loss_path = ckpt_paths(base_dir, tag, dname, run)
                if not os.path.exists(ckpt_path):
                    continue

                ckpt = torch.load(ckpt_path, map_location="cpu")
                h_dim = int(ckpt.get("hidden_dim", args.hidden_dim))

                model = build_model(spec, input_dim, 1, h_dim)
                model.load_state_dict(ckpt["model_state_dict"])
                model.to(device)

                if param_count is None:
                    param_count = count_parameters(model)

                for std in noise_stds:
                    X_noisy = torch.from_numpy(X_test_s + test_noises[std]).float()
                    y_true = torch.from_numpy(y_test_s).float().unsqueeze(-1)
                    loader = DataLoader(
                        TensorDataset(X_noisy, y_true),
                        batch_size=args.batch_size,
                        shuffle=False,
                    )
                    run_metrics[std].append(eval_on_test(model, loader, device, is_complex=spec.is_complex))

                lc = read_loss_npz(loss_path)
                if lc is not None:
                    tr, va = lc
                    all_loss_curves[dname][spec.key]["train"].append(tr)
                    all_loss_curves[dname][spec.key]["val"].append(va)

            if param_count is None:
                continue

            for std in noise_stds:
                vals = np.asarray(run_metrics[std], dtype=float)
                if vals.size == 0:
                    continue
                final_rows.append({
                    "Dataset": dname,
                    "Model": spec.name,
                    "ModelKey": spec.key,
                    "Family": spec.family,
                    "Params": int(param_count),
                    "Noise": float(std),
                    "MSE_Mean": float(vals.mean()),
                    "MSE_Std": float(vals.std(ddof=0)),
                    "RunsUsed": int(vals.size),
                })

        plot_loss_curves(all_loss_curves[dname], dname, args.out_dir)

    df = pd.DataFrame(final_rows)
    df.to_csv(os.path.join(args.out_dir, "results.csv"), index=False)

    make_latex_table(df, os.path.join(args.out_dir, "results_table.tex"))

    for dname in df["Dataset"].unique():
        plot_test_mse_vs_noise(df, dname, args.out_dir)

    print(f"[OK] Wrote artifacts to: {args.out_dir}")
    print("     - results.csv")
    print("     - results_table.tex")
    print("     - loss_<dataset>.png")
    print("     - mse_<dataset>.png")


if __name__ == "__main__":
    main()
