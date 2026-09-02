#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SHAP interaction comparison for all four paper models.

Key changes vs your version:
1) rank->score uses exponential decay exp(-alpha*rank) instead of linear (n_features-rank)
   -> makes maps less uniformly filled, easier to distinguish concentration.
2) interaction metrics support Top-K: compute entropy/HHI on only strongest K off-diagonal entries
3) annotate plots with H_topK (and optionally HHI_topK)
4) save richer CSV summary

NOTE:
Interaction matrix is derived from approximate_interactions ranking (not true SHAP interaction values),
so metrics quantify concentration of the ranking-based interaction map.
"""

import os
import argparse
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_friedman1, fetch_california_housing

# ===== Real models =====
from pure_resmlp.models.mlp import build_4layer_mlp
from pure_resmlp.models.pure import RealPUResNet1D

# ===== Complex models =====
from pure_resmlp.models.complex_mlp import build_complex_mlp
from pure_resmlp.models.complex_pure import ComplexPUResNet1D
from pure_resmlp.paths import (
    DATA_DIR,
    MAIN_COMPLEX_CHECKPOINT_DIR,
    MAIN_REAL_CHECKPOINT_DIR,
    SHAP_RESULTS_DIR,
)


# ============================================================
# 0) Interaction metrics (Entropy + HHI) with optional Top-K
# ============================================================
def _offdiag_abs(mat: np.ndarray) -> np.ndarray:
    A = np.abs(np.array(mat, dtype=np.float64))
    np.fill_diagonal(A, 0.0)
    return A


def _topk_mask_offdiag(A: np.ndarray, topk: Optional[int]) -> np.ndarray:
    """
    Return a masked version of A that keeps only topk largest off-diagonal entries (global),
    and sets the rest to 0. If topk is None -> return A unchanged.
    """
    if topk is None:
        return A

    d = A.shape[0]
    assert A.shape == (d, d)

    # flatten off-diagonal entries
    mask_off = ~np.eye(d, dtype=bool)
    vals = A[mask_off]
    if topk <= 0:
        return np.zeros_like(A)
    if topk >= vals.size:
        return A

    # find threshold for topk
    # argpartition gives indices of topk largest values
    idx = np.argpartition(vals, -topk)[-topk:]
    keep = np.zeros_like(vals, dtype=bool)
    keep[idx] = True

    A2 = np.zeros_like(A)
    A2[mask_off] = vals * keep
    return A2


def interaction_metrics(mat: np.ndarray, topk: Optional[int] = None, eps: float = 1e-12) -> Dict[str, float]:
    """
    Compute:
      - H_norm: normalized Shannon entropy of off-diagonal mass in [0,1] (lower = more concentrated)
      - N_eff: exp(H) (effective number of interaction pairs; lower = more concentrated)
      - HHI: sum(p^2) (higher = more concentrated)

    If topk is provided, metrics are computed on topk strongest off-diagonal entries (global).
    """
    A = _offdiag_abs(mat)
    A = _topk_mask_offdiag(A, topk)

    s = A.sum()
    d = A.shape[0]
    if s <= eps:
        # no interactions (or topk too small)
        return {"H": 0.0, "H_norm": 0.0, "N_eff": 1.0, "HHI": 1.0}

    P = A / s
    off = P[~np.eye(d, dtype=bool)]
    off = off[off > eps]

    H = float(-np.sum(off * np.log(off)))
    H_norm = float(H / np.log(d * (d - 1)))  # comparable across same d
    N_eff = float(np.exp(H))
    HHI = float(np.sum(off ** 2))
    return {"H": H, "H_norm": H_norm, "N_eff": N_eff, "HHI": HHI}


# ============================================================
# 1) Data loading
# ============================================================
def load_data(dataset_name: str):
    if dataset_name == "friedman":
        X, y = make_friedman1(n_samples=10000, n_features=10, noise=0.0, random_state=0)
        feature_names = [f"x{i}" for i in range(10)]
    elif dataset_name == "concrete":
        df = pd.read_excel(DATA_DIR / "concrete+compressive+strength" / "Concrete_Data.xls")
        X = df.iloc[:, :8].values
        y = df.iloc[:, 8].values
        feature_names = ["Cement", "Slag", "Ash", "Water",
                         "Superplast", "Coarse", "Fine", "Age"]
    elif dataset_name == "california":
        data = fetch_california_housing()
        X, y = data.data, data.target
        feature_names = list(data.feature_names)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=0)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=0)

    x_scaler = StandardScaler()
    X_train_s = x_scaler.fit_transform(X_train)
    X_test_s = x_scaler.transform(X_test)

    return X_train_s, X_test_s, feature_names


# ============================================================
# 2) Predict functions for SHAP
# ============================================================
def build_predict_fn_real(model: torch.nn.Module, device: torch.device):
    def predict_fn(X_numpy: np.ndarray) -> np.ndarray:
        X_t = torch.from_numpy(X_numpy).float().to(device)
        with torch.no_grad():
            y = model(X_t).detach().cpu().numpy().ravel()
        return y
    return predict_fn


def build_predict_fn_complex(model: torch.nn.Module, device: torch.device):
    def predict_fn(X_numpy: np.ndarray) -> np.ndarray:
        X_t = torch.from_numpy(X_numpy).float().to(device)
        X_c = X_t.to(torch.complex64)  # imag = 0
        with torch.no_grad():
            y = model(X_c).detach().cpu().numpy().ravel()
        return y
    return predict_fn


# ============================================================
# 3) SHAP interaction matrix (PermutationExplainer + approximate_interactions)
# ============================================================
def compute_interaction_matrix(
    model: torch.nn.Module,
    X_train: np.ndarray,
    X_test: np.ndarray,
    predict_fn_builder,
    n_bg: int = 100,
    n_eval: int = 200,
    seed: int = 42,
    rank_decay_alpha: float = 0.6,
    per_row_topk: Optional[int] = None,
) -> np.ndarray:
    """
    Returns normalized interaction strength matrix in [0, 1].

    - shap.PermutationExplainer for SHAP values
    - shap.utils.approximate_interactions gives a ranking for feature interactions
    - We convert ranking -> score via exp(-alpha*rank) to emphasize top interactions.

    per_row_topk: if set, only fill top-k interactions per row (faster + sharper).
    """
    rs = np.random.RandomState(seed)
    device = next(model.parameters()).device
    predict_fn = predict_fn_builder(model, device)

    n_bg = min(n_bg, X_train.shape[0])
    n_eval = min(n_eval, X_test.shape[0])
    bg_idx = rs.choice(X_train.shape[0], n_bg, replace=False)
    ev_idx = rs.choice(X_test.shape[0], n_eval, replace=False)

    X_bg = X_train[bg_idx]
    X_ev = X_test[ev_idx]

    explainer = shap.PermutationExplainer(predict_fn, X_bg)
    shap_values = explainer(X_ev, silent=True)

    phi = shap_values.values if hasattr(shap_values, "values") else np.array(shap_values)
    n_features = phi.shape[1]

    mat = np.zeros((n_features, n_features), dtype=np.float64)

    for i in range(n_features):
        ranking = shap.utils.approximate_interactions(i, phi, X_ev)

        if per_row_topk is not None:
            ranking = ranking[: min(per_row_topk, len(ranking))]

        for rank, j in enumerate(ranking):
            # exponential decay score: top ranks dominate
            mat[i, j] = np.exp(-rank_decay_alpha * rank)

    # symmetrize + normalize
    mat = (mat + mat.T) / 2.0
    mx = mat.max()
    if mx > 0:
        mat = mat / mx
    return mat


# ============================================================
# 4) Load checkpoints
# ============================================================
def load_model_from_ckpt(
    model_kind: str,
    ckpt_path: str,
    input_dim: int,
    hidden_dim: int,
    device: torch.device,
) -> Optional[torch.nn.Module]:
    if not os.path.exists(ckpt_path):
        return None

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)

    if model_kind == "rv_mlp":
        model = build_4layer_mlp(input_dim, 1, 128)
    elif model_kind == "rv_pu":
        model = RealPUResNet1D(input_dim, 128, 1)
    elif model_kind == "cv_mlp":
        model = build_complex_mlp(input_dim, 1, hidden_dim)
    elif model_kind == "cv_pu":
        model = ComplexPUResNet1D(input_dim, hidden_dim, 1)
    else:
        raise ValueError(f"Unknown model_kind: {model_kind}")

    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


# ============================================================
# 5) Plot + annotate metrics
# ============================================================
def plot_4panel_interactions(
    mats: List[Optional[np.ndarray]],
    titles: List[str],
    feature_names: List[str],
    dataset: str,
    out_path: str,
    metrics: Optional[List[Optional[Dict[str, float]]]] = None,
    show_hhi: bool = False,
):
    n = 4
    assert len(mats) == n and len(titles) == n
    if metrics is not None:
        assert len(metrics) == n

    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.8))
    fig.subplots_adjust(wspace=-0.25, bottom=0.28, right=0.88, top=0.88)

    last_im = None
    for idx, ax in enumerate(axes):
        ax.set_title(titles[idx], fontsize=17)

        if mats[idx] is None:
            ax.axis("off")
            ax.text(0.5, 0.5, "Missing\ncheckpoint", ha="center", va="center", fontsize=14)
            continue

        last_im = ax.imshow(mats[idx], origin="lower", vmin=0, vmax=1, cmap="viridis")

        ax.set_xticks(range(len(feature_names)))
        ax.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=17)

        ax.set_yticks(range(len(feature_names)))
        if idx == 0:
            ax.set_yticklabels(feature_names, fontsize=17)
            ax.set_ylabel("Feature", fontsize=17)
        else:
            ax.set_yticklabels([])
            ax.tick_params(left=False)

        if metrics is not None and metrics[idx] is not None:
            Hn = metrics[idx].get("H_norm", np.nan)
            if show_hhi:
                HHI = metrics[idx].get("HHI", np.nan)
                label = f"H={Hn:.3f}\nHHI={HHI:.3f}"
            else:
                label = f"H={Hn:.3f}"

            ax.text(
                0.97, 0.97,
                label,
                transform=ax.transAxes,
                ha="right", va="top",
                fontsize=13,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, linewidth=0.0),
            )

    if last_im is not None:
        cax = fig.add_axes([0.865, 0.27, 0.020, 0.62])
        cb = fig.colorbar(last_im, cax=cax)
        cb.set_label("Normalized Interaction Strength", fontsize=17)
        cb.ax.tick_params(labelsize=11)

    fig.suptitle(
        f"SHAP Feature Interaction Comparison ({dataset.upper()})",
        fontsize=20,
        y=1.05
    )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 6) Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["friedman", "concrete", "california"])
    parser.add_argument("--out_dir", type=str, default=str(SHAP_RESULTS_DIR))

    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--real_dir", type=str, default=str(MAIN_REAL_CHECKPOINT_DIR))
    parser.add_argument("--complex_dir", type=str, default=str(MAIN_COMPLEX_CHECKPOINT_DIR))
    parser.add_argument("--hidden_dim", type=int, default=64)

    parser.add_argument("--n_bg", type=int, default=100)
    parser.add_argument("--n_eval", type=int, default=200)

    # NEW: sharpen ranking->score
    parser.add_argument("--rank_decay_alpha", type=float, default=0.6,
                        help="exp(-alpha*rank): larger alpha => more concentrated maps")
    parser.add_argument("--per_row_topk", type=int, default=0,
                        help="keep only top-k interactions per row; 0 means keep all")

    # NEW: metrics computed on top-K strongest global interactions
    parser.add_argument("--metric_topk", type=int, default=10,
                        help="compute and annotate metrics on global top-K off-diagonal entries; <=0 means full-matrix")
    parser.add_argument("--show_hhi", action="store_true", help="also annotate HHI on each panel")

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    panel_specs = [
        ("rv_mlp", "RV-MLP", "mlp4_h128"),
        ("rv_pu",  "RV-PURe", "puresnet_h128"),
        ("cv_mlp", "CV-MLP", None),
        ("cv_pu",  "CV-PURe", None),
    ]

    summary_rows: List[Dict[str, Any]] = []

    per_row_topk = args.per_row_topk if args.per_row_topk and args.per_row_topk > 0 else None
    metric_topk = args.metric_topk if args.metric_topk and args.metric_topk > 0 else None

    for ds in args.datasets:
        print(f"\n{'='*60}\nDataset: {ds.upper()}\n{'='*60}")

        try:
            X_train, X_test, feature_names = load_data(ds)
        except Exception as e:
            print(f"[SKIP] failed to load {ds}: {e}")
            continue

        input_dim = X_train.shape[1]

        mats: List[Optional[np.ndarray]] = [None, None, None, None]
        titles: List[str] = [p[1] for p in panel_specs]

        # compute two sets of metrics: full + topk
        metrics_for_plot: List[Optional[Dict[str, float]]] = [None, None, None, None]

        for idx, (kind, title, tag0) in enumerate(panel_specs):
            if kind in ("rv_mlp", "rv_pu"):
                tag = tag0
                ckpt_path = os.path.join(args.real_dir, f"{tag}_{ds}_run{args.run_id}_best.pth")
            else:
                tag = f"{'complex_mlp' if kind=='cv_mlp' else 'complex_pu'}_h{args.hidden_dim}"
                ckpt_path = os.path.join(args.complex_dir, f"{tag}_{ds}_run{args.run_id}_best.pth")

            model = load_model_from_ckpt(kind, ckpt_path, input_dim, args.hidden_dim, device)
            if model is None:
                print(f"  - {title}: missing {ckpt_path}")
                mats[idx] = None
                metrics_for_plot[idx] = None
                summary_rows.append({
                    "dataset": ds, "panel": title, "kind": kind,
                    "H_norm_full": np.nan, "N_eff_full": np.nan, "HHI_full": np.nan,
                    "H_norm_topk": np.nan, "N_eff_topk": np.nan, "HHI_topk": np.nan,
                    "topk": metric_topk if metric_topk is not None else "full",
                    "alpha": args.rank_decay_alpha, "per_row_topk": per_row_topk,
                    "ckpt": ckpt_path, "status": "missing",
                })
                continue

            print(f"  - {title}: computing interactions...")
            predict_builder = build_predict_fn_real if kind.startswith("rv_") else build_predict_fn_complex

            mat = compute_interaction_matrix(
                model,
                X_train,
                X_test,
                predict_builder,
                n_bg=args.n_bg,
                n_eval=args.n_eval,
                seed=args.seed,
                rank_decay_alpha=args.rank_decay_alpha,
                per_row_topk=per_row_topk,
            )
            mats[idx] = mat

            # metrics
            m_full = interaction_metrics(mat, topk=None)
            m_topk = interaction_metrics(mat, topk=metric_topk)

            # what to annotate on plot: topk version (more discriminative)
            metrics_for_plot[idx] = m_topk

            print(f"    full: H_norm={m_full['H_norm']:.4f}, HHI={m_full['HHI']:.4f}")
            if metric_topk is not None:
                print(f"    topK={metric_topk}: H_norm={m_topk['H_norm']:.4f}, N_eff={m_topk['N_eff']:.2f}, HHI={m_topk['HHI']:.4f}")
            else:
                print(f"    topK=full: H_norm={m_topk['H_norm']:.4f}, N_eff={m_topk['N_eff']:.2f}, HHI={m_topk['HHI']:.4f}")

            summary_rows.append({
                "dataset": ds, "panel": title, "kind": kind,
                "H_norm_full": m_full["H_norm"], "N_eff_full": m_full["N_eff"], "HHI_full": m_full["HHI"],
                "H_norm_topk": m_topk["H_norm"], "N_eff_topk": m_topk["N_eff"], "HHI_topk": m_topk["HHI"],
                "topk": metric_topk if metric_topk is not None else "full",
                "alpha": args.rank_decay_alpha, "per_row_topk": per_row_topk,
                "ckpt": ckpt_path, "status": "ok",
            })

        out_path = os.path.join(args.out_dir, f"shap_compare_all_{ds}_run{args.run_id}.png")
        plot_4panel_interactions(
            mats, titles, feature_names, ds, out_path,
            metrics=metrics_for_plot,
            show_hhi=args.show_hhi,
        )
        print(f"[OK] saved: {out_path}")

    if summary_rows:
        df = pd.DataFrame(summary_rows)
        csv_path = os.path.join(args.out_dir, f"shap_interaction_metrics_run{args.run_id}.csv")
        df.to_csv(csv_path, index=False)
        print(f"[OK] saved metrics table: {csv_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
