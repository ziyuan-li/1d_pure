#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
shap_compare_all.py

Merge of:
- 08_shap_compare.py (real-valued SHAP compare)
- 13_complex.py (complex-valued SHAP compare)

Goal:
- One figure per dataset with 4 panels (RV-MLP, RV-PU, CV-MLP, CV-PU).
- Keep x-axis feature names on ALL panels.
- Keep y-axis feature names ONLY on the leftmost panel.
- Make inter-panel spacing tighter (paper-style).
- Save PNG only.

Example:
python shap_compare_all.py \
  --real_dir ./checkpoints_test_ini0 \
  --complex_dir ./checkpoints_complex_test \
  --hidden_dim 64 \
  --run_id 0 \
  --out_dir ./pub_shap \
  --datasets friedman concrete california
"""

import os
import argparse
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_friedman1, fetch_california_housing

# ===== Real models =====
from mlp_models import build_4layer_mlp
from pu_models import RealPUResNet1D

# ===== Complex models =====
from complex_mlp_models import build_complex_mlp
from complex_pu_model import ComplexPUResNet1D


# ============================================================
# 1) Data loading (aligned with your evaluator scripts)
# ============================================================
def load_data(dataset_name: str):
    if dataset_name == "friedman":
        X, y = make_friedman1(n_samples=10000, n_features=10, noise=0.0, random_state=0)
        feature_names = [f"x{i}" for i in range(10)]
    elif dataset_name == "concrete":
        df = pd.read_excel("./data/concrete+compressive+strength/Concrete_Data.xls")
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
            y = model(X_c).real.detach().cpu().numpy().ravel()
        return y
    return predict_fn


# ============================================================
# 3) SHAP interaction matrix construction (PermutationExplainer)
# ============================================================
def compute_interaction_matrix(
    model: torch.nn.Module,
    X_train: np.ndarray,
    X_test: np.ndarray,
    predict_fn_builder,
    n_bg: int = 100,
    n_eval: int = 200,
    seed: int = 42,
) -> np.ndarray:
    """
    Returns normalized interaction strength matrix in [0, 1].
    Uses:
      - shap.PermutationExplainer
      - shap.utils.approximate_interactions
    """
    rs = np.random.RandomState(seed)
    device = next(model.parameters()).device
    predict_fn = predict_fn_builder(model, device)

    # background & evaluation sets
    n_bg = min(n_bg, X_train.shape[0])
    n_eval = min(n_eval, X_test.shape[0])
    bg_idx = rs.choice(X_train.shape[0], n_bg, replace=False)
    ev_idx = rs.choice(X_test.shape[0], n_eval, replace=False)

    X_bg = X_train[bg_idx]
    X_ev = X_test[ev_idx]

    # SHAP
    explainer = shap.PermutationExplainer(predict_fn, X_bg)
    shap_values = explainer(X_ev, silent=True)

    phi = shap_values.values if hasattr(shap_values, "values") else np.array(shap_values)
    # phi: (n_eval, n_features)
    n_features = phi.shape[1]

    # approximate interaction rankings -> convert to score matrix
    mat = np.zeros((n_features, n_features), dtype=float)
    for i in range(n_features):
        ranking = shap.utils.approximate_interactions(i, phi, X_ev)
        # higher score for higher-ranked interactions
        for rank, j in enumerate(ranking):
            mat[i, j] = n_features - rank

    # symmetrize + normalize
    mat = (mat + mat.T) / 2.0
    mx = mat.max()
    if mx > 0:
        mat = mat / mx
    return mat


# ============================================================
# 4) Load checkpoints (safe)
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
# 5) Plot: 4 panels in one image, x labels all, y labels left only
# ============================================================
def plot_4panel_interactions(
    mats: List[Optional[np.ndarray]],
    titles: List[str],
    feature_names: List[str],
    dataset: str,
    out_path: str,
):
    n = 4
    assert len(mats) == n and len(titles) == n

    # ===== Canvas =====
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.8))

    # tighter spacing (keep your original intention)
    fig.subplots_adjust(wspace=-0.25, bottom=0.28, right=0.88, top=0.88)

    last_im = None
    for idx, ax in enumerate(axes):
        # ---------- title ----------
        ax.set_title(titles[idx], fontsize=17)

        if mats[idx] is None:
            ax.axis("off")
            ax.text(
                0.5, 0.5, "Missing\ncheckpoint",
                ha="center", va="center", fontsize=14
            )
            continue

        last_im = ax.imshow(
            mats[idx],
            origin="lower",
            vmin=0,
            vmax=1,
            cmap="viridis"
        )

        # ---------- x-axis ----------
        ax.set_xticks(range(len(feature_names)))
        ax.set_xticklabels(
            feature_names,
            rotation=45,
            ha="right",
            fontsize=17
        )

        # ---------- y-axis ----------
        ax.set_yticks(range(len(feature_names)))
        if idx == 0:
            ax.set_yticklabels(feature_names, fontsize=17)
            ax.set_ylabel("Feature", fontsize=17)
        else:
            ax.set_yticklabels([])
            ax.tick_params(left=False)

    # ---------- shared colorbar ----------
    if last_im is not None:
        cax = fig.add_axes([0.865, 0.27, 0.020, 0.62])
        cb = fig.colorbar(last_im, cax=cax)
        cb.set_label("Normalized Interaction Strength", fontsize=17)
        cb.ax.tick_params(labelsize=11)

    # ---------- super title ----------
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
    parser.add_argument("--out_dir", type=str, default="./pub_shap")

    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--real_dir", type=str, default="./checkpoints_test_ini0")
    parser.add_argument("--complex_dir", type=str, default="./checkpoints_complex_test")
    parser.add_argument("--hidden_dim", type=int, default=64)

    parser.add_argument("--n_bg", type=int, default=100)
    parser.add_argument("--n_eval", type=int, default=200)

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # fixed order: always 4 panels
    panel_specs = [
        ("rv_mlp", "RV-MLP", "mlp4_h128"),
        ("rv_pu",  "RV-PURe",  "puresnet_h128"),
        ("cv_mlp", "CV-MLP", None),  # tag includes hidden_dim
        ("cv_pu",  "CV-PURe",  None),
    ]

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

        for idx, (kind, title, tag0) in enumerate(panel_specs):
            if kind in ("rv_mlp", "rv_pu"):
                tag = tag0
                ckpt_path = os.path.join(args.real_dir, f"{tag}_{ds}_run{args.run_id}_best.pth")
            else:
                # complex tags follow: complex_mlp_h{hidden_dim} / complex_pu_h{hidden_dim}
                tag = f"{'complex_mlp' if kind=='cv_mlp' else 'complex_pu'}_h{args.hidden_dim}"
                ckpt_path = os.path.join(args.complex_dir, f"{tag}_{ds}_run{args.run_id}_best.pth")

            model = load_model_from_ckpt(kind, ckpt_path, input_dim, args.hidden_dim, device)
            if model is None:
                print(f"  - {title}: missing {ckpt_path}")
                mats[idx] = None
                continue

            print(f"  - {title}: computing interactions...")
            predict_builder = build_predict_fn_real if kind.startswith("rv_") else build_predict_fn_complex
            mats[idx] = compute_interaction_matrix(
                model,
                X_train,
                X_test,
                predict_builder,
                n_bg=args.n_bg,
                n_eval=args.n_eval,
                seed=args.seed,
            )

        out_path = os.path.join(args.out_dir, f"shap_compare_all_{ds}_run{args.run_id}.png")
        plot_4panel_interactions(mats, titles, feature_names, ds, out_path)
        print(f"[OK] saved: {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
