import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_friedman1, fetch_california_housing

from pure_resmlp.models.complex_pure import ComplexPUResNet1D
from pure_resmlp.models.complex_mlp import build_complex_mlp


def load_data(dataset_name):
    if dataset_name == "friedman":
        X, y = make_friedman1(n_samples=10000, n_features=10, noise=0.0, random_state=0)
        feature_names = [f"x{i}" for i in range(10)]
    elif dataset_name == "concrete":
        df = pd.read_excel("./data/concrete+compressive+strength/Concrete_Data.xls")
        X = df.iloc[:, :8].values
        y = df.iloc[:, 8].values
        feature_names = ["Cement", "Slag", "Ash", "Water", "Superplast", "Coarse", "Fine", "Age"]
    else:  # california
        data = fetch_california_housing()
        X, y = data.data, data.target
        feature_names = data.feature_names

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=0)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=0)

    x_scaler = StandardScaler()
    X_train_s = x_scaler.fit_transform(X_train)
    X_test_s = x_scaler.transform(X_test)

    return X_train_s, X_test_s, feature_names


def build_predict_fn_complex(model, device):
    def predict_fn(X_numpy):
        X_t = torch.from_numpy(X_numpy).float().to(device)
        # Embed real features with zero imaginary components.
        X_c = X_t.to(torch.complex64)
        with torch.no_grad():
            y = model(X_c).detach().cpu().numpy().ravel()
        return y

    return predict_fn


def get_interaction_matrix(model, X_train, X_test, n_bg=100, n_eval=200):
    device = next(model.parameters()).device
    predict_fn = build_predict_fn_complex(model, device)

    X_bg = X_train[np.random.choice(X_train.shape[0], min(len(X_train), n_bg), replace=False)]
    X_eval = X_test[np.random.choice(X_test.shape[0], min(len(X_test), n_eval), replace=False)]

    print(f"    -> Running PermutationExplainer...")
    explainer = shap.PermutationExplainer(predict_fn, X_bg)
    shap_values = explainer(X_eval, silent=True)

    phi = shap_values.values if hasattr(shap_values, "values") else np.array(shap_values)
    n_features = phi.shape[1]
    matrix = np.zeros((n_features, n_features))

    for i in range(n_features):
        ranking = shap.utils.approximate_interactions(i, phi, X_eval)
        for rank, j in enumerate(ranking):
            score = n_features - rank
            matrix[i, j] = score

    # Symmetrize and normalize the ranking scores.
    matrix = (matrix + matrix.T) / 2.0
    if matrix.max() > 0:
        matrix /= matrix.max()
    return matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="./checkpoints_complex_test")
    parser.add_argument("--seed", type=int, default=42)

    # Checkpoint names include the hidden width.
    parser.add_argument("--hidden_dim", type=int, default=64)

    parser.add_argument("--n_bg", type=int, default=100)
    parser.add_argument("--n_eval", type=int, default=200)

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets = ["friedman", "concrete", "california"]

    # Tags must match the checkpoint filenames.
    models_config = [
        ("Complex MLP (Baseline)", "complex_mlp", f"complex_mlp_h{args.hidden_dim}"),
        ("Complex PU-ResNet (Ours)", "complex_pu", f"complex_pu_h{args.hidden_dim}"),
    ]

    for ds in datasets:
        print(f"\n{'=' * 50}\nProcessing Dataset: {ds.upper()}\n{'=' * 50}")

        try:
            X_train_s, X_test_s, feature_names = load_data(ds)
        except Exception as e:
            print(f"Error loading {ds}: {e}")
            continue

        input_dim = X_train_s.shape[1]
        matrices = []

        for label, m_type, m_tag in models_config:
            ckpt_path = os.path.join(args.save_dir, f"{m_tag}_{ds}_run{args.run_id}_best.pth")
            print(f"  - Model: {label}")

            if not os.path.exists(ckpt_path):
                print(f"    [Skip] Checkpoint not found: {ckpt_path}")
                matrices.append(None)
                continue

            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

            if m_type == "complex_mlp":
                model = build_complex_mlp(input_dim=input_dim, output_dim=1, hidden_dim=args.hidden_dim)
            else:
                model = ComplexPUResNet1D(input_dim=input_dim, hidden_dim=args.hidden_dim, output_dim=1)

            model.load_state_dict(ckpt["model_state_dict"])
            model.to(device).eval()

            mat = get_interaction_matrix(
                model, X_train_s, X_test_s,
                n_bg=args.n_bg, n_eval=args.n_eval
            )
            matrices.append(mat)

        if any(m is None for m in matrices):
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, matrix, (label, _, _) in zip(axes, matrices, models_config):
            im = ax.imshow(matrix, cmap="viridis", origin="lower", vmin=0, vmax=1)
            ax.set_title(label, fontsize=14, fontweight="bold")
            ax.set_xticks(range(len(feature_names)))
            ax.set_yticks(range(len(feature_names)))
            ax.set_xticklabels(feature_names, rotation=45, ha="right")
            ax.set_yticklabels(feature_names)

        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax, label="Normalized Interaction Strength")

        plt.suptitle(f"Complex SHAP Feature Interaction Comparison: {ds.capitalize()}", fontsize=16, y=1.02)

        save_path = os.path.join(args.save_dir, f"shap_compare_complex_{ds}_run{args.run_id}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"\nDone! Figure saved to: {save_path}")

    print("\nAll complex SHAP comparisons completed.")


if __name__ == "__main__":
    main()
