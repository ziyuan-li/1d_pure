# 03_shap_viz.py
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.datasets import make_friedman1, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import shap

from mlp_models import build_4layer_mlp
from pu_models import RealPUResNet1D


# ========= 1. 数据加载 =========

def load_friedman():
    X, y = make_friedman1(n_samples=10000, n_features=10, noise=0.0, random_state=0)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=0
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=0
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def load_concrete():
    df = pd.read_excel("./data/concrete+compressive+strength/Concrete_Data.xls")
    X = df.iloc[:, :8].values
    y = df.iloc[:, 8].values
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=0
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=0
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def load_california():
    data = fetch_california_housing()
    X, y = data.data, data.target
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.20, random_state=0
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=0
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def standardize_splits(X_train, X_val, X_test, y_train, y_val, y_test):
    x_scaler = StandardScaler()
    X_train_s = x_scaler.fit_transform(X_train)
    X_val_s = x_scaler.transform(X_val)
    X_test_s = x_scaler.transform(X_test)
    y_scaler = StandardScaler()
    y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_s = y_scaler.transform(y_val.reshape(-1, 1)).ravel()
    y_test_s = y_scaler.transform(y_test.reshape(-1, 1)).ravel()
    return (X_train_s, X_val_s, X_test_s, y_train_s, y_val_s, y_test_s, x_scaler, y_scaler)


# ========= 2. SHAP 交互矩阵计算 =========

def build_predict_fn(model, device):
    def predict_fn(X_numpy):
        X_t = torch.from_numpy(X_numpy).float().to(device)
        with torch.no_grad():
            y = model(X_t).cpu().numpy().ravel()
        return y

    return predict_fn


def compute_shap_interaction_matrix(model, X_train, X_test, max_background=100, eval_samples=300):
    device = next(model.parameters()).device
    predict_fn = build_predict_fn(model, device)

    # 注意：这里的随机选择受全局 seed 控制，所以结果是固定的
    if X_train.shape[0] > max_background:
        bg_idx = np.random.choice(X_train.shape[0], max_background, replace=False)
        X_bg = X_train[bg_idx]
    else:
        X_bg = X_train

    if X_test.shape[0] > eval_samples:
        eval_idx = np.random.choice(X_test.shape[0], eval_samples, replace=False)
        X_eval = X_test[eval_idx]
    else:
        X_eval = X_test

    print(f"SHAP sampling: Background={X_bg.shape[0]}, Eval={X_eval.shape[0]}")

    explainer = shap.PermutationExplainer(predict_fn, X_bg)
    shap_exp = explainer(X_eval)

    if hasattr(shap_exp, "values"):
        phi = shap_exp.values
    else:
        phi = np.array(shap_exp)

    n_features = phi.shape[1]
    interaction_matrix = np.zeros((n_features, n_features))

    for i in range(n_features):
        ranking = shap.utils.approximate_interactions(i, phi, X_eval)
        scores = np.zeros(n_features)
        for rank, j in enumerate(ranking):
            scores[j] = (n_features - rank)
        interaction_matrix[i] = scores

    interaction_matrix = (interaction_matrix + interaction_matrix.T) / 2.0
    max_val = interaction_matrix.max()
    if max_val > 0:
        interaction_matrix = interaction_matrix / max_val

    return interaction_matrix


def plot_interaction_heatmap(interaction_matrix, feature_names=None, title="SHAP Interaction", save_path=None):
    n_features = interaction_matrix.shape[0]
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(n_features)]

    plt.figure(figsize=(8, 6))
    im = plt.imshow(interaction_matrix, cmap="viridis", origin="lower")
    plt.colorbar(im, label="Normalized interaction strength")

    plt.xticks(ticks=np.arange(n_features), labels=feature_names, rotation=45, ha="right")
    plt.yticks(ticks=np.arange(n_features), labels=feature_names)
    plt.title(title)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)
        print(f"Saved interaction heatmap to: {save_path}")
    plt.show()


# ========= 3. 主函数 =========

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="concrete", choices=["friedman", "concrete", "california"])
    parser.add_argument("--model", type=str, default="mlp", choices=["mlp", "pu"])
    parser.add_argument("--run_id", type=int, default=0, help="要分析第几个run的模型")
    parser.add_argument("--seed", type=int, default=42, help="固定随机种子以保证SHAP采样一致")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    # === 关键：设置全局种子 ===
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f"Random seed set to: {args.seed}")

    # 加载数据
    if args.dataset == "friedman":
        X_train, X_val, X_test, y_train, y_val, y_test = load_friedman()
        feature_names = [f"x{i}" for i in range(10)]
    elif args.dataset == "concrete":
        X_train, X_val, X_test, y_train, y_val, y_test = load_concrete()
        feature_names = [f"x{i}" for i in range(8)]
    else:
        X_train, X_val, X_test, y_train, y_val, y_test = load_california()
        feature_names = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Lat", "Lon"]

    (X_train_s, X_val_s, X_test_s, _, _, _, _, _) = standardize_splits(
        X_train, X_val, X_test, y_train, y_val, y_test
    )

    input_dim = X_train_s.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 确定模型 Tag
    if args.model == "mlp":
        model_tag = "mlp4_h128"
    else:
        model_tag = "puresnet_h128"

    ckpt_path = os.path.join(args.save_dir, f"{model_tag}_{args.dataset}_run{args.run_id}_best.pth")

    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found at {ckpt_path}")
        return

    print(f"Loading checkpoint: {ckpt_path}")
    # 添加 weights_only=False 以兼容 numpy 数据
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    hidden_dim = ckpt["hidden_dim"]
    state_dict = ckpt["model_state_dict"]

    if args.model == "mlp":
        model = build_4layer_mlp(input_dim, 1, hidden_dim)
    else:
        model = RealPUResNet1D(input_dim, hidden_dim, 1)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 计算矩阵
    interaction_matrix = compute_shap_interaction_matrix(
        model, X_train_s, X_test_s, max_background=100, eval_samples=300
    )

    # 画图
    title = f"SHAP Interaction ({args.dataset.capitalize()}, {args.model.upper()}, Run {args.run_id})"
    save_path = os.path.join(args.save_dir,
                             f"shap_heatmap_{args.dataset}_{args.model}_run{args.run_id}_seed{args.seed}.png")
    plot_interaction_heatmap(interaction_matrix, feature_names, title, save_path)


if __name__ == "__main__":
    main()