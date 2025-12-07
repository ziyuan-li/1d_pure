# 03_shap_interaction.py
import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_friedman1, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import shap

from mlp_models import build_4layer_mlp
from pu_models import RealPUResNet1D


# ========= 1. 数据加载（跟你训练脚本保持一致） =========

def load_friedman():
    X, y = make_friedman1(
        n_samples=10000,
        n_features=10,
        noise=0.0,
        random_state=0
    )
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=0
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=0
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def load_concrete():
    df = pd.read_excel(
        "./data/concrete+compressive+strength/Concrete_Data.xls"
    )
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

    return (X_train_s, X_val_s, X_test_s,
            y_train_s, y_val_s, y_test_s,
            x_scaler, y_scaler)


# ========= 2. SHAP 交互矩阵计算 =========

def build_predict_fn(model, device):
    """
    把 PyTorch 模型封装成 numpy -> numpy 的预测函数，给 SHAP 用
    """
    def predict_fn(X_numpy: np.ndarray):
        X_t = torch.from_numpy(X_numpy).float().to(device)
        with torch.no_grad():
            y = model(X_t).cpu().numpy().ravel()
        return y
    return predict_fn


def compute_shap_interaction_matrix(model, X_train, X_test,
                                    max_background=100,
                                    eval_samples=300):
    """
    使用 PermutationExplainer + shap.utils.approximate_interactions
    估计全局特征交互矩阵。

    返回: interaction_matrix (n_features, n_features)
    """
    device = next(model.parameters()).device
    predict_fn = build_predict_fn(model, device)

    # 1) 选 background（太大会很慢，这里截断到 max_background）
    if X_train.shape[0] > max_background:
        bg_idx = np.random.choice(X_train.shape[0], max_background, replace=False)
        X_bg = X_train[bg_idx]
    else:
        X_bg = X_train

    # 2) 选一部分测试样本做 SHAP 计算
    if X_test.shape[0] > eval_samples:
        eval_idx = np.random.choice(X_test.shape[0], eval_samples, replace=False)
        X_eval = X_test[eval_idx]
    else:
        X_eval = X_test

    print(f"Background size: {X_bg.shape[0]}, "
          f"evaluation samples: {X_eval.shape[0]}")

    # 3) 构建 PermutationExplainer（比 KernelExplainer 稍快，也更简单）
    explainer = shap.PermutationExplainer(predict_fn, X_bg)

    # 4) 计算 SHAP values
    shap_exp = explainer(X_eval)   # 新版本返回 Explanation 对象
    if hasattr(shap_exp, "values"):
        phi = shap_exp.values      # (n_samples, n_features)
    else:
        # 兼容老版本（直接返回 ndarray）
        phi = np.array(shap_exp)

    n_features = phi.shape[1]
    interaction_matrix = np.zeros((n_features, n_features))

    # 5) 对每个特征 i，用 approximate_interactions(i, φ, X_eval)
    #    得到与其它特征的交互强度排序，用来构建交互矩阵
    for i in range(n_features):
        # 返回的是一个按交互强度排序的特征索引列表
        ranking = shap.utils.approximate_interactions(i, phi, X_eval)
        # 我们把排名转成分数（排在前面的交互权重大）
        scores = np.zeros(n_features)
        for rank, j in enumerate(ranking):
            scores[j] = (n_features - rank)  # 越靠前权重越大
        interaction_matrix[i] = scores

    # 6) 对称化（i 与 j 的交互，取平均）
    interaction_matrix = (interaction_matrix + interaction_matrix.T) / 2.0

    # 7) 归一化到 [0,1]，方便画图
    max_val = interaction_matrix.max()
    if max_val > 0:
        interaction_matrix = interaction_matrix / max_val

    return interaction_matrix


def plot_interaction_heatmap(interaction_matrix, feature_names=None,
                             title="SHAP Interaction", save_path=None):
    n_features = interaction_matrix.shape[0]
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(n_features)]

    plt.figure(figsize=(6, 5))
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


# ========= 3. 主函数：选模型 + 数据集，画交互图 =========

def main():
    dataset = "california"   # "friedman" / "concrete" / "california"
    model_type = "mlp"      # "mlp" 或 "pu"
    run_id = 0             # 用哪个 run 的 checkpoint

    save_dir = "./checkpoints"

    # ===== 加载数据 =====
    if dataset == "friedman":
        X_train, X_val, X_test, y_train, y_val, y_test = load_friedman()
        feature_names = [f"x{i}" for i in range(10)]
    elif dataset == "concrete":
        X_train, X_val, X_test, y_train, y_val, y_test = load_concrete()
        feature_names = [f"x{i}" for i in range(8)]
    else:
        X_train, X_val, X_test, y_train, y_val, y_test = load_california()
        # California 有 8 个特征:
        feature_names = [
            "MedInc", "HouseAge", "AveRooms", "AveBedrms",
            "Population", "AveOccup", "Latitude", "Longitude"
        ]

    (X_train_s, X_val_s, X_test_s,
     y_train_s, y_val_s, y_test_s,
     x_scaler, y_scaler) = standardize_splits(
        X_train, X_val, X_test, y_train, y_val, y_test
    )

    input_dim = X_train_s.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== 加载模型 checkpoint =====
    if model_type == "mlp":
        model_tag = "mlp4_h128"
    else:
        model_tag = "puresnet_h128"

    ckpt_path = os.path.join(
        save_dir,
        f"{model_tag}_{dataset}_run{run_id}_best.pth"
    )
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    hidden_dim = ckpt["hidden_dim"]
    state_dict = ckpt["model_state_dict"]

    # 构建模型
    if model_type == "mlp":
        model = build_4layer_mlp(
            input_dim=input_dim,
            output_dim=1,
            hidden_dim=hidden_dim
        )
    else:
        model = RealPUResNet1D(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=1
        )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # ===== 计算 SHAP 交互矩阵 =====
    interaction_matrix = compute_shap_interaction_matrix(
        model,
        X_train_s,
        X_test_s,
        max_background=100,
        eval_samples=300,
    )

    # ===== 画图 =====
    title = f"SHAP Interaction ({dataset}, {model_type})"
    save_path = os.path.join(
        save_dir,
        f"shap_interaction_{dataset}_{model_type}.png"
    )
    plot_interaction_heatmap(interaction_matrix, feature_names, title, save_path)


if __name__ == "__main__":
    main()
