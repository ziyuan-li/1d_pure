# 08_shap_compare.py
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

# 假设模型定义在这些文件中
from mlp_models import build_4layer_mlp
from pu_models import RealPUResNet1D


# ==========================================
# 1. 数据加载 (复用之前的逻辑)
# ==========================================
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

    # 统一划分
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=0)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=0)

    # 标准化
    x_scaler = StandardScaler()
    X_train_s = x_scaler.fit_transform(X_train)
    X_test_s = x_scaler.transform(X_test)

    return X_train_s, X_test_s, feature_names


# ==========================================
# 2. SHAP 计算核心
# ==========================================
def build_predict_fn(model, device):
    def predict_fn(X_numpy):
        X_t = torch.from_numpy(X_numpy).float().to(device)
        with torch.no_grad():
            y = model(X_t).cpu().numpy().ravel()
        return y

    return predict_fn


def get_interaction_matrix(model, X_train, X_test, n_bg=100, n_eval=200):
    device = next(model.parameters()).device
    predict_fn = build_predict_fn(model, device)

    # 采样背景数据
    if X_train.shape[0] > n_bg:
        X_bg = X_train[np.random.choice(X_train.shape[0], n_bg, replace=False)]
    else:
        X_bg = X_train

    # 采样评估数据
    if X_test.shape[0] > n_eval:
        X_eval = X_test[np.random.choice(X_test.shape[0], n_eval, replace=False)]
    else:
        X_eval = X_test

    print(f"  -> Calculating SHAP values (Background: {len(X_bg)}, Eval: {len(X_eval)})...")
    explainer = shap.PermutationExplainer(predict_fn, X_bg)
    shap_values = explainer(X_eval)

    if hasattr(shap_values, "values"):
        phi = shap_values.values
    else:
        phi = np.array(shap_values)

    n_features = phi.shape[1]
    matrix = np.zeros((n_features, n_features))

    # 计算交互强度矩阵 (近似方法，速度较快)
    for i in range(n_features):
        ranking = shap.utils.approximate_interactions(i, phi, X_eval)
        for rank, j in enumerate(ranking):
            score = n_features - rank
            matrix[i, j] = score

    # 对称化并归一化
    matrix = (matrix + matrix.T) / 2.0
    if matrix.max() > 0:
        matrix /= matrix.max()

    return matrix


# ==========================================
# 3. 主程序：加载双模型并绘图
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="friedman")
    parser.add_argument("--run_id", type=int, default=0, help="使用哪个 Run 的模型")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 固定随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 1. 准备数据
    print(f"Loading data: {args.dataset}...")
    X_train_s, X_test_s, feature_names = load_data(args.dataset)
    input_dim = X_train_s.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models_config = [
        ("MLP (Baseline)", "mlp", "mlp4_h128"),
        ("PU-ResNet (Ours)", "pu", "puresnet_h128")
    ]

    matrices = []

    # 2. 计算两个模型的矩阵
    for label, m_type, m_tag in models_config:
        ckpt_path = os.path.join(args.save_dir, f"{m_tag}_{args.dataset}_run{args.run_id}_best.pth")
        print(f"\nProcessing {label}...")

        if not os.path.exists(ckpt_path):
            print(f"Warning: Checkpoint not found at {ckpt_path}. Skipping.")
            matrices.append(None)
            continue

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        hidden_dim = ckpt["hidden_dim"]

        if m_type == "mlp":
            model = build_4layer_mlp(input_dim, 1, hidden_dim)
        else:
            model = RealPUResNet1D(input_dim, hidden_dim, 1)

        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()

        mat = get_interaction_matrix(model, X_train_s, X_test_s)
        matrices.append(mat)

    # 3. 绘图 (Side-by-Side Heatmap)
    if any(m is None for m in matrices):
        print("Cannot plot because one or more models failed to load.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, matrix, (label, _, _) in zip(axes, matrices, models_config):
        im = ax.imshow(matrix, cmap="viridis", origin="lower", vmin=0, vmax=1)
        ax.set_title(label, fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(feature_names)))
        ax.set_yticks(range(len(feature_names)))
        ax.set_xticklabels(feature_names, rotation=45, ha='right')
        ax.set_yticklabels(feature_names)

        # 标注Friedman的关键交互 (x0, x1) 如果是 Friedman 数据集
        if args.dataset == "friedman":
            # 在 (0,1) 和 (1,0) 画红框
            import matplotlib.patches as patches
            rect1 = patches.Rectangle((0.5, -0.5), 1, 1, linewidth=2, edgecolor='r', facecolor='none')
            rect2 = patches.Rectangle((-0.5, 0.5), 1, 1, linewidth=2, edgecolor='r', facecolor='none')
            # 这里坐标可能需要微调，imshow 的 grid 是以整数为中心的
            # 更简单的做法是只在标题里提
            pass

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), label="Normalized Interaction Strength")

    plt.suptitle(f"Feature Interaction Analysis on {args.dataset.capitalize()} Dataset", fontsize=16)

    save_path = os.path.join(args.save_dir, f"shap_compare_{args.dataset}_run{args.run_id}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {save_path}")
    plt.show()


if __name__ == "__main__":
    main()