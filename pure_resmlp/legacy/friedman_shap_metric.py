# 04_friedman_metric.py
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import shap

from pure_resmlp.models.mlp import build_4layer_mlp
from pure_resmlp.models.pure import RealPUResNet1D


# ==========================================
# 1. 数据加载
# ==========================================
def load_friedman_data():
    X, y = make_friedman1(n_samples=10000, n_features=10, noise=0.0, random_state=0)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=0
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=0
    )
    x_scaler = StandardScaler()
    X_train_s = x_scaler.fit_transform(X_train)
    X_test_s = x_scaler.transform(X_test)
    return X_train_s, X_test_s


# ==========================================
# 2. SHAP 计算逻辑
# ==========================================
def build_predict_fn(model, device):
    def predict_fn(X_numpy):
        X_t = torch.from_numpy(X_numpy).float().to(device)
        with torch.no_grad():
            y = model(X_t).cpu().numpy().ravel()
        return y

    return predict_fn


def get_interaction_matrix(model, X_train, X_test, n_bg=100, n_eval=300):
    device = next(model.parameters()).device
    predict_fn = build_predict_fn(model, device)

    # 这里的随机选择现在受到 main 函数里 seed 的控制
    if X_train.shape[0] > n_bg:
        idx_bg = np.random.choice(X_train.shape[0], n_bg, replace=False)
        X_bg = X_train[idx_bg]
    else:
        X_bg = X_train

    if X_test.shape[0] > n_eval:
        idx_eval = np.random.choice(X_test.shape[0], n_eval, replace=False)
        X_eval = X_test[idx_eval]
    else:
        X_eval = X_test

    explainer = shap.PermutationExplainer(predict_fn, X_bg)
    shap_obj = explainer(X_eval)

    if hasattr(shap_obj, "values"):
        phi = shap_obj.values
    else:
        phi = np.array(shap_obj)

    n_features = phi.shape[1]
    matrix = np.zeros((n_features, n_features))

    for i in range(n_features):
        ranking = shap.utils.approximate_interactions(i, phi, X_eval)
        for rank, j in enumerate(ranking):
            score = n_features - rank
            matrix[i, j] = score

    matrix = (matrix + matrix.T) / 2.0
    if matrix.max() > 0:
        matrix /= matrix.max()

    return matrix


def calculate_friedman_metrics(matrix):
    n = matrix.shape[0]
    target_score = matrix[0, 1]  # Friedman: x0-x1 is the key

    other_scores = []
    for i in range(n):
        for j in range(i + 1, n):
            if (i == 0 and j == 1):
                continue
            other_scores.append(matrix[i, j])

    max_other = max(other_scores) if other_scores else 0.0

    # 排名
    all_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            all_pairs.append(((i, j), matrix[i, j]))

    all_pairs.sort(key=lambda x: x[1], reverse=True)

    rank = -1
    for r, (pair, score) in enumerate(all_pairs):
        if pair == (0, 1):
            rank = r + 1
            break

    ratio = target_score / (max_other + 1e-6)
    return target_score, rank, ratio


# ==========================================
# 3. 主流程
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="./checkpoints_test_ini0")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17, help="固定随机种子以保证SHAP采样一致")
    args = parser.parse_args()

    # === 关键：设置全局种子 ===
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f"Random seed set to: {args.seed}")

    input_dim = 10  # Friedman
    X_train_s, X_test_s = load_friedman_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {"mlp": [], "pu": []}

    print("==== 开始 Friedman1 交互验证 ====")
    print(f"Ground Truth Interaction: (x0, x1)\n")

    for model_type in ["mlp", "pu"]:
        if model_type == "mlp":
            model_tag = "mlp4_h128"
        else:
            model_tag = "puresnet_h128"

        print(f"--- Analyzing Model: {model_type} ---")

        for run in range(args.runs):
            ckpt_path = os.path.join(args.save_dir, f"{model_tag}_friedman_run{run}_best.pth")

            if not os.path.exists(ckpt_path):
                print(f"Run {run}: Checkpoint not found, skipping.")
                continue

            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            hidden_dim = ckpt["hidden_dim"]

            if model_type == "mlp":
                model = build_4layer_mlp(input_dim, 1, hidden_dim)
            else:
                model = RealPUResNet1D(input_dim, hidden_dim, 1)

            model.load_state_dict(ckpt["model_state_dict"])
            model.to(device)
            model.eval()

            matrix = get_interaction_matrix(model, X_train_s, X_test_s)
            score, rank, ratio = calculate_friedman_metrics(matrix)

            results[model_type].append({
                "run": run,
                "rank": rank,
                "ratio": ratio,
                "score": score
            })

            print(f"Run {run}: Pair(x0,x1) Rank = {rank:>2d}/45 | Ratio = {ratio:.2f}x")

    print("\n==== Final Summary ====")
    for m in ["mlp", "pu"]:
        data = results[m]
        if not data:
            continue

        ranks = [d["rank"] for d in data]
        ratios = [d["ratio"] for d in data]

        avg_rank = np.mean(ranks)
        avg_ratio = np.mean(ratios)
        perfect_rate = sum(1 for r in ranks if r == 1) / len(ranks) * 100

        print(
            f"[{m.upper()}] Avg Rank: {avg_rank:.1f} | Signal-Noise Ratio: {avg_ratio:.2f} | Perfect Rate: {perfect_rate:.0f}%")


if __name__ == "__main__":
    main()
