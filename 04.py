import os
import argparse
import numpy as np
import pandas as pd
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import shap

from mlp_models import build_4layer_mlp
from pu_models import RealPUResNet1D


# ==========================================
# 1. 数据加载与预处理 (与训练保持一致)
# ==========================================
def load_friedman_data():
    """
    Friedman1 Ground Truth:
    y(x) = 10 * sin(pi * x0 * x1) + 20 * (x2 - 0.5)**2 + 10 * x3 + 5 * x4 + noise

    关键交互对: (x0, x1) -> Index 0 和 1
    """
    X, y = make_friedman1(
        n_samples=10000,
        n_features=10,
        noise=0.0,
        random_state=0
    )
    # 划分 (与 train_mlp.py 一致)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=0
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=0
    )

    # 标准化
    x_scaler = StandardScaler()
    X_train_s = x_scaler.fit_transform(X_train)
    X_test_s = x_scaler.transform(X_test)

    return X_train_s, X_test_s


# ==========================================
# 2. SHAP 交互矩阵计算核心逻辑
# ==========================================
def build_predict_fn(model, device):
    def predict_fn(X_numpy):
        X_t = torch.from_numpy(X_numpy).float().to(device)
        with torch.no_grad():
            y = model(X_t).cpu().numpy().ravel()
        return y

    return predict_fn


def get_interaction_matrix(model, X_train, X_test, n_bg=100, n_eval=300):
    """
    计算基于排名的交互矩阵。
    """
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

    # 计算 SHAP values
    explainer = shap.PermutationExplainer(predict_fn, X_bg)
    shap_obj = explainer(X_eval)
    if hasattr(shap_obj, "values"):
        phi = shap_obj.values
    else:
        phi = np.array(shap_obj)

    n_features = phi.shape[1]
    matrix = np.zeros((n_features, n_features))

    # 使用 approximate_interactions 获取交互排名
    for i in range(n_features):
        # ranking 是一个索引列表，排在前面表示与特征 i 交互最强
        ranking = shap.utils.approximate_interactions(i, phi, X_eval)

        # 将排名转换为分数: 第一名得 n分, 最后一名得 1分
        for rank, j in enumerate(ranking):
            score = n_features - rank
            matrix[i, j] = score

    # 对称化
    matrix = (matrix + matrix.T) / 2.0

    # 归一化到 [0, 1]
    if matrix.max() > 0:
        matrix /= matrix.max()

    return matrix


# ==========================================
# 3. 针对 Friedman1 的特定度量
# ==========================================
def calculate_friedman_metrics(matrix):
    """
    针对 Friedman1: 检查 (x0, x1) 是否是模型认为的最强交互。
    """
    n = matrix.shape[0]

    # 提取 (0, 1) 的交互强度
    target_score = matrix[0, 1]

    # 提取所有非 (0, 1) 且非对角线的交互强度
    other_scores = []
    for i in range(n):
        for j in range(i + 1, n):
            if (i == 0 and j == 1):
                continue
            other_scores.append(matrix[i, j])

    max_other = max(other_scores) if other_scores else 0.0
    mean_other = np.mean(other_scores) if other_scores else 0.0

    # 1. 计算 (0, 1) 在所有 pair 中的排名 (1为最高)
    # 把所有 pair 的分数放一起排序
    all_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            all_pairs.append(((i, j), matrix[i, j]))

    # 按分数降序排列
    all_pairs.sort(key=lambda x: x[1], reverse=True)

    rank = -1
    for r, (pair, score) in enumerate(all_pairs):
        if pair == (0, 1):
            rank = r + 1  # 1-based rank
            break

    # 2. 计算信噪比 Ratio: Target / Max_Other
    ratio = target_score / (max_other + 1e-6)

    return target_score, rank, ratio


# ==========================================
# 4. 主流程
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    # 这里的参数需要跟训练时的 checkpoint 匹配
    # Friedman1 默认 10 features
    input_dim = 10

    # 准备数据
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
            ckpt_path = os.path.join(
                args.save_dir,
                f"{model_tag}_friedman_run{run}_best.pth"
            )

            if not os.path.exists(ckpt_path):
                print(f"Run {run}: Checkpoint not found, skipping.")
                continue

            # 加载模型
            ckpt = torch.load(ckpt_path, map_location="cpu")
            hidden_dim = ckpt["hidden_dim"]

            if model_type == "mlp":
                model = build_4layer_mlp(input_dim, 1, hidden_dim)
            else:
                model = RealPUResNet1D(input_dim, hidden_dim, 1)

            model.load_state_dict(ckpt["model_state_dict"])
            model.to(device)
            model.eval()

            # 计算矩阵
            matrix = get_interaction_matrix(model, X_train_s, X_test_s)

            # 计算指标
            score, rank, ratio = calculate_friedman_metrics(matrix)

            results[model_type].append({
                "run": run,
                "rank": rank,
                "ratio": ratio,
                "score": score
            })

            print(f"Run {run}: Pair(x0,x1) Rank = {rank:>2d}/45 | Strength Ratio = {ratio:.2f}x (vs next best)")

    # 统计汇总
    print("\n==== Final Summary ====")
    for m in ["mlp", "pu"]:
        data = results[m]
        if not data:
            continue

        ranks = [d["rank"] for d in data]
        ratios = [d["ratio"] for d in data]

        avg_rank = np.mean(ranks)
        avg_ratio = np.mean(ratios)

        # 计算完美发现交互的比例 (Rank == 1 的次数)
        perfect_count = sum(1 for r in ranks if r == 1)
        perfect_rate = perfect_count / len(ranks) * 100

        print(f"[{m.upper()}] over {len(data)} runs:")
        print(f"  > Avg Rank of (x0,x1): {avg_rank:.1f} (Lower is better, ideal=1.0)")
        print(f"  > Perfect Detection Rate: {perfect_rate:.0f}% (Rank=1)")
        print(f"  > Interaction Signal-to-Noise Ratio: {avg_ratio:.2f} (Higher is better)")
        print(f"  > Raw Ranks: {ranks}")


if __name__ == "__main__":
    main()