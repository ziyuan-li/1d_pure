# 09_pareto_plot.py
import os
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_friedman1, fetch_california_housing

from mlp_models import build_4layer_mlp
from pu_models import RealPUResNet1D


# ==========================================
# 工具函数：数据与延迟测量
# ==========================================
def load_test_data(dataset_name):
    """只加载 Test Set 用于评估 MSE"""
    if dataset_name == "friedman":
        X, y = make_friedman1(n_samples=10000, n_features=10, noise=0.0, random_state=0)
    elif dataset_name == "concrete":
        df = pd.read_excel("./data/concrete+compressive+strength/Concrete_Data.xls")
        X = df.iloc[:, :8].values
        y = df.iloc[:, 8].values
    else:
        data = fetch_california_housing()
        X, y = data.data, data.target

    # 保持和训练一致的 Split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=0)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=0)

    # 标准化
    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))

    X_test_s = x_scaler.transform(X_test)
    y_test_s = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

    return torch.tensor(X_test_s).float(), torch.tensor(y_test_s).float().unsqueeze(1)


def measure_latency(model, input_dim, device, n_warmup=20, n_repeat=500):
    dummy = torch.randn(1, input_dim).to(device)
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy)

    if device.type == 'cuda': torch.cuda.synchronize()

    # Timing
    start = time.time()
    with torch.no_grad():
        for _ in range(n_repeat):
            _ = model(dummy)
            if device.type == 'cuda': torch.cuda.synchronize()
    end = time.time()

    return ((end - start) / n_repeat) * 1000  # ms


def evaluate_mse(model, X_test, y_test, device):
    model.eval()
    X_test, y_test = X_test.to(device), y_test.to(device)
    with torch.no_grad():
        pred = model(X_test)
        loss = nn.MSELoss()(pred, y_test)
    return loss.item()


# ==========================================
# 主程序
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="friedman")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--runs", type=int, default=5, help="取前N个run的平均值")
    args = parser.parse_args()

    device = torch.device("cpu")  # 测速建议用 CPU 单核性能或指定 GPU，这里默认 CPU 方便复现
    print(f"Benchmarking on {device}...")

    # 1. 加载数据
    X_test_t, y_test_t = load_test_data(args.dataset)
    input_dim = X_test_t.shape[1]

    # 定义要对比的模型
    models_config = [
        {"name": "MLP (4-Layer)", "type": "mlp", "tag": "mlp4_h128", "color": "tab:blue", "marker": "o"},
        {"name": "PU-ResNet (Ours)", "type": "pu", "tag": "puresnet_h128", "color": "tab:red", "marker": "s"},
    ]

    results = []

    # 2. 遍历模型并收集数据
    for cfg in models_config:
        latencies = []
        mses = []

        print(f"\nEvaluating {cfg['name']}...")
        for run in range(args.runs):
            ckpt_path = os.path.join(args.save_dir, f"{cfg['tag']}_{args.dataset}_run{run}_best.pth")
            if not os.path.exists(ckpt_path):
                continue

            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

            if cfg['type'] == 'mlp':
                model = build_4layer_mlp(input_dim, 1, ckpt['hidden_dim'])
            else:
                model = RealPUResNet1D(input_dim, ckpt['hidden_dim'], 1)

            model.load_state_dict(ckpt['model_state_dict'])
            model.to(device)

            # 测速
            lat = measure_latency(model, input_dim, device)
            # 测精度
            mse = evaluate_mse(model, X_test_t, y_test_t, device)

            latencies.append(lat)
            mses.append(mse)
            print(f"  Run {run}: Latency={lat:.3f}ms, MSE={mse:.4f}")

        if latencies:
            cfg['avg_lat'] = np.mean(latencies)
            cfg['avg_mse'] = np.mean(mses)
            cfg['std_lat'] = np.std(latencies)
            cfg['std_mse'] = np.std(mses)
            results.append(cfg)

    # 3. 绘制 Pareto 图
    plt.figure(figsize=(8, 6))

    for res in results:
        plt.errorbar(
            res['avg_lat'], res['avg_mse'],
            xerr=res['std_lat'], yerr=res['std_mse'],
            fmt=res['marker'], color=res['color'], label=res['name'],
            capsize=5, markersize=10, elinewidth=2
        )
        # 添加文字标注
        plt.text(
            res['avg_lat'], res['avg_mse'] * 1.05,
            f"{res['name']}\n({res['avg_mse']:.4f})",
            ha='center', va='bottom', fontsize=10, fontweight='bold', color=res['color']
        )

    plt.xlabel(f"Inference Latency (ms) on {device.type.upper()} [Lower is better]", fontsize=12, fontweight='bold')
    plt.ylabel("Test MSE (Standardized) [Lower is better]", fontsize=12, fontweight='bold')
    plt.title(f"Accuracy vs. Efficiency Trade-off ({args.dataset.capitalize()})", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=11)

    # 稍微调整坐标轴范围，让点不要贴边
    ylim = plt.ylim()
    plt.ylim(ylim[0] * 0.9, ylim[1] * 1.2)

    save_path = os.path.join(args.save_dir, f"pareto_plot_{args.dataset}.png")
    plt.savefig(save_path, dpi=300)
    print(f"\nPlot saved to: {save_path}")
    plt.show()


if __name__ == "__main__":
    main()