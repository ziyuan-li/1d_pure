import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from thop import profile, clever_format

from pure_resmlp.models.mlp import build_4layer_mlp
from pure_resmlp.models.pure import RealPUResNet1D


def measure_latency(model, input_tensor, n_warmup=50, n_repeat=1000):
    """Measure inference latency after warm-up."""
    device = input_tensor.device
    model.eval()

    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(input_tensor)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    start_time = time.time()
    with torch.no_grad():
        for _ in range(n_repeat):
            _ = model(input_tensor)
            if device.type == 'cuda':
                torch.cuda.synchronize()
    end_time = time.time()

    avg_time_ms = ((end_time - start_time) / n_repeat) * 1000
    return avg_time_ms


def measure_flops(model, input_tensor):
    """Estimate FLOPs and parameter count."""
    model.eval()
    macs, params = profile(model, inputs=(input_tensor,), verbose=False)
    macs_str, params_str = clever_format([macs, params], "%.3f")
    return macs, params, macs_str, params_str


def plot_results(results, save_dir, device_name):
    """Plot latency and parameter count on separate axes."""
    models = [r['Model'] for r in results]
    latencies = [r['Latency (ms)'] for r in results]
    params = [r['Params'] / 1000.0 for r in results]  # Parameters in thousands.

    x = np.arange(len(models))
    width = 0.5

    fig, ax1 = plt.subplots(figsize=(8, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Model Architecture', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Inference Latency (ms) [Lower is better]', color=color, fontsize=12)
    bars = ax1.bar(x, latencies, width, color=color, alpha=0.7, label='Latency')
    ax1.tick_params(axis='y', labelcolor=color)

    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.3f} ms',
                 ha='center', va='bottom', fontsize=11, fontweight='bold', color='black')

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Parameters (k) [Iso-parameter check]', color=color, fontsize=12)
    line = ax2.plot(x, params, color=color, marker='o', linestyle='--', linewidth=2, markersize=8, label='Params')
    ax2.tick_params(axis='y', labelcolor=color)

    min_p, max_p = min(params), max(params)
    ax2.set_ylim(min_p * 0.9, max_p * 1.1)

    for i, p in enumerate(params):
        ax2.text(x[i], p, f'{p:.1f}k', ha='center', va='bottom', color='red', fontsize=10, fontweight='bold')

    plt.title(f'Efficiency Benchmark on {device_name.upper()}', fontsize=14)
    plt.xticks(x, models, fontsize=11)

    fig.tight_layout()

    save_path = os.path.join(save_dir, "benchmark_latency_comparison.png")
    plt.savefig(save_path, dpi=300)
    print(f"\n[Plot] 图表已保存到: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="friedman", choices=["friedman", "concrete", "california"])
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--run_id", type=int, default=0, help="使用哪个 run 的模型进行测速")
    parser.add_argument("--batch_size", type=int, default=1, help="测速时的 Batch Size")
    parser.add_argument("--device", type=str, default="cpu", choices=["cuda", "cpu"])
    args = parser.parse_args()

    if args.dataset == "friedman":
        input_dim = 10
    else:
        input_dim = 8

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA不可用，切换到CPU...")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"\n==== Efficiency Benchmark [{args.dataset}] on {device} ====")

    dummy_input = torch.randn(args.batch_size, input_dim).to(device)

    models_config = [
        ("MLP", "mlp", "mlp4_h128"),
        ("PU-ResNet", "pu", "puresnet_h128")
    ]

    results = []

    for name, m_type, m_tag in models_config:
        ckpt_path = os.path.join(args.save_dir, f"{m_tag}_{args.dataset}_run{args.run_id}_best.pth")

        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu")
            hidden_dim = ckpt["hidden_dim"]
            print(f"\n>> Loading {name} from checkpoint...")
        else:
            print(f"\n>> Checkpoint for {name} not found, utilizing random init...")
            hidden_dim = 128

        if m_type == "mlp":
            model = build_4layer_mlp(input_dim, 1, hidden_dim)
        else:
            model = RealPUResNet1D(input_dim, hidden_dim, 1)

        model.to(device)

        flops, params, flops_str, params_str = measure_flops(model, dummy_input.clone())

        latency_ms = measure_latency(model, dummy_input)

        throughput = (1000 / latency_ms) * args.batch_size

        results.append({
            "Model": name,
            "Params": params,
            "FLOPs (MACs)": flops,
            "Latency (ms)": latency_ms,
            "Throughput (sample/s)": throughput
        })

        print(f"   Params: {params_str}")
        print(f"   Latency: {latency_ms:.4f} ms")

    print("\n==== Final Comparison Summary ====")
    print(f"{'Model':<15} | {'Params':<10} | {'Latency (ms)':<15} | {'Speedup':<15}")
    print("-" * 60)
    base_latency = results[0]["Latency (ms)"]
    for res in results:
        speedup = base_latency / res["Latency (ms)"]
        p_str = f"{res['Params'] / 1e3:.1f}k"
        print(f"{res['Model']:<15} | {p_str:<10} | {res['Latency (ms)']:<15.4f} | {speedup:.2f}x")

    plot_results(results, args.save_dir, args.device)


if __name__ == "__main__":
    main()
