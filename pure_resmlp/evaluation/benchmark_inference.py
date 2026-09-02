"""Rebenchmark JoCS scientific-model inference with warm-up and CUDA sync."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from pure_resmlp.evaluation.run_interactions import _load_data
from pure_resmlp.paths import RUNS_DIR
from pure_resmlp.training.run_experiment import (
    build_torch_model,
    predict_torch,
    resolve_device,
    transform_input_matrix,
)


EXPERIMENT = "jocs_benchmark_v1"
DATASETS = ("friedman", "yacht", "auto_mpg")
MODELS = ("mlp", "resmlp", "rv_pure", "cv_pure")
TRAINING_SEEDS = tuple(range(5))


def summarize_timings(raw: pd.DataFrame) -> pd.DataFrame:
    """Summarize per-checkpoint median latency across training seeds."""
    records: list[dict[str, Any]] = []
    for (dataset, model), values in raw.groupby(["dataset", "model"], sort=True):
        latency = values["inference_ms_median"].to_numpy(dtype=np.float64)
        records.append(
            {
                "dataset": dataset,
                "model": model,
                "training_seeds": int(values["training_seed"].nunique()),
                "test_samples": int(values["test_samples"].iloc[0]),
                "inference_ms_mean": float(latency.mean()),
                "inference_ms_std": float(latency.std(ddof=1)),
            }
        )
    return pd.DataFrame.from_records(records).sort_values(["dataset", "model"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=RUNS_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/reports/jocs_v1/inference_efficiency"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--warmup-repeats", type=int, default=20)
    parser.add_argument("--warmup-seconds", type=float, default=0.5)
    parser.add_argument("--timed-repeats", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if min(args.batch_size, args.warmup_repeats, args.timed_repeats) <= 0:
        parser.error("batch size and repeat counts must be positive")
    if args.warmup_seconds < 0:
        parser.error("warmup-seconds must be non-negative")

    raw_path = args.output_dir / "inference_timings_raw.csv"
    summary_path = args.output_dir / "inference_timings_summary.csv"
    protocol_path = args.output_dir / "protocol.json"
    if not args.overwrite and any(
        path.exists() for path in (raw_path, summary_path, protocol_path)
    ):
        parser.error("timing outputs already exist; pass --overwrite to replace them")

    device = resolve_device(args.device)
    if device.type != "cuda":
        parser.error("publication inference timing requires a CUDA device")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)

    records: list[dict[str, Any]] = []
    data_cache: dict[str, Any] = {}
    for dataset in DATASETS:
        for training_seed in TRAINING_SEEDS:
            for model_name in MODELS:
                run_dir = (
                    args.runs_root
                    / EXPERIMENT
                    / dataset
                    / model_name
                    / f"seed_{training_seed}"
                )
                config = json.loads(
                    (run_dir / "config.json").read_text(encoding="utf-8")
                )
                if dataset not in data_cache:
                    data_cache[dataset] = _load_data(config)
                data = data_cache[dataset]
                parameters = config["parameters"]
                model = build_torch_model(
                    model_name,
                    input_dim=int(parameters["input_dim"]),
                    hidden_dim=int(parameters["hidden_dim"]),
                ).to(device)
                checkpoint = torch.load(
                    run_dir / "checkpoint.pt", map_location=device, weights_only=True
                )
                model.load_state_dict(checkpoint["model_state_dict"])
                model.eval()
                transformed, _ = transform_input_matrix(data, model_name, data.X_test)
                complex_input = model_name == "cv_pure"

                warmup_start = time.perf_counter()
                warmup_count = 0
                while (
                    warmup_count < args.warmup_repeats
                    or time.perf_counter() - warmup_start < args.warmup_seconds
                ):
                    predict_torch(
                        model,
                        transformed,
                        device=device,
                        batch_size=args.batch_size,
                        complex_input=complex_input,
                    )
                    warmup_count += 1
                torch.cuda.synchronize(device)

                elapsed_ms: list[float] = []
                for _ in range(args.timed_repeats):
                    torch.cuda.synchronize(device)
                    start_ns = time.perf_counter_ns()
                    predict_torch(
                        model,
                        transformed,
                        device=device,
                        batch_size=args.batch_size,
                        complex_input=complex_input,
                    )
                    torch.cuda.synchronize(device)
                    elapsed_ms.append((time.perf_counter_ns() - start_ns) / 1_000_000)

                records.append(
                    {
                        "experiment": EXPERIMENT,
                        "dataset": dataset,
                        "model": model_name,
                        "training_seed": training_seed,
                        "test_samples": len(transformed),
                        "batch_size": args.batch_size,
                        "warmup_repeats": warmup_count,
                        "minimum_warmup_seconds": args.warmup_seconds,
                        "timed_repeats": args.timed_repeats,
                        "inference_ms_median": float(np.median(elapsed_ms)),
                        "inference_ms_mean": float(np.mean(elapsed_ms)),
                        "inference_ms_std": float(np.std(elapsed_ms, ddof=1)),
                    }
                )
                print(
                    f"timed {dataset}/{model_name}/seed_{training_seed}",
                    flush=True,
                )
                del model, checkpoint

    raw = pd.DataFrame.from_records(records)
    if len(raw) != len(DATASETS) * len(MODELS) * len(TRAINING_SEEDS):
        raise RuntimeError(f"unexpected timing row count: {len(raw)}")
    summary = summarize_timings(raw)
    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    protocol_path.write_text(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(device),
                "device_index": int(device.index or 0),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "batch_size": args.batch_size,
                "minimum_warmup_repeats": args.warmup_repeats,
                "minimum_warmup_seconds": args.warmup_seconds,
                "timed_repeats": args.timed_repeats,
                "timing_unit": "milliseconds per complete test split",
                "checkpoint_statistic": "median over timed repeats",
                "reported_statistic": "mean and sample SD over five training seeds",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
