"""Run scientific CV-MLP benchmarks with the existing split and training protocol."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from pure_resmlp.paths import ARTIFACTS_DIR, RUNS_DIR


EXPERIMENT = "jocs_scientific_cv_mlp_v1"
BASELINE_EXPERIMENT = "jocs_benchmark_v1"
DATASETS = ("friedman", "concrete", "california", "energy", "yacht", "auto_mpg")
SEEDS = tuple(range(5))
PARAMETERS = {
    "hidden_dim": 64,
    "epochs": 200,
    "patience": 20,
    "batch_size": 128,
    "learning_rate": 0.001,
    "output_head": "learnable_real_projection_v1",
}
REQUIRED = (
    "config.json", "environment.json", "split_indices.npz", "history.csv",
    "predictions.csv", "metrics.json", "checkpoint.pt", "manifest.json",
)


def jobs() -> list[tuple[str, int]]:
    return [(dataset, seed) for dataset in DATASETS for seed in SEEDS]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_path(root: Path, dataset: str, seed: int, model: str = "cv_mlp") -> Path:
    experiment = EXPERIMENT if model == "cv_mlp" else BASELINE_EXPERIMENT
    return root / experiment / dataset / model / f"seed_{seed}"


def check_config(config: dict, dataset: str, seed: int, model: str) -> None:
    experiment = EXPERIMENT if model == "cv_mlp" else BASELINE_EXPERIMENT
    expected = dict(experiment=experiment, dataset=dataset, model=model,
                    seed=seed, split_seed=0)
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"{dataset}/{model}/{seed}: unexpected {key}")
    for key, value in PARAMETERS.items():
        if config["parameters"].get(key) != value:
            raise ValueError(f"{dataset}/{model}/{seed}: unexpected {key}")
    if config["parameters"].get("train_fraction", 1.0) != 1.0:
        raise ValueError("Full training data are required")


def check_splits(path: Path, expected: dict[str, np.ndarray]) -> None:
    with np.load(path) as saved:
        if set(saved.files) != set(expected):
            raise ValueError(f"Split keys differ: {path}")
        for key, indices in expected.items():
            if not np.array_equal(saved[key], indices):
                raise ValueError(f"Split indices differ: {path}, {key}")


def preflight(root: Path) -> None:
    from pure_resmlp.data import prepare_benchmark

    for dataset in DATASETS:
        data = prepare_benchmark(dataset, split_seed=0, energy_target="Y1")
        for seed in SEEDS:
            baseline = run_path(root, dataset, seed, "cv_pure")
            check_config(read_json(baseline / "config.json"), dataset, seed, "cv_pure")
            check_splits(baseline / "split_indices.npz", data.split_indices)
            candidate = run_path(root, dataset, seed)
            if candidate.exists() and any(candidate.iterdir()):
                validate_run(root, dataset, seed)
        print(f"Preflight: {dataset}, all five baseline configs/splits match", flush=True)


def validate_run(root: Path, dataset: str, seed: int) -> dict:
    import torch

    path = run_path(root, dataset, seed)
    for name in REQUIRED:
        if not (path / name).is_file():
            raise RuntimeError(f"Incomplete run, refusing to overwrite: {path / name}")
    check_config(read_json(path / "config.json"), dataset, seed, "cv_mlp")
    if read_json(path / "manifest.json").get("status") != "complete":
        raise RuntimeError(f"Run is not complete: {path}")
    baseline = run_path(root, dataset, seed, "cv_pure")
    with np.load(baseline / "split_indices.npz") as saved:
        check_splits(path / "split_indices.npz", dict(saved))
    checkpoint = torch.load(path / "checkpoint.pt", map_location="cpu", weights_only=True)
    if not all(torch.isfinite(value).all() for value in checkpoint["model_state_dict"].values()):
        raise RuntimeError(f"Non-finite checkpoint: {path}")
    metrics = read_json(path / "metrics.json")
    if not all(np.isfinite(metrics[key]) for key in ("test_rmse", "test_mae", "test_r2")):
        raise RuntimeError(f"Non-finite metrics: {path}")
    return metrics


def command(root: Path, dataset: str, seed: int) -> list[str]:
    result = [sys.executable, "-m", "pure_resmlp.training.run_experiment",
              "--experiment", EXPERIMENT, "--dataset", dataset, "--model", "cv_mlp",
              "--seed", str(seed), "--split-seed", "0", "--device", "cuda:0",
              "--output-root", str(root), "--train-fraction", "1.0",
              "--energy-target", "Y1"]
    for key in ("hidden_dim", "epochs", "patience", "batch_size", "learning_rate"):
        result.extend(["--" + key.replace("_", "-"), str(PARAMETERS[key])])
    return result


def run_queue(gpu: int, queue: list[tuple[str, int]], root: Path, log_dir: Path) -> None:
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    for dataset, seed in queue:
        path = run_path(root, dataset, seed)
        if path.exists() and any(path.iterdir()):
            validate_run(root, dataset, seed)
            print(f"REUSE {dataset} seed={seed}", flush=True)
            continue
        print(f"START gpu={gpu} {dataset} seed={seed}", flush=True)
        with (log_dir / f"{dataset}_seed_{seed}.log").open("w", encoding="utf-8") as log:
            subprocess.run(command(root, dataset, seed), env=environment,
                           cwd=Path(__file__).resolve().parents[2], stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        metrics = validate_run(root, dataset, seed)
        print(f"DONE gpu={gpu} {dataset} seed={seed} RMSE={metrics['test_rmse']:.6f}", flush=True)


def summarize(root: Path, report_dir: Path) -> None:
    models = ("mlp", "resmlp", "rv_pure", "cv_mlp", "cv_pure")
    labels = ("MLP", "ResMLP", "RV-PURe", "CV-MLP", "CV-PURe")
    rows = []
    text = ["# Scientific regression: five-model comparison", "",
            "Local export from saved checkpoints. The consolidated paper results are in jocs_v1.",
            "Split seed 0; training seeds 0-4. Test RMSE: mean and sample SD (ddof=1).", "",
            "| Dataset | " + " | ".join(labels) + " |",
            "| --- | " + " | ".join(["---:"] * len(models)) + " |"]
    for dataset in DATASETS:
        cells = []
        for model in models:
            values = []
            for seed in SEEDS:
                metrics = (validate_run(root, dataset, seed) if model == "cv_mlp"
                           else read_json(run_path(root, dataset, seed, model) / "metrics.json"))
                values.append(metrics["test_rmse"])
                rows.append(dict(dataset=dataset, model=model, seed=seed, **metrics))
            cells.append(f"{np.mean(values):.4f} +/- {np.std(values, ddof=1):.4f}")
        text.append("| " + dataset + " | " + " | ".join(cells) + " |")
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "summary.json").write_text(json.dumps({
        "experiment": EXPERIMENT, "split_seed": 0, "training_seeds": SEEDS,
        "cv_mlp_parameters": PARAMETERS, "records": rows,
    }, indent=2) + "\n", encoding="utf-8")
    (report_dir / "summary.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    print("\n".join(text), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, nargs="+", default=(0, 1))
    parser.add_argument("--runs-root", type=Path, default=RUNS_DIR)
    parser.add_argument("--log-dir", type=Path,
                        default=ARTIFACTS_DIR / "logs" / EXPERIMENT)
    parser.add_argument("--report-dir", type=Path,
                        default=ARTIFACTS_DIR / "reports" / EXPERIMENT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if len(set(args.gpus)) != len(args.gpus) or min(args.gpus) < 0:
        parser.error("GPU identifiers must be unique and non-negative")
    preflight(args.runs_root)
    if args.dry_run:
        print(f"Ready: {len(jobs())} CV-MLP jobs; no files changed, no training started")
        return 0
    if not args.summarize_only:
        import torch
        if not torch.cuda.is_available() or max(args.gpus) >= torch.cuda.device_count():
            parser.error("Requested GPUs are unavailable")
        args.log_dir.mkdir(parents=True, exist_ok=True)
        grid = jobs()
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
            futures = [pool.submit(run_queue, gpu, grid[index::len(args.gpus)],
                                   args.runs_root, args.log_dir)
                       for index, gpu in enumerate(args.gpus)]
            for future in futures:
                future.result()
    summarize(args.runs_root, args.report_dir)
    print("Complete: all 30 CV-MLP runs validated", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
