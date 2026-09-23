"""Reload the scientific CV-MLP checkpoints and verify their saved predictions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from pure_resmlp.data import prepare_benchmark
from pure_resmlp.paths import ARTIFACTS_DIR, RUNS_DIR
from pure_resmlp.training.run_experiment import build_torch_model, predict_torch
from pure_resmlp.training.run_scientific_cv_mlp import (
    DATASETS, EXPERIMENT, SEEDS, read_json, run_path, validate_run,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=RUNS_DIR)
    parser.add_argument("--report-dir", type=Path,
                        default=ARTIFACTS_DIR / "reports" / EXPERIMENT)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    records = []
    device = torch.device(args.device)
    for dataset in DATASETS:
        data = prepare_benchmark(dataset, split_seed=0, energy_target="Y1")
        for seed in SEEDS:
            path = run_path(args.runs_root, dataset, seed)
            metrics = validate_run(args.runs_root, dataset, seed)
            checkpoint = torch.load(path / "checkpoint.pt", map_location="cpu", weights_only=True)
            assert checkpoint["config"] == read_json(path / "config.json")
            baseline = torch.load(run_path(args.runs_root, dataset, seed, "cv_pure")
                                  / "checkpoint.pt", map_location="cpu", weights_only=True)
            metadata = checkpoint["metadata"]
            for name, values in {
                "x_scaler_mean": data.x_scaler.mean_, "x_scaler_scale": data.x_scaler.scale_,
                "y_scaler_mean": data.y_scaler.mean_, "y_scaler_scale": data.y_scaler.scale_,
            }.items():
                np.testing.assert_allclose(metadata[name], values, rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(metadata[name], baseline["metadata"][name],
                                           rtol=1e-12, atol=1e-12)
            assert list(metadata["feature_names"]) == list(data.feature_names)
            model = build_torch_model("cv_mlp", data.input_dim, 64).to(device)
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            scaled = predict_torch(model, data.X_test, device=device, batch_size=128,
                                   complex_input=True)
            restored = data.y_scaler.inverse_transform(scaled.reshape(-1, 1)).ravel()
            with (path / "predictions.csv").open(encoding="utf-8", newline="") as stream:
                saved = list(csv.DictReader(stream))
            assert all(row["split"] == "test" for row in saved)
            np.testing.assert_array_equal([int(row["sample_index"]) for row in saved], data.test_indices)
            true = np.array([float(row["y_true"]) for row in saved])
            predicted = np.array([float(row["y_pred"]) for row in saved])
            np.testing.assert_allclose(true, data.y_test_raw, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(restored, predicted, rtol=1e-5, atol=1e-5)
            residual = true - predicted
            recomputed = dict(test_rmse=float(np.sqrt(np.mean(residual ** 2))),
                              test_mae=float(np.mean(np.abs(residual))),
                              test_r2=float(1 - np.sum(residual ** 2) / np.sum((true - true.mean()) ** 2)))
            for key, value in recomputed.items():
                np.testing.assert_allclose(value, metrics[key], rtol=1e-10, atol=1e-10)
            with (path / "history.csv").open(encoding="utf-8", newline="") as stream:
                history = list(csv.DictReader(stream))
            best = min(history, key=lambda row: float(row["validation_mse"]))
            assert int(float(best["epoch"])) == checkpoint["epoch"] == metrics["best_epoch"]
            dof = sum(p.numel() * (2 if torch.is_complex(p) else 1)
                      for p in model.parameters() if p.requires_grad)
            assert dof == metrics["trainable_real_dof"]
            records.append(dict(dataset=dataset, seed=seed, status="passed",
                                max_prediction_difference=float(np.max(np.abs(restored - predicted))),
                                trainable_real_dof=dof, **recomputed))
            print(f"PASS {dataset} seed={seed}: checkpoint, split, scalers, predictions, metrics", flush=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "audit.json").write_text(json.dumps({
        "experiment": EXPERIMENT, "runs_checked": len(records), "status": "passed",
        "checks": "Checkpoint reload; baseline-identical splits/scalers; prediction replay; "
                  "recomputed metrics; validation-selected epoch; parameter count.",
        "records": records,
    }, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
