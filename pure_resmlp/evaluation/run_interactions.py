"""Evaluate finite-difference feature-group interactions for a saved run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from pure_resmlp.data import (
    PreparedData,
    prepare_benchmark,
    prepare_bike_sharing,
    prepare_parkinsons_telemonitoring,
    prepare_synthetic_interactions,
)
from pure_resmlp.evaluation.finite_difference_interactions import (
    InteractionResult,
    finite_difference_interactions,
    interaction_recovery_metrics,
    summarize_recovery_resamples,
)
from pure_resmlp.training.run_experiment import (
    build_torch_model,
    predict_torch,
    resolve_device,
    transform_input_matrix,
)


def _load_data(config: Mapping[str, Any]) -> PreparedData:
    dataset = str(config["dataset"])
    parameters = config["parameters"]
    split_seed = int(config["split_seed"])
    if dataset == "parkinsons_telemonitoring":
        return prepare_parkinsons_telemonitoring()
    if dataset == "bike_sharing":
        return prepare_bike_sharing()
    if dataset.startswith("synthetic_"):
        return prepare_synthetic_interactions(
            dataset.removeprefix("synthetic_"),
            n_samples=int(parameters["synthetic_samples"]),
            irrelevant_features=int(parameters["irrelevant_features"]),
            noise_fraction=float(parameters["noise_fraction"]),
            data_seed=int(parameters["data_seed"]),
            split_seed=split_seed,
        ).prepared
    return prepare_benchmark(
        dataset,
        split_seed=split_seed,
        energy_target=parameters.get("energy_target") or "Y1",
    )


def _select_groups(
    data: PreparedData,
    scope: str,
    excluded_groups: tuple[str, ...] = (),
) -> dict[str, tuple[int, ...]]:
    if scope == "all":
        return {
            name: indices
            for name, indices in data.feature_groups.items()
            if name not in excluded_groups
        }
    selected: dict[str, tuple[int, ...]] = {}
    for name, indices in data.feature_groups.items():
        if name in excluded_groups:
            continue
        role = data.feature_group_roles[name]
        if scope == "ordinary" and role == "ordinary":
            selected[name] = indices
        elif scope == "context" and role == "context":
            selected[name] = indices
    return selected


def _write_matrix(path: Path, result: InteractionResult) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("feature_group", *result.group_names))
        for name, row in zip(result.group_names, result.matrix, strict=True):
            writer.writerow((name, *(float(value) for value in row)))


def _write_ranked(path: Path, result: InteractionResult) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("rank", "group_a", "group_b", "interaction_strength"))
        for rank, (left, right, score) in enumerate(result.ranked_pairs(), start=1):
            writer.writerow((rank, left, right, score))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--n-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--resample-seeds",
        type=int,
        nargs="+",
        help=(
            "repeat only the interaction reference sampling for these seeds and "
            "write interaction_robustness.json without overwriting the primary result"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--scope", choices=("all", "ordinary", "context"), default="all")
    args = parser.parse_args()

    config = json.loads((args.run_dir / "config.json").read_text(encoding="utf-8"))
    model_name = str(config["model"])
    data = _load_data(config)
    parameters = config["parameters"]
    excluded_groups = tuple(parameters.get("excluded_groups") or ())
    device = resolve_device(args.device)
    model = build_torch_model(
        model_name,
        input_dim=int(parameters["input_dim"]),
        hidden_dim=int(parameters["hidden_dim"]),
    ).to(device)
    checkpoint = torch.load(
        args.run_dir / "checkpoint.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    complex_input = model_name in {"cv_mlp", "cv_pure"}

    def predict_base(X: np.ndarray) -> np.ndarray:
        transformed, _names = transform_input_matrix(
            data, model_name, X, excluded_groups
        )
        return predict_torch(
            model,
            transformed,
            device=device,
            batch_size=args.batch_size,
            complex_input=complex_input,
        )

    groups = _select_groups(data, args.scope, excluded_groups)
    true_pairs = parameters.get("true_pairs")
    if args.resample_seeds is not None:
        if not true_pairs:
            parser.error("--resample-seeds requires a dataset with known true pairs")
        if len(set(args.resample_seeds)) != len(args.resample_seeds):
            parser.error("--resample-seeds must be unique")
        records: list[dict[str, Any]] = []
        for sampling_seed in args.resample_seeds:
            result = finite_difference_interactions(
                predict_base,
                data.X_test,
                data.X_train,
                groups,
                n_samples=args.n_samples,
                seed=sampling_seed,
            )
            record: dict[str, Any] = {
                "sampling_seed": sampling_seed,
                "top_pairs": result.ranked_pairs()[:10],
                **interaction_recovery_metrics(result, true_pairs),
            }
            records.append(record)
        robustness: dict[str, Any] = {
            "schema_version": 1,
            "n_samples_per_resample": min(args.n_samples, len(data.X_test)),
            "scope": args.scope,
            "sampling_seeds": args.resample_seeds,
            "summary": summarize_recovery_resamples(records),
            "records": records,
        }
        (args.run_dir / "interaction_robustness.json").write_text(
            json.dumps(robustness, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(robustness, indent=2))
    else:
        result = finite_difference_interactions(
            predict_base,
            data.X_test,
            data.X_train,
            groups,
            n_samples=args.n_samples,
            seed=args.seed,
        )
        _write_matrix(args.run_dir / "interaction_matrix.csv", result)
        _write_ranked(args.run_dir / "interaction_ranked.csv", result)
        metrics: dict[str, Any] = {
            "n_samples": min(args.n_samples, len(data.X_test)),
            "scope": args.scope,
            "top_pairs": result.ranked_pairs()[:10],
        }
        if true_pairs:
            metrics.update(interaction_recovery_metrics(result, true_pairs))
        (args.run_dir / "interaction_metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
