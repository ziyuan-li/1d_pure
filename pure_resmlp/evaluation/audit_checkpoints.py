"""Audit every checkpoint and run artifact used by the JoCS study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from pure_resmlp.evaluation.summarize_jocs import collect_runs
from pure_resmlp.paths import RUNS_DIR
from pure_resmlp.training.run_sample_efficiency_suite import (
    DATASET_MODELS,
    FRACTIONS,
)


REQUIRED_RUN_FILES = (
    "config.json",
    "environment.json",
    "split_indices.npz",
    "history.csv",
    "predictions.csv",
    "metrics.json",
    "checkpoint.pt",
    "manifest.json",
)
TRAINING_SEEDS = tuple(range(5))


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_sample_efficiency_runs(root: Path) -> list[Path]:
    expected_per_fraction = (
        sum(len(models) for models in DATASET_MODELS.values()) * len(TRAINING_SEEDS)
    )
    run_dirs: list[Path] = []
    for fraction, experiment in FRACTIONS.items():
        manifests = sorted((root / experiment).rglob("manifest.json"))
        if len(manifests) != expected_per_fraction:
            raise RuntimeError(
                f"{experiment} has {len(manifests)} manifests; "
                f"expected {expected_per_fraction}"
            )
        actual: set[tuple[str, str, int]] = set()
        for manifest in manifests:
            config = _load_json(manifest.parent / "config.json")
            parameters = config["parameters"]
            if float(parameters["train_fraction"]) != fraction:
                raise RuntimeError(
                    f"unexpected train_fraction in {manifest.parent}: "
                    f"{parameters['train_fraction']}"
                )
            actual.add(
                (config["dataset"], config["model"], int(config["seed"]))
            )
            run_dirs.append(manifest.parent)
        expected = {
            (dataset, model, seed)
            for dataset, models in DATASET_MODELS.items()
            for model in models
            for seed in TRAINING_SEEDS
        }
        if actual != expected:
            raise RuntimeError(
                f"{experiment} composition differs from the frozen protocol"
            )
    return run_dirs


def audit_group(label: str, run_dirs: list[Path], show_missing: int) -> int:
    problems: list[str] = []
    for run_dir in run_dirs:
        for name in REQUIRED_RUN_FILES:
            path = run_dir / name
            if not path.is_file():
                problems.append(f"missing: {path}")
        checkpoint = run_dir / "checkpoint.pt"
        if checkpoint.is_file():
            try:
                torch.load(checkpoint, map_location="cpu", weights_only=True)
            except Exception as error:  # pragma: no cover - depends on corrupt files
                problems.append(f"unreadable checkpoint: {checkpoint}: {error}")
    print(
        f"{label}: {len(run_dirs)} runs, "
        f"{len(problems)} missing or unreadable artifacts"
    )
    for problem in problems[:show_missing]:
        print(f"  {problem}")
    if len(problems) > show_missing:
        print(f"  ... and {len(problems) - show_missing} more")
    return len(problems)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=RUNS_DIR)
    parser.add_argument(
        "--show-missing",
        type=int,
        default=5,
        help="maximum number of problems to print per experiment group",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return a non-zero exit code when an artifact is missing or unreadable",
    )
    args = parser.parse_args()

    try:
        formal = collect_runs(args.runs_root)
        formal_dirs = [Path(path) for path in formal["run_dir"]]
        low_data_dirs = collect_sample_efficiency_runs(args.runs_root)
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        print(f"Protocol audit failed: {error}")
        return 1

    problems = 0
    problems += audit_group("Formal JoCS suite", formal_dirs, args.show_missing)
    problems += audit_group(
        "Low-data additions", low_data_dirs, args.show_missing
    )
    print(
        f"Audit total: {len(formal_dirs) + len(low_data_dirs)} runs, "
        f"{problems} problems"
    )
    return int(args.strict and problems > 0)


if __name__ == "__main__":
    raise SystemExit(main())
