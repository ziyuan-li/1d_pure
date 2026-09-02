"""Summarize the complete JoCS experiment suite without recomputing models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


BENCHMARK_EXPERIMENT = "jocs_benchmark_v1"
BENCHMARK_DATASETS = (
    "friedman",
    "concrete",
    "california",
    "energy",
    "yacht",
    "auto_mpg",
)
BENCHMARK_MODELS = ("mlp", "resmlp", "rv_pure", "cv_pure")
BENCHMARK_SEEDS = tuple(range(5))
INTERACTION_SAMPLING_SEEDS = tuple(range(5))
SYNTHETIC_EXPERIMENT = "jocs_synthetic_v1"
CV_BASELINE_EXPERIMENT = "jocs_complex_architecture_control_v1"
CONTEXT_EXPERIMENTS = (
    "jocs_bike_context_v1",
    "jocs_parkinsons_context_v1",
)
ABLATION_EXPERIMENTS = (
    "jocs_bike_ablation_no_time_v1",
    "jocs_bike_ablation_no_calendar_v1",
    "jocs_bike_ablation_no_weather_v1",
)
EXPECTED_MANIFESTS = {
    BENCHMARK_EXPERIMENT: (
        len(BENCHMARK_DATASETS) * len(BENCHMARK_MODELS) * len(BENCHMARK_SEEDS)
    ),
    SYNTHETIC_EXPERIMENT: 24,
    "jocs_bike_context_v1": 30,
    "jocs_parkinsons_context_v1": 30,
    CV_BASELINE_EXPERIMENT: 13,
    **{name: 3 for name in ABLATION_EXPERIMENTS},
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_runs(root: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for experiment, expected in EXPECTED_MANIFESTS.items():
        manifests = sorted((root / experiment).rglob("manifest.json"))
        if len(manifests) != expected:
            raise RuntimeError(
                f"{experiment} has {len(manifests)} manifests; expected {expected}"
            )
        for manifest in manifests:
            run_dir = manifest.parent
            config = _load_json(run_dir / "config.json")
            metrics = _load_json(run_dir / "metrics.json")
            parameters = config.get("parameters", {})
            record: dict[str, Any] = {
                "experiment": config["experiment"],
                "dataset": config["dataset"],
                "model": config["model"],
                "seed": int(config["seed"]),
                "split_seed": int(config["split_seed"]),
                "context_ablation": parameters.get("context_ablation", "none"),
                "run_dir": str(run_dir),
            }
            for key in (
                "test_mae",
                "test_rmse",
                "test_r2",
                "best_epoch",
                "training_seconds",
                "inference_seconds",
                "trainable_real_dof",
                "test_samples",
            ):
                record[key] = metrics.get(key)
            record["catastrophic_test_failure"] = int(
                metrics.get("test_r2") is not None and metrics["test_r2"] < 0
            )
            interaction_path = run_dir / "interaction_metrics.json"
            if experiment != BENCHMARK_EXPERIMENT and not interaction_path.is_file():
                raise RuntimeError(f"missing interaction metrics: {run_dir}")
            if interaction_path.is_file():
                interactions = _load_json(interaction_path)
                for key in (
                    "precision_at_k",
                    "recall_at_k",
                    "true_interaction_mean_rank",
                ):
                    record[key] = interactions.get(key)
                record["interaction_samples"] = interactions.get("n_samples")
            records.append(record)
    frame = pd.DataFrame.from_records(records)
    if len(frame) != sum(EXPECTED_MANIFESTS.values()):
        raise RuntimeError("collected run count does not match the frozen protocol")
    benchmark = frame[frame["experiment"] == BENCHMARK_EXPERIMENT]
    actual_benchmark_runs = set(
        benchmark[["dataset", "model", "seed"]].itertuples(index=False, name=None)
    )
    expected_benchmark_runs = {
        (dataset, model, seed)
        for dataset in BENCHMARK_DATASETS
        for model in BENCHMARK_MODELS
        for seed in BENCHMARK_SEEDS
    }
    if actual_benchmark_runs != expected_benchmark_runs:
        missing = sorted(expected_benchmark_runs - actual_benchmark_runs)
        unexpected = sorted(actual_benchmark_runs - expected_benchmark_runs)
        raise RuntimeError(
            "benchmark composition does not match the frozen protocol; "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    controls = frame[frame["experiment"] == CV_BASELINE_EXPERIMENT]
    actual_control_runs = set(
        controls[["dataset", "model", "seed"]].itertuples(index=False, name=None)
    )
    expected_control_runs = {
        *(('synthetic_higher_order', 'cv_mlp', seed) for seed in range(3)),
        *(('parkinsons_telemonitoring', 'cv_mlp', seed) for seed in range(5)),
        *(('bike_sharing', 'cv_mlp', seed) for seed in range(5)),
    }
    if actual_control_runs != expected_control_runs:
        raise RuntimeError("CV baseline composition differs from the frozen protocol")
    return frame.sort_values(["experiment", "dataset", "model", "seed"])


def aggregate(
    frame: pd.DataFrame,
    group_columns: list[str],
    metric_columns: Iterable[str],
) -> pd.DataFrame:
    metrics = [name for name in metric_columns if frame[name].notna().any()]
    grouped = frame.groupby(group_columns, sort=True, dropna=False)
    pieces: list[pd.Series] = []
    for keys, values in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: dict[str, Any] = dict(zip(group_columns, keys, strict=True))
        row["n"] = len(values)
        for metric in metrics:
            numeric = pd.to_numeric(values[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(numeric.mean()) if len(numeric) else np.nan
            row[f"{metric}_std"] = (
                float(numeric.std(ddof=1)) if len(numeric) > 1 else 0.0
            )
        pieces.append(pd.Series(row))
    return pd.DataFrame(pieces)


def collect_interaction_robustness(frame: pd.DataFrame) -> pd.DataFrame:
    """Collect nested attribution resamples without counting them as model runs."""

    records: list[dict[str, Any]] = []
    synthetic = frame[
        frame["experiment"].isin((SYNTHETIC_EXPERIMENT, CV_BASELINE_EXPERIMENT))
        & frame["dataset"].str.startswith("synthetic_")
    ]
    for _, run in synthetic.iterrows():
        path = Path(run["run_dir"]) / "interaction_robustness.json"
        if not path.is_file():
            raise RuntimeError(f"missing interaction robustness results: {path.parent}")
        payload = _load_json(path)
        sampling_seeds = tuple(int(seed) for seed in payload["sampling_seeds"])
        if sampling_seeds != INTERACTION_SAMPLING_SEEDS:
            raise RuntimeError(
                f"unexpected interaction sampling seeds in {path}: {sampling_seeds}"
            )
        for record in payload["records"]:
            records.append(
                {
                    "dataset": run["dataset"],
                    "model": run["model"],
                    "training_seed": int(run["seed"]),
                    "sampling_seed": int(record["sampling_seed"]),
                    "precision_at_k": float(record["precision_at_k"]),
                    "recall_at_k": float(record["recall_at_k"]),
                    "true_interaction_mean_rank": float(
                        record["true_interaction_mean_rank"]
                    ),
                }
            )
    robustness = pd.DataFrame.from_records(records)
    expected = len(synthetic) * len(INTERACTION_SAMPLING_SEEDS)
    if len(robustness) != expected:
        raise RuntimeError(
            f"interaction robustness has {len(robustness)} resamples; expected {expected}"
        )
    return robustness.sort_values(
        ["dataset", "model", "training_seed", "sampling_seed"]
    )


def aggregate_interaction_robustness(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset, model), values in frame.groupby(["dataset", "model"], sort=True):
        row: dict[str, Any] = {
            "dataset": dataset,
            "model": model,
            "training_runs": int(values["training_seed"].nunique()),
            "attribution_resamples": len(values),
        }
        for metric in (
            "precision_at_k",
            "recall_at_k",
            "true_interaction_mean_rank",
        ):
            numeric = values[metric].to_numpy(dtype=np.float64)
            row[f"{metric}_mean"] = float(numeric.mean())
            row[f"{metric}_std"] = float(numeric.std(ddof=1))
            row[f"{metric}_min"] = float(numeric.min())
            row[f"{metric}_max"] = float(numeric.max())
        exact = (values["precision_at_k"] == 1.0) & (
            values["recall_at_k"] == 1.0
        )
        row["exact_recovery_rate"] = float(exact.mean())
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def paired_effects(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
    baseline: str,
    candidates: Iterable[str],
    metric: str,
    bootstrap_samples: int = 10_000,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(0)
    for keys, values in frame.groupby(group_columns, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        by_model = values.pivot(index="seed", columns="model", values=metric)
        for candidate in candidates:
            if baseline not in by_model or candidate not in by_model:
                continue
            paired = by_model[[baseline, candidate]].dropna()
            improvement = (
                paired[baseline].to_numpy(dtype=np.float64)
                - paired[candidate].to_numpy(dtype=np.float64)
            )
            draws = rng.choice(
                improvement,
                size=(bootstrap_samples, len(improvement)),
                replace=True,
            ).mean(axis=1)
            spread = float(np.std(improvement, ddof=1))
            row = dict(zip(group_columns, keys, strict=True))
            row.update(
                {
                    "metric": metric,
                    "baseline": baseline,
                    "candidate": candidate,
                    "n_pairs": len(improvement),
                    "mean_improvement": float(np.mean(improvement)),
                    "ci_2.5": float(np.quantile(draws, 0.025)),
                    "ci_97.5": float(np.quantile(draws, 0.975)),
                    "paired_effect_dz": (
                        float(np.mean(improvement) / spread) if spread > 0 else np.nan
                    ),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def paired_ablation_effects(
    context_runs: pd.DataFrame,
    ablation_runs: pd.DataFrame,
    *,
    bootstrap_samples: int = 10_000,
) -> pd.DataFrame:
    """Compare Bike ablations with the full RV-PURe on the same seeds."""

    full = context_runs[
        (context_runs["experiment"] == "jocs_bike_context_v1")
        & (context_runs["dataset"] == "bike_sharing")
        & (context_runs["model"] == "rv_pure")
    ][["seed", "test_mae"]]
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(0)
    for ablation, values in ablation_runs.groupby("context_ablation", sort=True):
        paired = full.merge(
            values[["seed", "test_mae"]],
            on="seed",
            suffixes=("_full", "_ablation"),
            validate="one_to_one",
        )
        difference = (
            paired["test_mae_ablation"].to_numpy(dtype=np.float64)
            - paired["test_mae_full"].to_numpy(dtype=np.float64)
        )
        draws = rng.choice(
            difference,
            size=(bootstrap_samples, len(difference)),
            replace=True,
        ).mean(axis=1)
        rows.append(
            {
                "dataset": "bike_sharing",
                "model": "rv_pure",
                "context_ablation": ablation,
                "metric": "test_mae",
                "n_pairs": len(difference),
                "mean_ablation_minus_full": float(np.mean(difference)),
                "ci_2.5": float(np.quantile(draws, 0.025)),
                "ci_97.5": float(np.quantile(draws, 0.975)),
            }
        )
    return pd.DataFrame.from_records(rows)


def _mean_std(row: pd.Series, metric: str, digits: int = 4) -> str:
    return (
        f"{row[f'{metric}_mean']:.{digits}f} ± "
        f"{row[f'{metric}_std']:.{digits}f}"
    )


def markdown_table(
    frame: pd.DataFrame,
    columns: list[tuple[str, str]],
) -> str:
    headers = [label for _column, label in columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in frame.iterrows():
        values: list[str] = []
        for column, _label in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_outputs(frame: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "all_runs.csv", index=False)

    benchmark_runs = frame[frame["experiment"] == BENCHMARK_EXPERIMENT]
    benchmark = aggregate(
        benchmark_runs,
        ["dataset", "model"],
        ("test_mae", "test_rmse", "test_r2"),
    )
    benchmark.to_csv(output_dir / "table1_benchmark.csv", index=False)

    context_runs = frame[
        frame["experiment"].isin((*CONTEXT_EXPERIMENTS, CV_BASELINE_EXPERIMENT))
        & frame["dataset"].isin(("bike_sharing", "parkinsons_telemonitoring"))
    ]
    context = aggregate(
        context_runs,
        ["dataset", "model"],
        ("test_mae", "test_rmse", "test_r2", "catastrophic_test_failure"),
    )
    context.to_csv(output_dir / "table2_context_main.csv", index=False)

    ablation_runs = frame[frame["experiment"].isin(ABLATION_EXPERIMENTS)]
    ablations = aggregate(
        ablation_runs,
        ["dataset", "context_ablation", "model"],
        ("test_mae", "test_rmse", "test_r2"),
    )
    ablations.to_csv(output_dir / "table2_context_ablations.csv", index=False)
    ablation_effects = paired_ablation_effects(context_runs, ablation_runs)
    ablation_effects.to_csv(
        output_dir / "table2_context_ablation_paired_effects.csv", index=False
    )

    synthetic_runs = frame[
        frame["experiment"].isin((SYNTHETIC_EXPERIMENT, CV_BASELINE_EXPERIMENT))
        & frame["dataset"].str.startswith("synthetic_")
    ]
    synthetic = aggregate(
        synthetic_runs,
        ["dataset", "model"],
        (
            "test_mae",
            "test_rmse",
            "test_r2",
            "precision_at_k",
            "recall_at_k",
            "true_interaction_mean_rank",
        ),
    )
    synthetic.to_csv(output_dir / "table3_synthetic.csv", index=False)

    robustness_long = collect_interaction_robustness(frame)
    robustness = aggregate_interaction_robustness(robustness_long)
    robustness.to_csv(
        output_dir / "table3_synthetic_interaction_robustness.csv", index=False
    )

    effects = [
        paired_effects(
            benchmark_runs,
            group_columns=["dataset"],
            baseline="mlp",
            candidates=("resmlp", "rv_pure", "cv_pure"),
            metric="test_rmse",
        ),
        paired_effects(
            context_runs,
            group_columns=["dataset"],
            baseline="mlp_context",
            candidates=(
                "mlp_ordinary",
                "mlp_explicit_cross",
                "dcnv2_context",
                "rv_pure",
                "cv_mlp",
                "cv_pure",
            ),
            metric="test_mae",
        ),
        paired_effects(
            synthetic_runs,
            group_columns=["dataset"],
            baseline="mlp",
            candidates=("resmlp", "rv_pure", "cv_mlp", "cv_pure"),
            metric="test_rmse",
        ),
        paired_effects(
            context_runs,
            group_columns=["dataset"],
            baseline="cv_mlp",
            candidates=("cv_pure",),
            metric="test_mae",
        ),
        paired_effects(
            context_runs,
            group_columns=["dataset"],
            baseline="cv_mlp",
            candidates=("cv_pure",),
            metric="test_rmse",
        ),
        paired_effects(
            synthetic_runs,
            group_columns=["dataset"],
            baseline="cv_mlp",
            candidates=("cv_pure",),
            metric="test_rmse",
        ),
    ]
    all_effects = pd.concat(effects, ignore_index=True)
    all_effects.to_csv(
        output_dir / "paired_effects_bootstrap.csv", index=False
    )
    all_effects[
        (all_effects["baseline"] == "cv_mlp")
        & (all_effects["candidate"] == "cv_pure")
    ].to_csv(output_dir / "cv_architecture_effects.csv", index=False)

    benchmark_display = benchmark.copy()
    for metric in ("test_mae", "test_rmse", "test_r2"):
        benchmark_display[metric] = benchmark_display.apply(
            _mean_std, axis=1, metric=metric
        )
    context_display = context.copy()
    for metric in ("test_mae", "test_rmse", "test_r2"):
        context_display[metric] = context_display.apply(
            _mean_std, axis=1, metric=metric
        )
    context_display["failures"] = context_display.apply(
        lambda row: (
            f"{int(round(row['catastrophic_test_failure_mean'] * row['n']))}/"
            f"{int(row['n'])}"
        ),
        axis=1,
    )
    ablation_display = ablations.copy()
    for metric in ("test_mae", "test_rmse", "test_r2"):
        ablation_display[metric] = ablation_display.apply(
            _mean_std, axis=1, metric=metric
        )
    synthetic_display = synthetic.copy()
    for metric in (
        "test_rmse",
        "precision_at_k",
        "recall_at_k",
        "true_interaction_mean_rank",
    ):
        synthetic_display[metric] = synthetic_display.apply(
            _mean_std, axis=1, metric=metric
        )
    robustness_display = robustness.copy()
    for metric in (
        "precision_at_k",
        "recall_at_k",
        "true_interaction_mean_rank",
    ):
        robustness_display[metric] = robustness_display.apply(
            _mean_std, axis=1, metric=metric
        )

    report = [
        "# PURe JoCS 正式实验自动汇总",
        "",
        f"完整正式运行：{len(frame)}。本文件由保存的运行结果生成，不包含 pilot/smoke。",
        "",
        "## Table 1：Scientific benchmark",
        "",
        markdown_table(
            benchmark_display,
            [
                ("dataset", "Dataset"),
                ("model", "Model"),
                ("test_mae", "MAE"),
                ("test_rmse", "RMSE"),
                ("test_r2", "R²"),
            ],
        ),
        "",
        "## Table 2：Context main results",
        "",
        markdown_table(
            context_display,
            [
                ("dataset", "Dataset"),
                ("model", "Model"),
                ("test_mae", "MAE"),
                ("test_rmse", "RMSE"),
                ("test_r2", "R²"),
                ("failures", "R²<0 runs"),
            ],
        ),
        "",
        "## Table 2b：Context ablations",
        "",
        markdown_table(
            ablation_display,
            [
                ("dataset", "Dataset"),
                ("context_ablation", "Ablation"),
                ("test_mae", "MAE"),
                ("test_rmse", "RMSE"),
                ("test_r2", "R²"),
            ],
        ),
        "",
        "## Table 3：Synthetic interaction recovery",
        "",
        markdown_table(
            synthetic_display,
            [
                ("dataset", "Dataset"),
                ("model", "Model"),
                ("test_rmse", "RMSE"),
                ("precision_at_k", "Precision@K"),
                ("recall_at_k", "Recall@K"),
                ("true_interaction_mean_rank", "Mean rank"),
            ],
        ),
        "",
        "## Table 3b：Interaction attribution sampling robustness",
        "",
        "Five reference-sampling seeds per saved model; these are attribution "
        "resamples, not additional training runs.",
        "",
        markdown_table(
            robustness_display,
            [
                ("dataset", "Dataset"),
                ("model", "Model"),
                ("attribution_resamples", "Resamples"),
                ("precision_at_k", "Precision@K"),
                ("recall_at_k", "Recall@K"),
                ("true_interaction_mean_rank", "Mean rank"),
                ("exact_recovery_rate", "Exact recovery rate"),
            ],
        ),
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(report), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    frame = collect_runs(args.runs_root)
    write_outputs(frame, args.output_dir)
    print(
        json.dumps(
            {
                "formal_runs": len(frame),
                "output_dir": str(args.output_dir),
                "files": sorted(path.name for path in args.output_dir.iterdir()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
