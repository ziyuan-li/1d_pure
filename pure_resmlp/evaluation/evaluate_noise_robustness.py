"""Evaluate saved JoCS models under standardized Gaussian feature noise."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from pure_resmlp.evaluation.plot_style import (
    FULL_WIDTH,
    apply_publication_style,
    model_label,
    model_style,
    panel_label,
    save_publication_figure,
)
from pure_resmlp.evaluation.run_interactions import _load_data
from pure_resmlp.paths import ARTIFACTS_DIR, RUNS_DIR
from pure_resmlp.training.run_experiment import (
    build_torch_model,
    predict_torch,
    regression_metrics,
    resolve_device,
    transform_input_matrix,
)


SUITES = (
    (
        "jocs_benchmark_v1",
        ("friedman", "concrete", "california", "energy", "yacht", "auto_mpg"),
        ("mlp", "resmlp", "rv_pure", "cv_pure"),
    ),
    (
        "jocs_bike_context_v1",
        ("bike_sharing",),
        (
            "mlp_ordinary",
            "mlp_context",
            "mlp_explicit_cross",
            "dcnv2_context",
            "rv_pure",
            "cv_pure",
        ),
    ),
    (
        "jocs_parkinsons_context_v1",
        ("parkinsons_telemonitoring",),
        (
            "mlp_ordinary",
            "mlp_context",
            "mlp_explicit_cross",
            "dcnv2_context",
            "rv_pure",
            "cv_pure",
        ),
    ),
    (
        "jocs_complex_architecture_control_v1",
        ("bike_sharing", "parkinsons_telemonitoring"),
        ("cv_mlp",),
    ),
)


def continuous_ordinary_columns(data: Any) -> np.ndarray:
    """Return continuous ordinary columns, leaving context/one-hot columns intact."""

    ordinary = {
        index
        for group, indices in data.feature_groups.items()
        if data.feature_group_roles[group] == "ordinary"
        for index in indices
    }
    raw_train = data.x_scaler.inverse_transform(data.X_train)
    return np.asarray(
        [index for index in sorted(ordinary) if np.unique(raw_train[:, index]).size > 2],
        dtype=np.int64,
    )


def add_gaussian_feature_noise(
    X: np.ndarray,
    columns: np.ndarray,
    noise_std: float,
    seed: int,
) -> np.ndarray:
    """Return a copy with Gaussian noise on selected standardized columns."""

    if noise_std < 0:
        raise ValueError("noise standard deviation must be non-negative")
    result = np.asarray(X, dtype=np.float32).copy()
    if noise_std == 0 or len(columns) == 0:
        return result
    noise = np.random.default_rng(seed).normal(
        0.0, noise_std, size=(len(result), len(columns))
    )
    result[:, columns] += noise.astype(np.float32)
    return result


def _bootstrap_interval(values: np.ndarray, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(10_000, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def summarize_noise_rows(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average corruptions within training seed, then summarize across seeds."""

    seed_rows = (
        raw.groupby(
            ["experiment", "dataset", "model", "training_seed", "noise_std"],
            as_index=False,
        )[["test_mae", "test_rmse", "test_r2"]]
        .mean()
        .sort_values(["dataset", "model", "training_seed", "noise_std"])
    )
    clean = seed_rows.loc[seed_rows["noise_std"] == 0].rename(
        columns={
            "test_mae": "clean_mae",
            "test_rmse": "clean_rmse",
            "test_r2": "clean_r2",
        }
    )[["experiment", "dataset", "model", "training_seed", "clean_mae", "clean_rmse", "clean_r2"]]
    seed_rows = seed_rows.merge(
        clean, on=["experiment", "dataset", "model", "training_seed"], validate="many_to_one"
    )
    seed_rows["mae_increase_pct"] = (
        100.0 * (seed_rows["test_mae"] - seed_rows["clean_mae"]) / seed_rows["clean_mae"]
    )
    seed_rows["rmse_increase_pct"] = (
        100.0 * (seed_rows["test_rmse"] - seed_rows["clean_rmse"]) / seed_rows["clean_rmse"]
    )
    seed_rows["r2_drop"] = seed_rows["clean_r2"] - seed_rows["test_r2"]
    seed_rows["test_target_sd"] = seed_rows["clean_rmse"] / np.sqrt(
        np.maximum(1.0 - seed_rows["clean_r2"], np.finfo(np.float64).eps)
    )
    seed_rows["test_nrmse"] = (
        seed_rows["test_rmse"] / seed_rows["test_target_sd"]
    )
    seed_rows["degradation_nrmse"] = (
        (seed_rows["test_rmse"] - seed_rows["clean_rmse"])
        / seed_rows["test_target_sd"]
    )

    summaries: list[dict[str, Any]] = []
    group_columns = ["experiment", "dataset", "model", "noise_std"]
    for keys, values in seed_rows.groupby(group_columns, sort=True):
        row = dict(zip(group_columns, keys, strict=True))
        row["training_seeds"] = int(values["training_seed"].nunique())
        for metric in (
            "test_mae",
            "test_rmse",
            "test_r2",
            "mae_increase_pct",
            "rmse_increase_pct",
            "r2_drop",
            "test_nrmse",
            "degradation_nrmse",
        ):
            numbers = values[metric].to_numpy(dtype=np.float64)
            low, high = _bootstrap_interval(numbers)
            row[f"{metric}_mean"] = float(numbers.mean())
            row[f"{metric}_std"] = float(numbers.std(ddof=1))
            row[f"{metric}_ci_2.5"] = low
            row[f"{metric}_ci_97.5"] = high
        summaries.append(row)
    return seed_rows, pd.DataFrame.from_records(summaries)


def _auc_by_training_seed(seed_rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    group_columns = ["experiment", "dataset", "model", "training_seed"]
    for keys, curve in seed_rows.groupby(group_columns, sort=True):
        curve = curve.sort_values("noise_std")
        noise = curve["noise_std"].to_numpy(dtype=np.float64)
        noise_range = float(noise[-1] - noise[0])
        if noise_range <= 0:
            raise ValueError("AUC requires at least two distinct noise levels")
        row = dict(zip(group_columns, keys, strict=True))
        row["performance_auc"] = float(
            np.trapz(curve["test_nrmse"].to_numpy(), noise) / noise_range
        )
        row["degradation_auc"] = float(
            np.trapz(curve["degradation_nrmse"].to_numpy(), noise) / noise_range
        )
        records.append(row)
    return pd.DataFrame.from_records(records)


def summarize_auc(seed_rows: pd.DataFrame) -> pd.DataFrame:
    per_seed = _auc_by_training_seed(seed_rows)
    rows: list[dict[str, Any]] = []
    group_columns = ["experiment", "dataset", "model"]
    for keys, values in per_seed.groupby(group_columns, sort=True):
        row = dict(zip(group_columns, keys, strict=True))
        row["training_seeds"] = int(values["training_seed"].nunique())
        for metric in ("performance_auc", "degradation_auc"):
            numbers = values[metric].to_numpy(dtype=np.float64)
            low, high = _bootstrap_interval(numbers)
            row[f"{metric}_mean"] = float(numbers.mean())
            row[f"{metric}_std"] = float(numbers.std(ddof=1))
            row[f"{metric}_ci_2.5"] = low
            row[f"{metric}_ci_97.5"] = high
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def summarize_overall_auc(seed_rows: pd.DataFrame) -> pd.DataFrame:
    """Average normalized AUC across datasets, with every dataset weighted equally."""

    per_seed = _auc_by_training_seed(seed_rows)
    suite_definitions = {
        "scientific": {
            "datasets": {"friedman", "concrete", "california", "energy", "yacht", "auto_mpg"},
            "models": {"mlp", "resmlp", "rv_pure", "cv_pure"},
        },
        "context": {
            "datasets": {"bike_sharing", "parkinsons_telemonitoring"},
            "models": {
                "mlp_context",
                "mlp_explicit_cross",
                "dcnv2_context",
                "rv_pure",
                "cv_mlp",
                "cv_pure",
            },
        },
    }
    rows: list[dict[str, Any]] = []
    for suite, definition in suite_definitions.items():
        selected = per_seed.loc[
            per_seed["dataset"].isin(definition["datasets"])
            & per_seed["model"].isin(definition["models"])
        ]
        within_seed = (
            selected.groupby(["model", "training_seed"], as_index=False)[
                ["performance_auc", "degradation_auc"]
            ]
            .mean()
        )
        for model, values in within_seed.groupby("model", sort=True):
            model_dataset_count = int(
                selected.loc[selected["model"] == model, "dataset"].nunique()
            )
            row: dict[str, Any] = {
                "suite": suite,
                "model": model,
                "datasets": model_dataset_count,
                "training_seeds": int(values["training_seed"].nunique()),
                "dataset_weighting": "equal",
            }
            for metric in ("performance_auc", "degradation_auc"):
                numbers = values[metric].to_numpy(dtype=np.float64)
                low, high = _bootstrap_interval(numbers)
                row[f"{metric}_mean"] = float(numbers.mean())
                row[f"{metric}_std"] = float(numbers.std(ddof=1))
                row[f"{metric}_ci_2.5"] = low
                row[f"{metric}_ci_97.5"] = high
            rows.append(row)
    return pd.DataFrame.from_records(rows)


def summarize_auc_effects(seed_rows: pd.DataFrame) -> pd.DataFrame:
    """Paired CV-PURe AUC improvements after equal averaging over datasets."""

    per_seed = _auc_by_training_seed(seed_rows)
    definitions = {
        "scientific": {
            "datasets": {"friedman", "concrete", "california", "energy", "yacht", "auto_mpg"},
            "baselines": ("mlp", "resmlp", "rv_pure"),
        },
        "context": {
            "datasets": {"bike_sharing", "parkinsons_telemonitoring"},
            "baselines": (
                "mlp_context",
                "mlp_explicit_cross",
                "dcnv2_context",
                "rv_pure",
                "cv_mlp",
            ),
        },
    }
    rows: list[dict[str, Any]] = []
    for suite, definition in definitions.items():
        selected = per_seed.loc[per_seed["dataset"].isin(definition["datasets"])]
        for metric in ("performance_auc", "degradation_auc"):
            pivot = selected.pivot(
                index=["dataset", "training_seed"], columns="model", values=metric
            )
            for baseline in definition["baselines"]:
                paired = (pivot[baseline] - pivot["cv_pure"]).rename("improvement")
                within_seed = paired.groupby("training_seed").mean().to_numpy()
                low, high = _bootstrap_interval(within_seed)
                rows.append(
                    {
                        "suite": suite,
                        "metric": metric,
                        "baseline": baseline,
                        "candidate": "cv_pure",
                        "datasets": len(definition["datasets"]),
                        "training_seed_pairs": len(within_seed),
                        "mean_improvement": float(within_seed.mean()),
                        "ci_2.5": low,
                        "ci_97.5": high,
                    }
                )
    return pd.DataFrame.from_records(rows)


def _plot(summary: pd.DataFrame, output_dir: Path) -> None:
    apply_publication_style()
    datasets = list(
        dict.fromkeys(
            dataset for _experiment, items, _models in SUITES for dataset in items
        )
    )
    fig = plt.figure(figsize=(FULL_WIDTH, 4.70), constrained_layout=True)
    grid = fig.add_gridspec(3, 4, height_ratios=(1.0, 1.0, 0.16))
    axes = np.asarray(
        [[fig.add_subplot(grid[row, column]) for column in range(4)] for row in range(2)]
    )
    legend_axis = fig.add_subplot(grid[2, :])
    legend_axis.axis("off")
    for panel_index, (axis, dataset) in enumerate(
        zip(axes.flat, datasets, strict=True)
    ):
        values = summary.loc[summary["dataset"] == dataset]
        for model, curve in values.groupby("model", sort=False):
            curve = curve.sort_values("noise_std")
            style = model_style(model)
            x = curve["noise_std"].to_numpy(dtype=float)
            mean = curve["test_nrmse_mean"].to_numpy(dtype=float)
            std = curve["test_nrmse_std"].to_numpy(dtype=float)
            axis.plot(
                x, mean, label=model_label(model),
                color=style["color"], marker=style["marker"],
                linestyle=style["linestyle"], markeredgecolor="white",
                markeredgewidth=0.3,
            )
            axis.fill_between(
                x, mean - std, mean + std, color=style["color"],
                alpha=0.09, linewidth=0,
            )
        axis.set_title(dataset.replace("_", " ").title())
        if panel_index >= 4:
            axis.set_xlabel(r"Noise $\sigma$ (training-SD units)")
        if panel_index % 4 == 0:
            axis.set_ylabel("Normalized test RMSE")
        axis.set_xticks((0.0, 0.02, 0.05, 0.10))
        axis.grid(axis="y")
        axis.set_axisbelow(True)
        panel_label(axis, chr(ord("a") + panel_index))
    legend_entries: dict[str, Any] = {}
    for axis in axes.flat:
        handles, labels = axis.get_legend_handles_labels()
        legend_entries.update(zip(labels, handles, strict=True))
    legend_axis.legend(
        list(legend_entries.values()),
        list(legend_entries),
        loc="center",
        ncol=5,
        frameon=False,
        columnspacing=1.1,
        handlelength=2.2,
    )
    fig.get_layout_engine().set(h_pad=0.03, w_pad=0.03, hspace=0.04, wspace=0.04)
    save_publication_figure(fig, output_dir, "figure3_noise_robustness")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=RUNS_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ARTIFACTS_DIR / "reports" / "jocs_v1" / "robustness",
    )
    parser.add_argument("--noise-levels", type=float, nargs="+", default=(0.0, 0.02, 0.05, 0.10))
    parser.add_argument("--corruption-seeds", type=int, nargs="+", default=tuple(range(5)))
    parser.add_argument("--training-seeds", type=int, nargs="+", default=tuple(range(5)))
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if sorted(set(args.noise_levels)) != sorted(args.noise_levels) or 0.0 not in args.noise_levels:
        parser.error("noise-levels must be unique and include 0")
    if any(level < 0 for level in args.noise_levels):
        parser.error("noise-levels must be non-negative")
    if len(set(args.corruption_seeds)) != len(args.corruption_seeds):
        parser.error("corruption-seeds must be unique")

    output_files = (
        "noise_robustness_raw.csv",
        "noise_robustness_training_seed.csv",
        "noise_robustness_summary.csv",
        "noise_robustness_auc.csv",
        "noise_robustness_overall_auc.csv",
        "noise_robustness_auc_effects.csv",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.overwrite and any((args.output_dir / name).exists() for name in output_files):
        parser.error("robustness outputs already exist; pass --overwrite to replace them")

    device = resolve_device(args.device)
    records: list[dict[str, Any]] = []
    data_cache: dict[tuple[str, str], Any] = {}
    for experiment, datasets, models in SUITES:
        for dataset in datasets:
            for training_seed in args.training_seeds:
                for model_name in models:
                    run_dir = args.runs_root / experiment / dataset / model_name / f"seed_{training_seed}"
                    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
                    cache_key = (experiment, dataset)
                    if cache_key not in data_cache:
                        data_cache[cache_key] = _load_data(config)
                    data = data_cache[cache_key]
                    parameters = config["parameters"]
                    excluded_groups = tuple(parameters.get("excluded_groups") or ())
                    model = build_torch_model(
                        model_name,
                        input_dim=int(parameters["input_dim"]),
                        hidden_dim=int(parameters["hidden_dim"]),
                    ).to(device)
                    checkpoint = torch.load(
                        run_dir / "checkpoint.pt", map_location=device, weights_only=False
                    )
                    model.load_state_dict(checkpoint["model_state_dict"])
                    noisy_columns = continuous_ordinary_columns(data)
                    complex_input = model_name in {"cv_mlp", "cv_pure"}
                    for noise_std in sorted(args.noise_levels):
                        corruption_seeds = (-1,) if noise_std == 0 else args.corruption_seeds
                        for corruption_seed in corruption_seeds:
                            corrupted = add_gaussian_feature_noise(
                                data.X_test,
                                noisy_columns,
                                noise_std,
                                0 if corruption_seed == -1 else corruption_seed,
                            )
                            transformed, _names = transform_input_matrix(
                                data, model_name, corrupted, excluded_groups
                            )
                            prediction = predict_torch(
                                model,
                                transformed,
                                device=device,
                                batch_size=args.batch_size,
                                complex_input=complex_input,
                            )
                            _prediction_raw, metrics = regression_metrics(data, prediction)
                            records.append(
                                {
                                    "experiment": experiment,
                                    "dataset": dataset,
                                    "model": model_name,
                                    "training_seed": training_seed,
                                    "noise_std": noise_std,
                                    "corruption_seed": corruption_seed,
                                    "perturbed_features": len(noisy_columns),
                                    "test_mae": metrics["test_mae"],
                                    "test_rmse": metrics["test_rmse"],
                                    "test_r2": metrics["test_r2"],
                                }
                            )
                    del model

    raw = pd.DataFrame.from_records(records)
    seed_rows, summary = summarize_noise_rows(raw)
    auc = summarize_auc(seed_rows)
    overall_auc = summarize_overall_auc(seed_rows)
    auc_effects = summarize_auc_effects(seed_rows)
    raw.to_csv(args.output_dir / output_files[0], index=False)
    seed_rows.to_csv(args.output_dir / output_files[1], index=False)
    summary.to_csv(args.output_dir / output_files[2], index=False)
    auc.to_csv(args.output_dir / output_files[3], index=False)
    overall_auc.to_csv(args.output_dir / output_files[4], index=False)
    auc_effects.to_csv(args.output_dir / output_files[5], index=False)
    protocol = {
        "noise_space": "standardized feature space",
        "perturbed_columns": "continuous ordinary features only; context and binary one-hot columns fixed",
        "noise_levels": sorted(args.noise_levels),
        "corruption_seeds": args.corruption_seeds,
        "training_seeds": args.training_seeds,
        "aggregation": "corruption seeds averaged within training seed; uncertainty across training seeds",
        "primary_metric": "performance_auc: interval-average target-SD-normalized noisy RMSE; lower is better",
        "secondary_metric": "degradation_auc: interval-average target-SD-normalized RMSE increase from clean; lower is better",
        "dataset_weighting": "equal within scientific and context suites",
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )
    _plot(summary, args.output_dir)
    print(json.dumps({"evaluations": len(raw), "output_dir": str(args.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
