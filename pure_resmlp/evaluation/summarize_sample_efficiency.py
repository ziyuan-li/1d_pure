"""Summarize the compact JoCS sample-efficiency experiment."""

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

from pure_resmlp.evaluation.plot_style import (
    FULL_WIDTH,
    apply_publication_style,
    model_label,
    model_style,
    panel_label,
    save_publication_figure,
)
from pure_resmlp.paths import ARTIFACTS_DIR, RUNS_DIR


FRACTION_EXPERIMENTS = {
    0.10: "jocs_sample_efficiency_f010_v1",
    0.25: "jocs_sample_efficiency_f025_v1",
    0.50: "jocs_sample_efficiency_f050_v1",
}
FULL_EXPERIMENTS = {
    "friedman": "jocs_benchmark_v1",
    "concrete": "jocs_benchmark_v1",
    "parkinsons_telemonitoring": "jocs_parkinsons_context_v1",
}
MODELS = {
    "friedman": ("mlp", "resmlp", "rv_pure", "cv_pure"),
    "concrete": ("mlp", "resmlp", "rv_pure", "cv_pure"),
    "parkinsons_telemonitoring": (
        "mlp_ordinary",
        "mlp_context",
        "mlp_explicit_cross",
        "dcnv2_context",
        "rv_pure",
        "cv_pure",
    ),
}


def collect_rows(runs_root: Path, seeds: tuple[int, ...]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    missing: list[Path] = []
    for dataset, models in MODELS.items():
        for fraction in (*FRACTION_EXPERIMENTS, 1.0):
            experiment = (
                FRACTION_EXPERIMENTS[fraction]
                if fraction < 1.0
                else FULL_EXPERIMENTS[dataset]
            )
            for model in models:
                for seed in seeds:
                    run_dir = runs_root / experiment / dataset / model / f"seed_{seed}"
                    metric_path = run_dir / "metrics.json"
                    config_path = run_dir / "config.json"
                    if not metric_path.is_file() or not config_path.is_file():
                        missing.append(run_dir)
                        continue
                    metrics = json.loads(metric_path.read_text(encoding="utf-8"))
                    config = json.loads(config_path.read_text(encoding="utf-8"))
                    training_samples = config["parameters"].get("training_samples")
                    if training_samples is None:
                        with np.load(run_dir / "split_indices.npz") as splits:
                            training_samples = len(splits["train"])
                    records.append(
                        {
                            "experiment": experiment,
                            "dataset": dataset,
                            "model": model,
                            "training_seed": seed,
                            "train_fraction": fraction,
                            "training_samples": training_samples,
                            "test_mae": metrics["test_mae"],
                            "test_rmse": metrics["test_rmse"],
                            "test_r2": metrics["test_r2"],
                        }
                    )
    if missing:
        preview = "\n".join(str(path) for path in missing[:10])
        raise FileNotFoundError(f"missing {len(missing)} sample-efficiency runs:\n{preview}")
    return pd.DataFrame.from_records(records)


def aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    return (
        raw.groupby(["dataset", "model", "train_fraction"], as_index=False)
        .agg(
            training_samples_mean=("training_samples", "mean"),
            training_seeds=("training_seed", "nunique"),
            test_mae_mean=("test_mae", "mean"),
            test_mae_std=("test_mae", "std"),
            test_rmse_mean=("test_rmse", "mean"),
            test_rmse_std=("test_rmse", "std"),
            test_r2_mean=("test_r2", "mean"),
            test_r2_std=("test_r2", "std"),
        )
        .sort_values(["dataset", "model", "train_fraction"])
    )


def paired_effects(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(0)
    for dataset, models in MODELS.items():
        baseline = "mlp" if dataset != "parkinsons_telemonitoring" else "mlp_context"
        for fraction, values in raw.loc[raw["dataset"] == dataset].groupby("train_fraction"):
            pivot = values.pivot(index="training_seed", columns="model", values="test_rmse")
            for model in models:
                if model == baseline:
                    continue
                differences = (pivot[baseline] - pivot[model]).dropna().to_numpy()
                draws = rng.choice(
                    differences, size=(10_000, len(differences)), replace=True
                ).mean(axis=1)
                rows.append(
                    {
                        "dataset": dataset,
                        "train_fraction": fraction,
                        "baseline": baseline,
                        "candidate": model,
                        "training_seed_pairs": len(differences),
                        "rmse_improvement_mean": float(differences.mean()),
                        "rmse_improvement_ci_2.5": float(np.quantile(draws, 0.025)),
                        "rmse_improvement_ci_97.5": float(np.quantile(draws, 0.975)),
                    }
                )
    return pd.DataFrame.from_records(rows)


def learning_curve_auc(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset, model), values in raw.groupby(["dataset", "model"], sort=True):
        seed_aucs = []
        for _seed, curve in values.groupby("training_seed"):
            curve = curve.sort_values("train_fraction")
            seed_aucs.append(
                float(np.trapz(curve["test_rmse"], curve["train_fraction"]))
            )
        numbers = np.asarray(seed_aucs)
        rng = np.random.default_rng(0)
        draws = rng.choice(numbers, size=(10_000, len(numbers)), replace=True).mean(axis=1)
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "training_seeds": len(numbers),
                "rmse_aulc_mean": float(numbers.mean()),
                "rmse_aulc_std": float(numbers.std(ddof=1)),
                "rmse_aulc_ci_2.5": float(np.quantile(draws, 0.025)),
                "rmse_aulc_ci_97.5": float(np.quantile(draws, 0.975)),
            }
        )
    return pd.DataFrame.from_records(rows)


def paired_aulc_effects(raw: pd.DataFrame) -> pd.DataFrame:
    """Compare learning-curve area to the prespecified dataset baseline."""

    per_seed: list[dict[str, Any]] = []
    for (dataset, model, seed), curve in raw.groupby(
        ["dataset", "model", "training_seed"], sort=True
    ):
        curve = curve.sort_values("train_fraction")
        per_seed.append(
            {
                "dataset": dataset,
                "model": model,
                "training_seed": seed,
                "rmse_aulc": float(
                    np.trapz(curve["test_rmse"], curve["train_fraction"])
                ),
            }
        )
    frame = pd.DataFrame.from_records(per_seed)
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(0)
    for dataset, models in MODELS.items():
        baseline = "mlp" if dataset != "parkinsons_telemonitoring" else "mlp_context"
        pivot = frame.loc[frame["dataset"] == dataset].pivot(
            index="training_seed", columns="model", values="rmse_aulc"
        )
        for model in models:
            if model == baseline:
                continue
            improvement = (pivot[baseline] - pivot[model]).dropna().to_numpy()
            draws = rng.choice(
                improvement, size=(10_000, len(improvement)), replace=True
            ).mean(axis=1)
            rows.append(
                {
                    "dataset": dataset,
                    "baseline": baseline,
                    "candidate": model,
                    "training_seed_pairs": len(improvement),
                    "rmse_aulc_improvement_mean": float(improvement.mean()),
                    "rmse_aulc_improvement_ci_2.5": float(
                        np.quantile(draws, 0.025)
                    ),
                    "rmse_aulc_improvement_ci_97.5": float(
                        np.quantile(draws, 0.975)
                    ),
                }
            )
    return pd.DataFrame.from_records(rows)


def plot_learning_curves(summary: pd.DataFrame, output_dir: Path) -> None:
    apply_publication_style()
    fig = plt.figure(figsize=(FULL_WIDTH, 3.05), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=(1.0, 0.18))
    axes = np.asarray([fig.add_subplot(grid[0, column]) for column in range(3)])
    legend_axis = fig.add_subplot(grid[1, :])
    legend_axis.axis("off")
    legend_entries: dict[str, Any] = {}
    for panel_index, (axis, dataset) in enumerate(zip(axes, MODELS, strict=True)):
        values = summary.loc[summary["dataset"] == dataset]
        for model in MODELS[dataset]:
            curve = values.loc[values["model"] == model].sort_values("train_fraction")
            style = model_style(model)
            x = 100.0 * curve["train_fraction"].to_numpy(dtype=float)
            mean = curve["test_rmse_mean"].to_numpy(dtype=float)
            std = curve["test_rmse_std"].to_numpy(dtype=float)
            (line,) = axis.plot(
                x, mean, label=model_label(model),
                color=style["color"], marker=style["marker"],
                linestyle=style["linestyle"], markeredgecolor="white",
                markeredgewidth=0.3,
            )
            axis.fill_between(
                x, mean - std, mean + std,
                color=style["color"], alpha=0.10, linewidth=0,
            )
            legend_entries.setdefault(model_label(model), line)
        axis.set_title(dataset.replace("_", " ").title())
        axis.set_xlabel("Training data (%)")
        if panel_index == 0:
            axis.set_ylabel("Test RMSE")
        axis.set_xticks((10, 25, 50, 100))
        axis.grid(axis="y")
        axis.set_axisbelow(True)
        panel_label(axis, chr(ord("a") + panel_index))
    legend_axis.legend(
        list(legend_entries.values()), list(legend_entries),
        loc="center", ncol=4,
        columnspacing=1.1, handlelength=2.2,
    )
    save_publication_figure(fig, output_dir, "figure4_sample_efficiency")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=RUNS_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ARTIFACTS_DIR / "reports" / "jocs_v1" / "sample_efficiency",
    )
    parser.add_argument("--training-seeds", type=int, nargs="+", default=tuple(range(5)))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    marker = args.output_dir / "sample_efficiency_raw.csv"
    if marker.exists() and not args.overwrite:
        parser.error("sample-efficiency outputs exist; pass --overwrite to replace them")

    raw = collect_rows(args.runs_root, tuple(args.training_seeds))
    summary = aggregate(raw)
    effects = paired_effects(raw)
    auc = learning_curve_auc(raw)
    auc_effects = paired_aulc_effects(raw)
    raw.to_csv(marker, index=False)
    summary.to_csv(args.output_dir / "sample_efficiency_summary.csv", index=False)
    effects.to_csv(args.output_dir / "sample_efficiency_paired_effects.csv", index=False)
    auc.to_csv(args.output_dir / "sample_efficiency_aulc.csv", index=False)
    auc_effects.to_csv(
        args.output_dir / "sample_efficiency_aulc_effects.csv", index=False
    )
    plot_learning_curves(summary, args.output_dir)
    protocol = {
        "fractions": [0.10, 0.25, 0.50, 1.0],
        "subset_seed": 0,
        "training_seeds": args.training_seeds,
        "validation_and_test": "fixed from the main split",
        "subset_nesting": "fixed random prefixes; within-subject prefixes for Parkinsons",
        "scaling": "refit on each training subset only",
        "full_data_runs": "reused from the main experiments",
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"runs": len(raw), "output_dir": str(args.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
