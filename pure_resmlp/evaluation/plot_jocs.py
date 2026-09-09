"""Generate the main JoCS result figures."""

from __future__ import annotations

import argparse
from pathlib import Path

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


MODEL_ORDER = ("mlp", "resmlp", "rv_pure", "cv_pure")


def _point_range_panel(
    axis: plt.Axes,
    subset: pd.DataFrame,
    models: tuple[str, ...],
    mean_column: str,
    std_column: str,
    xlabel: str,
) -> None:
    positions = np.arange(len(models))
    for position, model in zip(positions, models, strict=True):
        style = model_style(model)
        mean = float(subset.loc[model, mean_column])
        std = float(subset.loc[model, std_column])
        axis.errorbar(
            mean, position, xerr=std,
            fmt=style["marker"], color=style["color"],
            markerfacecolor=style["color"], markeredgecolor="white",
            markeredgewidth=0.35, elinewidth=0.8, capsize=2.0, zorder=3,
        )
    axis.set_yticks(positions, [model_label(model) for model in models])
    axis.invert_yaxis()
    axis.set_xlabel(xlabel)
    axis.grid(axis="x")
    axis.set_axisbelow(True)
    axis.margins(x=0.12, y=0.20)


def figure1_synthetic(report_dir: Path) -> None:
    frame = pd.read_csv(report_dir / "table3_synthetic.csv")
    datasets = ("synthetic_pairwise", "synthetic_higher_order")
    titles = ("Pairwise task", "Higher-order task")
    fig, axes = plt.subplots(2, 2, figsize=(FULL_WIDTH, 5.25), constrained_layout=True)

    for column, (dataset, title) in enumerate(zip(datasets, titles, strict=True)):
        models = (
            (*MODEL_ORDER[:-1], "cv_mlp", MODEL_ORDER[-1])
            if dataset == "synthetic_higher_order" else MODEL_ORDER
        )
        subset = frame.loc[frame["dataset"] == dataset].set_index("model").loc[list(models)]
        _point_range_panel(
            axes[0, column], subset, models,
            "test_rmse_mean", "test_rmse_std", "Test RMSE ↓",
        )
        _point_range_panel(
            axes[1, column], subset, models,
            "true_interaction_mean_rank_mean", "true_interaction_mean_rank_std",
            "Mean rank of true interactions ↓",
        )
        axes[0, column].set_title(title, loc="left", fontweight="bold")
        axes[1, column].set_title(title, loc="left", fontweight="bold")

    for label, axis in zip("abcd", axes.flat, strict=True):
        panel_label(axis, label)
    save_publication_figure(fig, report_dir, "figure1_synthetic_recovery")


def _mean_interaction_matrix(
    runs_root: Path, experiment: str, dataset: str, model: str,
) -> pd.DataFrame:
    matrices: list[pd.DataFrame] = []
    for seed in range(5):
        path = runs_root / experiment / dataset / model / f"seed_{seed}" / "interaction_matrix.csv"
        matrix = pd.read_csv(path, index_col=0)
        matrix.index.name = None
        matrices.append(matrix)
    values = np.mean([matrix.to_numpy(dtype=np.float64) for matrix in matrices], axis=0)
    return pd.DataFrame(values, index=matrices[0].index, columns=matrices[0].columns)


def figure2_context(runs_root: Path, report_dir: Path) -> None:
    specifications = (
        (
            "jocs_bike_context_v1", "bike_sharing",
            ["temp", "atemp", "hum", "windspeed"],
            ["hr", "mnth", "season", "weathersit", "weekday", "workingday", "holiday", "yr"],
            "Bike Sharing",
        ),
        (
            "jocs_parkinsons_context_v1", "parkinsons_telemonitoring",
            [
                "Jitter(%)", "Jitter(Abs)", "Jitter:RAP", "Jitter:PPQ5", "Jitter:DDP",
                "Shimmer", "Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
                "Shimmer:APQ11", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "PPE",
            ],
            ["subject", "age", "sex", "test_time"],
            "Parkinsons Telemonitoring",
        ),
    )
    compared_models = ("mlp_context", "rv_pure", "cv_pure")
    fig, axes = plt.subplots(2, 3, figsize=(FULL_WIDTH, 6.05), constrained_layout=True)

    for row, (experiment, dataset, ordinary, context, dataset_label) in enumerate(specifications):
        selected = {
            model: _mean_interaction_matrix(runs_root, experiment, dataset, model).loc[ordinary, context]
            for model in compared_models
        }
        vmax = max(float(matrix.to_numpy().max()) for matrix in selected.values())
        images = []
        for column, model in enumerate(compared_models):
            axis = axes[row, column]
            matrix = selected[model]
            image = axis.imshow(
                matrix.to_numpy(), cmap="cividis", aspect="auto",
                vmin=0.0, vmax=vmax, interpolation="nearest",
            )
            images.append(image)
            axis.set_xticks(range(len(context)), context, rotation=45, ha="right", rotation_mode="anchor")
            axis.set_yticks(range(len(ordinary)))
            if column == 0:
                axis.set_yticklabels(ordinary)
                axis.set_ylabel("Ordinary feature")
            else:
                axis.set_yticklabels([])
                axis.tick_params(axis="y", length=0)
            axis.set_xlabel("Context feature")
            if row == 0:
                axis.set_title(model_label(model), fontweight="bold")
            panel_label(axis, chr(ord("a") + row * 3 + column))
        axes[row, 0].text(
            0.0, 1.17, dataset_label, transform=axes[row, 0].transAxes,
            fontsize=8.5, fontweight="bold", ha="left", va="bottom",
        )
        colorbar = fig.colorbar(images[-1], ax=axes[row, :], fraction=0.025, pad=0.018)
        colorbar.set_label("Interaction strength", fontsize=7.5)
        colorbar.ax.tick_params(labelsize=6.5, width=0.5, length=2.5)

    save_publication_figure(fig, report_dir, "figure2_context_interactions")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    apply_publication_style()
    figure1_synthetic(args.report_dir)
    figure2_context(args.runs_root, args.report_dir)
    print("generated publication-ready figures 1-2 (PDF, PNG, TIFF)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
