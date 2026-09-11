"""Rebuild paper result tables from the small, versioned per-seed records."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import pandas as pd

from pure_resmlp.evaluation import summarize_jocs as main_results
from pure_resmlp.evaluation import summarize_sample_efficiency as sample
from pure_resmlp.evaluation.benchmark_inference import summarize_timings
from pure_resmlp.evaluation.evaluate_noise_robustness import (
    summarize_auc, summarize_auc_effects, summarize_noise_rows, summarize_overall_auc,
)
from pure_resmlp.paths import ARTIFACTS_DIR


def validate_main_records(runs: pd.DataFrame, resamples: pd.DataFrame) -> None:
    """Check the published training grid and its nested attribution repetitions."""
    counts = runs.groupby("experiment").size().to_dict()
    if counts != main_results.EXPECTED_MANIFESTS:
        raise ValueError(f"unexpected training grid: {counts}")
    if runs.duplicated(["experiment", "dataset", "model", "seed"]).any():
        raise ValueError("duplicate training run")
    if not runs["split_seed"].eq(0).all():
        raise ValueError("published runs use split seed 0")
    scientific = runs[runs["experiment"].isin(main_results.SCIENTIFIC_EXPERIMENTS)]
    expected = {
        (dataset, model, seed)
        for dataset in main_results.BENCHMARK_DATASETS
        for model in (*main_results.BENCHMARK_MODELS, "cv_mlp")
        for seed in range(5)
    }
    if set(scientific[["dataset", "model", "seed"]].itertuples(
        index=False, name=None
    )) != expected:
        raise ValueError("scientific comparison needs six datasets and five models")
    synthetic = runs[runs["dataset"].str.startswith("synthetic_")]
    expected_resamples = {
        (row.dataset, row.model, row.seed, sampling_seed)
        for row in synthetic.itertuples()
        for sampling_seed in range(5)
    }
    keys = ["dataset", "model", "training_seed", "sampling_seed"]
    if (len(resamples) != len(expected_resamples)
            or set(resamples[keys].itertuples(index=False, name=None)) != expected_resamples):
        raise ValueError("attribution records do not match the synthetic checkpoints")


def rebuild(source: Path, output: Path) -> None:
    if source.resolve() == output.resolve():
        raise ValueError("choose a separate output directory")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory is not empty: {output}")
    runs = pd.read_csv(source / "all_runs.csv")
    resamples = pd.read_csv(source / "interaction_resamples.csv")
    validate_main_records(runs, resamples)
    main_results.write_outputs(runs, output, interaction_resamples=resamples)
    for name in ("context_interactions.csv", "environment.json"):
        shutil.copy2(source / name, output / name)

    for folder, raw_name in (
        ("robustness", "noise_robustness_raw.csv"),
        ("sample_efficiency", "sample_efficiency_raw.csv"),
        ("inference_efficiency", "inference_timings_raw.csv"),
    ):
        destination = output / folder
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / folder / raw_name, destination / raw_name)
        shutil.copy2(source / folder / "protocol.json", destination / "protocol.json")

    noise_dir = output / "robustness"
    seed_rows, noise_summary = summarize_noise_rows(
        pd.read_csv(noise_dir / "noise_robustness_raw.csv")
    )
    noise_tables = {
        "noise_robustness_training_seed.csv": seed_rows,
        "noise_robustness_summary.csv": noise_summary,
        "noise_robustness_auc.csv": summarize_auc(seed_rows),
        "noise_robustness_overall_auc.csv": summarize_overall_auc(seed_rows),
        "noise_robustness_auc_effects.csv": summarize_auc_effects(seed_rows),
    }
    for name, table in noise_tables.items():
        table.to_csv(noise_dir / name, index=False)

    sample_dir = output / "sample_efficiency"
    raw = pd.read_csv(sample_dir / "sample_efficiency_raw.csv")
    sample_tables = {
        "sample_efficiency_summary.csv": sample.aggregate(raw),
        "sample_efficiency_paired_effects.csv": sample.paired_effects(raw),
        "sample_efficiency_aulc.csv": sample.learning_curve_auc(raw),
        "sample_efficiency_aulc_effects.csv": sample.paired_aulc_effects(raw),
    }
    for name, table in sample_tables.items():
        table.to_csv(sample_dir / name, index=False)

    timing_dir = output / "inference_efficiency"
    summarize_timings(pd.read_csv(timing_dir / "inference_timings_raw.csv")).to_csv(
        timing_dir / "inference_timings_summary.csv", index=False
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path,
                        default=ARTIFACTS_DIR / "reports" / "jocs_v1")
    parser.add_argument("--output-dir", type=Path,
                        default=ARTIFACTS_DIR / "reports" / "reproduced")
    args = parser.parse_args()
    rebuild(args.source_dir, args.output_dir)
    print(f"Rebuilt result tables in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
