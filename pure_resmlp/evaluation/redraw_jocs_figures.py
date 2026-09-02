"""Redraw all JoCS figures from frozen report tables and run artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pure_resmlp.evaluation.evaluate_noise_robustness import _plot as plot_robustness
from pure_resmlp.evaluation.plot_jocs import (
    figure1_synthetic,
    figure2_context,
)
from pure_resmlp.evaluation.plot_style import apply_publication_style
from pure_resmlp.evaluation.summarize_sample_efficiency import plot_learning_curves
from pure_resmlp.paths import ARTIFACTS_DIR, RUNS_DIR


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=RUNS_DIR)
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=ARTIFACTS_DIR / "reports" / "jocs_v1",
    )
    args = parser.parse_args()

    robustness_dir = args.report_dir / "robustness"
    sample_dir = args.report_dir / "sample_efficiency"
    apply_publication_style()
    figure1_synthetic(args.report_dir)
    figure2_context(args.runs_root, args.report_dir)
    plot_robustness(
        pd.read_csv(robustness_dir / "noise_robustness_summary.csv"),
        robustness_dir,
    )
    plot_learning_curves(
        pd.read_csv(sample_dir / "sample_efficiency_summary.csv"),
        sample_dir,
    )
    print("redrawn four JoCS figures from frozen results (PDF, PNG, TIFF)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
