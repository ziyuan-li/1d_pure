"""Check that the versioned measurements reproduce the reported tables."""

from pathlib import Path

import pandas as pd
import pytest

from pure_resmlp.evaluation.plot_jocs import (
    CONTEXT_MAP_MODELS, CONTEXT_MAP_SPECIFICATIONS, mean_published_interactions,
)
from pure_resmlp.evaluation.rebuild_reports import rebuild, validate_main_records


REPORTS = Path(__file__).resolve().parents[1] / "artifacts/reports/jocs_v1"


def test_published_training_and_attribution_grids():
    runs = pd.read_csv(REPORTS / "all_runs.csv")
    resamples = pd.read_csv(REPORTS / "interaction_resamples.csv")
    validate_main_records(runs, resamples)
    assert len(runs) == 256
    assert len(resamples) == 135
    assert runs["run_dir"].str.startswith("artifacts/runs/").all()
    with pytest.raises(ValueError, match="training grid"):
        validate_main_records(runs.iloc[:-1], resamples)
    with pytest.raises(ValueError, match="attribution records"):
        validate_main_records(runs, resamples.iloc[:-1])


def test_scientific_summary_contains_all_five_models():
    summary = pd.read_csv(REPORTS / "table1_benchmark.csv")
    assert len(summary) == 30
    assert summary["n"].eq(5).all()
    for _, dataset in summary.groupby("dataset"):
        assert set(dataset["model"]) == {"mlp", "resmlp", "rv_pure", "cv_mlp", "cv_pure"}


def test_context_map_sources_have_complete_seed_coverage():
    scores = pd.read_csv(REPORTS / "context_interactions.csv")
    assert len(scores) == 1440
    for _, dataset, ordinary, context, _ in CONTEXT_MAP_SPECIFICATIONS:
        for model in CONTEXT_MAP_MODELS:
            matrix = mean_published_interactions(scores, dataset, model)
            assert set(matrix.index) == set(ordinary)
            assert set(matrix.columns) == set(context)
            assert matrix.notna().all().all()
    with pytest.raises(ValueError, match="five training seeds"):
        mean_published_interactions(scores.iloc[1:], "bike_sharing", "mlp_context")


def test_rebuild_reproduces_every_csv(tmp_path):
    output = tmp_path / "reports"
    rebuild(REPORTS, output)
    for expected_path in REPORTS.rglob("*.csv"):
        generated = pd.read_csv(output / expected_path.relative_to(REPORTS))
        expected = pd.read_csv(expected_path)
        pd.testing.assert_frame_equal(
            generated, expected, check_dtype=False, check_exact=False,
            rtol=1e-10, atol=1e-11,
        )
    with pytest.raises(ValueError, match="not empty"):
        rebuild(REPORTS, output)


def test_rebuild_does_not_replace_source_records():
    with pytest.raises(ValueError, match="separate output"):
        rebuild(REPORTS, REPORTS)
