"""Tests for synthetic data and finite-difference interaction recovery."""

import numpy as np
import pandas as pd

from pure_resmlp.data import prepare_synthetic_interactions
from pure_resmlp.evaluation.finite_difference_interactions import (
    finite_difference_interactions,
    interaction_recovery_metrics,
    summarize_recovery_resamples,
)
from pure_resmlp.evaluation.summarize_jocs import aggregate_interaction_robustness


def test_synthetic_pairwise_truth() -> None:
    data = prepare_synthetic_interactions(
        "pairwise", n_samples=500, irrelevant_features=2, data_seed=3
    )

    assert data.prepared.input_dim == 6
    assert set(data.true_pairs) == {("x1", "x2"), ("x3", "x4")}


def test_finite_difference_recovers_oracle_pair() -> None:
    rng = np.random.default_rng(4)
    X = rng.normal(size=(1_000, 3))

    def oracle(values: np.ndarray) -> np.ndarray:
        return values[:, 0] * values[:, 1] + values[:, 2]

    result = finite_difference_interactions(
        oracle,
        X[:500],
        X[500:],
        {"x1": (0,), "x2": (1,), "x3": (2,)},
        n_samples=256,
        seed=5,
    )
    metrics = interaction_recovery_metrics(result, (("x1", "x2"),), k=1)

    assert result.ranked_pairs()[0][:2] == ("x1", "x2")
    assert metrics["precision_at_k"] == 1.0
    assert metrics["recall_at_k"] == 1.0
    assert metrics["true_interaction_mean_rank"] == 1.0


def test_recovery_resample_summary_separates_sampling_variability() -> None:
    records = [
        {
            "precision_at_k": 1.0,
            "recall_at_k": 1.0,
            "true_interaction_mean_rank": 1.5,
        },
        {
            "precision_at_k": 0.5,
            "recall_at_k": 0.5,
            "true_interaction_mean_rank": 2.5,
        },
    ]

    summary = summarize_recovery_resamples(records)

    assert summary["resamples"] == 2
    assert summary["metrics"]["precision_at_k"]["mean"] == 0.75
    assert summary["metrics"]["true_interaction_mean_rank"]["max"] == 2.5
    assert summary["exact_recovery_rate"] == 0.5


def test_aggregate_interaction_robustness_keeps_resamples_distinct() -> None:
    frame = pd.DataFrame(
        {
            "dataset": ["synthetic_pairwise"] * 4,
            "model": ["mlp"] * 4,
            "training_seed": [0, 0, 1, 1],
            "sampling_seed": [0, 1, 0, 1],
            "precision_at_k": [1.0, 1.0, 1.0, 0.5],
            "recall_at_k": [1.0, 1.0, 1.0, 0.5],
            "true_interaction_mean_rank": [1.5, 1.5, 1.5, 2.5],
        }
    )

    row = aggregate_interaction_robustness(frame).iloc[0]

    assert row["training_runs"] == 2
    assert row["attribution_resamples"] == 4
    assert row["precision_at_k_mean"] == 0.875
    assert row["exact_recovery_rate"] == 0.75
