"""Tests for the strengthened robustness and sample-efficiency protocols."""

import numpy as np
import pandas as pd

from pure_resmlp.data import (
    prepare_benchmark,
    prepare_parkinsons_telemonitoring,
    subset_prepared_training_data,
)
from pure_resmlp.evaluation.evaluate_noise_robustness import (
    add_gaussian_feature_noise,
    continuous_ordinary_columns,
    summarize_auc,
    summarize_noise_rows,
)
from pure_resmlp.evaluation.summarize_jocs import paired_ablation_effects


def test_training_subsets_are_nested_and_refit_scalers() -> None:
    full = prepare_benchmark("friedman", split_seed=0)
    small = subset_prepared_training_data(full, 0.10, subset_seed=0)
    medium = subset_prepared_training_data(full, 0.25, subset_seed=0)

    assert set(small.train_indices) < set(medium.train_indices) < set(full.train_indices)
    assert np.array_equal(small.validation_indices, full.validation_indices)
    assert np.array_equal(small.test_indices, full.test_indices)
    assert np.allclose(small.X_train.mean(axis=0), 0.0, atol=1e-6)
    assert np.isclose(small.y_train.mean(), 0.0, atol=1e-6)


def test_subject_stratified_subsets_retain_every_subject() -> None:
    full = prepare_parkinsons_telemonitoring()
    small = subset_prepared_training_data(
        full, 0.10, subset_seed=0, group_metadata_key="subject_id"
    )
    medium = subset_prepared_training_data(
        full, 0.25, subset_seed=0, group_metadata_key="subject_id"
    )
    subjects = np.asarray(full.sample_metadata["subject_id"])

    assert set(subjects[small.train_indices]) == set(range(1, 43))
    assert set(small.train_indices) < set(medium.train_indices)


def test_noise_is_reproducible_and_only_changes_selected_columns() -> None:
    data = prepare_parkinsons_telemonitoring()
    columns = continuous_ordinary_columns(data)
    first = add_gaussian_feature_noise(data.X_test, columns, 0.05, seed=3)
    second = add_gaussian_feature_noise(data.X_test, columns, 0.05, seed=3)
    untouched = np.setdiff1d(np.arange(data.input_dim), columns)

    assert np.array_equal(first, second)
    assert np.array_equal(first[:, untouched], data.X_test[:, untouched])
    assert not np.array_equal(first[:, columns], data.X_test[:, columns])


def test_noise_summary_averages_corruptions_before_training_seeds() -> None:
    rows = []
    for training_seed, clean in ((0, 2.0), (1, 4.0)):
        rows.append(
            {
                "experiment": "e",
                "dataset": "d",
                "model": "m",
                "training_seed": training_seed,
                "noise_std": 0.0,
                "corruption_seed": -1,
                "test_mae": clean,
                "test_rmse": clean,
                "test_r2": 0.8,
            }
        )
        for corruption_seed, offset in ((0, 1.0), (1, 3.0)):
            rows.append(
                {
                    "experiment": "e",
                    "dataset": "d",
                    "model": "m",
                    "training_seed": training_seed,
                    "noise_std": 0.05,
                    "corruption_seed": corruption_seed,
                    "test_mae": clean + offset,
                    "test_rmse": clean + offset,
                    "test_r2": 0.7,
                }
            )
    seed_rows, summary = summarize_noise_rows(pd.DataFrame(rows))
    noisy = seed_rows.loc[seed_rows["noise_std"] == 0.05]

    assert noisy["test_rmse"].tolist() == [4.0, 6.0]
    assert summary.loc[summary["noise_std"] == 0.05, "test_rmse_mean"].item() == 5.0
    auc = summarize_auc(seed_rows)
    assert auc["performance_auc_mean"].item() > 0
    assert auc["degradation_auc_mean"].item() > 0


def test_bike_ablation_effects_are_paired_on_shared_seeds() -> None:
    context = pd.DataFrame(
        {
            "experiment": ["jocs_bike_context_v1"] * 3,
            "dataset": ["bike_sharing"] * 3,
            "model": ["rv_pure"] * 3,
            "seed": [0, 1, 2],
            "test_mae": [10.0, 20.0, 30.0],
        }
    )
    ablations = pd.DataFrame(
        {
            "context_ablation": ["bike_no_weather"] * 3,
            "seed": [0, 1, 2],
            "test_mae": [12.0, 18.0, 36.0],
        }
    )

    result = paired_ablation_effects(context, ablations)

    assert result["n_pairs"].item() == 3
    assert result["mean_ablation_minus_full"].item() == 2.0
