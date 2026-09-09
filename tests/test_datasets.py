"""Validation tests for the unified JoCS data pipeline."""

import numpy as np
import pytest

from pure_resmlp.data import (
    load_raw_dataset,
    prepare_benchmark,
    prepare_bike_sharing,
    prepare_parkinsons_telemonitoring,
)


@pytest.mark.parametrize(
    ("name", "rows", "features"),
    (
        ("friedman", 10_000, 10),
        ("concrete", 1_030, 8),
        ("california", 20_640, 8),
        ("energy", 768, 8),
        ("yacht", 308, 6),
        ("auto_mpg", 392, 9),
    ),
)
def test_raw_benchmark_shapes(name: str, rows: int, features: int) -> None:
    dataset = load_raw_dataset(name)

    assert dataset.X.shape == (rows, features)
    assert dataset.y.shape == (rows,)
    assert np.isfinite(dataset.X).all()
    assert np.isfinite(dataset.y).all()


def test_training_dataset_choices_cover_the_study_tasks() -> None:
    from pure_resmlp.evaluation.summarize_jocs import BENCHMARK_DATASETS
    from pure_resmlp.training.run_experiment import ALL_DATASETS

    assert set(ALL_DATASETS) == {
        *BENCHMARK_DATASETS,
        "bike_sharing",
        "parkinsons_telemonitoring",
        "synthetic_pairwise",
        "synthetic_higher_order",
    }


def test_benchmark_split_is_complete_disjoint_and_standardized() -> None:
    prepared = prepare_benchmark("yacht", split_seed=0)
    splits = tuple(prepared.split_indices.values())

    assert sum(len(split) for split in splits) == 308
    assert len(np.unique(np.concatenate(splits))) == 308
    assert np.allclose(
        prepared.X_train.astype(np.float64).mean(axis=0), 0.0, atol=5e-5
    )
    assert np.isclose(prepared.y_train.mean(), 0.0, atol=1e-6)


def test_benchmark_split_seed_is_reproducible_and_changes_partition() -> None:
    split_zero = prepare_benchmark("yacht", split_seed=0)
    split_one_a = prepare_benchmark("yacht", split_seed=1)
    split_one_b = prepare_benchmark("yacht", split_seed=1)

    assert np.array_equal(split_one_a.train_indices, split_one_b.train_indices)
    assert not np.array_equal(split_zero.train_indices, split_one_a.train_indices)


def test_bike_pipeline_is_chronological_grouped_and_leakage_safe() -> None:
    prepared = prepare_bike_sharing()

    assert prepared.train_indices[-1] < prepared.validation_indices[0]
    assert prepared.validation_indices[-1] < prepared.test_indices[0]
    assert not {"casual", "registered", "instant", "dteday"}.intersection(
        prepared.feature_groups
    )
    assert prepared.feature_group_roles["temp"] == "ordinary"
    assert prepared.feature_group_roles["hr"] == "context"
    assert len(prepared.feature_groups["hr"]) == 2
    assert np.allclose(
        prepared.X_train.astype(np.float64).mean(axis=0), 0.0, atol=5e-5
    )
    assert np.isclose(prepared.y_train.mean(), 0.0, atol=1e-6)


def test_parkinsons_pipeline_predicts_future_within_every_subject() -> None:
    prepared = prepare_parkinsons_telemonitoring()
    metadata = prepared.sample_metadata

    assert metadata is not None
    assert prepared.input_dim == 61
    assert sum(len(split) for split in prepared.split_indices.values()) == 5_875
    assert "motor_UPDRS" not in prepared.feature_names
    assert prepared.feature_group_roles["Jitter(%)"] == "ordinary"
    assert prepared.feature_group_roles["subject"] == "context"
    assert len(prepared.feature_groups["subject"]) == 42
    for split in prepared.split_indices.values():
        assert set(metadata["subject_id"][split]) == set(range(1, 43))
    for subject in range(1, 43):
        train_time = metadata["test_time"][prepared.train_indices][
            metadata["subject_id"][prepared.train_indices] == subject
        ]
        validation_time = metadata["test_time"][prepared.validation_indices][
            metadata["subject_id"][prepared.validation_indices] == subject
        ]
        test_time = metadata["test_time"][prepared.test_indices][
            metadata["subject_id"][prepared.test_indices] == subject
        ]
        assert train_time.max() <= validation_time.min()
        assert validation_time.max() <= test_time.min()
