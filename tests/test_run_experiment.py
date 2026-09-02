"""Unit tests for unified experiment model/input dispatch."""

import json
from pathlib import Path

from pure_resmlp.data import (
    prepare_bike_sharing,
    prepare_parkinsons_telemonitoring,
)
from pure_resmlp.training.run_experiment import (
    build_torch_model,
    prepare_model_inputs,
    resolve_context_ablation,
    transform_input_matrix,
)
from pure_resmlp.training.run_cv_baselines import (
    BIKE_EXPERIMENT,
    Job,
    _matches_frozen_protocol,
    jobs,
)


def test_bike_model_input_views() -> None:
    data = prepare_bike_sharing()
    ordinary = prepare_model_inputs(data, "mlp_ordinary")
    context = prepare_model_inputs(data, "mlp_context")
    crossed = prepare_model_inputs(data, "mlp_explicit_cross")

    assert ordinary.input_dim == 4
    assert context.input_dim == data.input_dim
    assert crossed.input_dim > context.input_dim
    transformed, transformed_names = transform_input_matrix(
        data, "mlp_explicit_cross", data.X_test[:3]
    )
    assert transformed.shape == (3, crossed.input_dim)
    assert transformed_names == crossed.feature_names


def test_all_torch_model_builders() -> None:
    names = (
        "mlp",
        "resmlp",
        "rv_pure",
        "cv_mlp",
        "cv_pure",
        "dcnv2_context",
    )

    for name in names:
        model = build_torch_model(name, input_dim=8, hidden_dim=8)
        assert model is not None


def test_parkinsons_model_input_views() -> None:
    parkinsons = prepare_parkinsons_telemonitoring()

    assert prepare_model_inputs(parkinsons, "mlp_ordinary").input_dim == 16
    assert prepare_model_inputs(parkinsons, "mlp_context").input_dim == 61


def test_context_ablation_input_views() -> None:
    bike = prepare_bike_sharing()

    assert prepare_model_inputs(
        bike, "rv_pure", resolve_context_ablation("bike_sharing", "bike_no_time")
    ).input_dim == bike.input_dim - 4
    assert prepare_model_inputs(
        bike,
        "rv_pure",
        resolve_context_ablation("bike_sharing", "bike_no_calendar"),
    ).input_dim == bike.input_dim - 9
    assert prepare_model_inputs(
        bike, "rv_pure", resolve_context_ablation("bike_sharing", "bike_no_weather")
    ).input_dim == bike.input_dim - 8


def test_cv_baseline_grid_is_compact_and_balanced() -> None:
    grid = jobs()

    assert len(grid) == 18
    assert sum(
        job.dataset == "synthetic_higher_order" and job.model == "cv_mlp"
        for job in grid
    ) == 3
    assert sum(
        job.dataset == "parkinsons_telemonitoring" and job.model == "cv_mlp"
        for job in grid
    ) == 5
    assert sum(job.dataset == "bike_sharing" and job.model == "cv_mlp" for job in grid) == 5
    assert sum(job.dataset == "bike_sharing" and job.model == "cv_pure" for job in grid) == 5


def test_cv_baseline_reuse_requires_the_frozen_protocol(tmp_path: Path) -> None:
    job = Job("bike_sharing", 0, "cv_pure", BIKE_EXPERIMENT)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text("{}", encoding="utf-8")
    config = {
        "experiment": BIKE_EXPERIMENT,
        "dataset": "bike_sharing",
        "model": "cv_pure",
        "seed": 0,
        "split_seed": 0,
        "parameters": {
            "hidden_dim": 64,
            "epochs": 100,
            "patience": 15,
            "batch_size": 256,
            "output_head": "learnable_real_projection_v1",
        },
    }
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

    assert _matches_frozen_protocol(run_dir, job)
    config["parameters"]["batch_size"] = 128
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    assert not _matches_frozen_protocol(run_dir, job)
