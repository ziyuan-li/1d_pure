"""Tests for the shared experiment artifact recorder."""

import json

import numpy as np
import pytest
import torch

from pure_resmlp.experiment import RunConfig, RunRecorder


def make_config() -> RunConfig:
    return RunConfig(
        experiment="smoke",
        dataset="toy",
        model="linear",
        seed=0,
        split_seed=42,
        parameters={"epochs": 2, "learning_rate": 0.01},
    )


def test_recorder_writes_complete_run(tmp_path) -> None:
    recorder = RunRecorder(make_config(), root=tmp_path)
    recorder.prepare()
    recorder.save_split_indices(train=[0, 1], validation=[2], test=[3])
    recorder.save_history(
        (
            {"epoch": 1, "train_loss": 1.0, "validation_loss": 1.2},
            {"epoch": 2, "train_loss": 0.5, "validation_loss": 0.7},
        )
    )
    recorder.save_predictions([1.0], [0.9], sample_indices=[3])
    recorder.save_metrics({"rmse": 0.1, "mae": 0.1})
    model = torch.nn.Linear(2, 1)
    recorder.save_checkpoint(model, epoch=2, metadata={"output_head": "real_linear"})

    manifest_path = recorder.finalize()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert set(manifest["files"]) == {
        "config.json",
        "environment.json",
        "split_indices.npz",
        "history.csv",
        "predictions.csv",
        "metrics.json",
        "checkpoint.pt",
    }


def test_recorder_refuses_to_overwrite_existing_run(tmp_path) -> None:
    first = RunRecorder(make_config(), root=tmp_path)
    first.prepare()

    second = RunRecorder(make_config(), root=tmp_path)

    with pytest.raises(FileExistsError):
        second.prepare()


def test_recorder_validates_prediction_lengths(tmp_path) -> None:
    recorder = RunRecorder(make_config(), root=tmp_path)
    recorder.prepare()

    with pytest.raises(ValueError):
        recorder.save_predictions(np.array([1.0, 2.0]), np.array([1.0]))


def test_recorder_accepts_generic_checkpoint_payload(tmp_path) -> None:
    recorder = RunRecorder(make_config(), root=tmp_path)
    recorder.prepare()

    path = recorder.save_checkpoint_payload(
        {"model_format": "custom_binary", "model_bytes": b"example"}
    )

    saved = torch.load(path, weights_only=False)
    assert saved["model_format"] == "custom_binary"
    assert saved["model_bytes"] == b"example"


def test_prediction_metadata_columns_are_saved(tmp_path) -> None:
    recorder = RunRecorder(make_config(), root=tmp_path)
    recorder.prepare()
    path = recorder.save_predictions(
        [10.0, 20.0],
        [11.0, 19.0],
        sample_indices=[100, 101],
        metadata={"subject_id": [1, 2], "test_time": [9.0, 10.0]},
    )

    rows = path.read_text(encoding="utf-8").splitlines()
    assert rows[0] == "split,sample_index,y_true,y_pred,subject_id,test_time"
    assert rows[1].endswith(",1,9.0")
