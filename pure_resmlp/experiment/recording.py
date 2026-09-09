"""Experiment configurations, artifacts, and completion manifests."""

from __future__ import annotations

import csv
import importlib.metadata
import json
import platform
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from pure_resmlp.paths import RUNS_DIR


_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9_.-]+$")


def _validate_segment(label: str, value: str) -> str:
    if not value or not _SAFE_SEGMENT.fullmatch(value):
        raise ValueError(
            f"{label} must contain only letters, digits, dot, underscore, or dash"
        )
    return value


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_value(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


@dataclass(frozen=True)
class RunConfig:
    """Run identity and hyperparameters."""

    experiment: str
    dataset: str
    model: str
    seed: int
    split_seed: int
    parameters: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        _validate_segment("experiment", self.experiment)
        _validate_segment("dataset", self.dataset)
        _validate_segment("model", self.model)
        if self.seed < 0 or self.split_seed < 0:
            raise ValueError("seed and split_seed must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


class RunRecorder:
    """Write the complete artifact set for one experiment run."""

    def __init__(
        self,
        config: RunConfig,
        root: Path | str = RUNS_DIR,
        *,
        overwrite: bool = False,
    ) -> None:
        self.config = config
        self.root = Path(root)
        self.run_dir = (
            self.root
            / config.experiment
            / config.dataset
            / config.model
            / f"seed_{config.seed}"
        )
        self.overwrite = overwrite

    def prepare(self) -> Path:
        if self.run_dir.exists() and any(self.run_dir.iterdir()) and not self.overwrite:
            raise FileExistsError(
                f"run directory already contains artifacts: {self.run_dir}"
            )
        self.run_dir.mkdir(parents=True, exist_ok=True)
        _write_json(self.run_dir / "config.json", self.config.to_dict())
        _write_json(self.run_dir / "environment.json", self.capture_environment())
        return self.run_dir

    @staticmethod
    def capture_environment() -> dict[str, Any]:
        packages: dict[str, str] = {}
        for package in (
            "numpy",
            "pandas",
            "scikit-learn",
            "scipy",
            "shap",
            "torch",
        ):
            try:
                packages[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                packages[package] = "not-installed"

        return {
            "captured_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "packages": packages,
            "torch_cuda_available": torch.cuda.is_available(),
            "torch_cuda_version": torch.version.cuda,
            "cuda_devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        }

    def save_split_indices(self, **indices: Sequence[int] | np.ndarray) -> Path:
        if not indices:
            raise ValueError("at least one split index array is required")
        path = self.run_dir / "split_indices.npz"
        np.savez(path, **{name: np.asarray(values) for name, values in indices.items()})
        return path

    def save_history(self, rows: Iterable[Mapping[str, Any]]) -> Path:
        records = [dict(row) for row in rows]
        if not records:
            raise ValueError("history cannot be empty")
        fields = list(records[0])
        if any(list(record) != fields for record in records):
            raise ValueError("all history rows must use the same ordered fields")
        path = self.run_dir / "history.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(_json_value(records))
        return path

    def save_predictions(
        self,
        y_true: Sequence[float] | np.ndarray,
        y_pred: Sequence[float] | np.ndarray,
        *,
        split: str = "test",
        sample_indices: Sequence[int] | np.ndarray | None = None,
        metadata: Mapping[str, Sequence[Any] | np.ndarray] | None = None,
    ) -> Path:
        true_values = np.asarray(y_true).reshape(-1)
        predictions = np.asarray(y_pred).reshape(-1)
        if true_values.shape != predictions.shape:
            raise ValueError("y_true and y_pred must have the same number of values")
        if sample_indices is None:
            sample_indices = np.arange(true_values.size)
        sample_indices = np.asarray(sample_indices).reshape(-1)
        if sample_indices.size != true_values.size:
            raise ValueError("sample_indices must match the prediction count")
        metadata_arrays = {
            str(name): np.asarray(values).reshape(-1)
            for name, values in (metadata or {}).items()
        }
        if any(values.size != true_values.size for values in metadata_arrays.values()):
            raise ValueError("prediction metadata must match the prediction count")

        path = self.run_dir / "predictions.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            metadata_names = tuple(metadata_arrays)
            writer.writerow(
                ("split", "sample_index", "y_true", "y_pred", *metadata_names)
            )
            writer.writerows(
                (
                    split,
                    int(index),
                    float(target),
                    float(prediction),
                    *(metadata_arrays[name][row].item() for name in metadata_names),
                )
                for row, (index, target, prediction) in enumerate(
                    zip(sample_indices, true_values, predictions, strict=True)
                )
            )
        return path

    def save_metrics(self, metrics: Mapping[str, Any]) -> Path:
        path = self.run_dir / "metrics.json"
        _write_json(path, metrics)
        return path

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        epoch: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        state: dict[str, Any] = {
            "model_state_dict": {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            },
            "epoch": epoch,
            "metadata": _json_value(metadata or {}),
        }
        if optimizer is not None:
            state["optimizer_state_dict"] = optimizer.state_dict()
        return self.save_checkpoint_payload(state)

    def save_checkpoint_payload(self, state: Mapping[str, Any]) -> Path:
        """Save a non-PyTorch model state under the same checkpoint contract."""

        payload: dict[str, Any] = {
            "schema_version": self.config.schema_version,
            "config": self.config.to_dict(),
            **dict(state),
        }
        path = self.run_dir / "checkpoint.pt"
        temporary = path.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
        return path

    def finalize(self) -> Path:
        required = (
            "config.json",
            "environment.json",
            "split_indices.npz",
            "history.csv",
            "predictions.csv",
            "metrics.json",
            "checkpoint.pt",
        )
        missing = [name for name in required if not (self.run_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"run is incomplete; missing: {', '.join(missing)}")
        manifest = {
            "status": "complete",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "files": {
                name: (self.run_dir / name).stat().st_size for name in required
            },
        }
        path = self.run_dir / "manifest.json"
        _write_json(path, manifest)
        return path
