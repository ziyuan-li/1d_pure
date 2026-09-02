"""Unified JoCS training entry point for benchmark, context, and synthetic tasks."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

from pure_resmlp.data import (
    PreparedData,
    prepare_benchmark,
    prepare_bike_sharing,
    prepare_parkinsons_telemonitoring,
    prepare_synthetic_interactions,
    subset_prepared_training_data,
)
from pure_resmlp.experiment import RunConfig, RunRecorder
from pure_resmlp.models import (
    ComplexPUResNet1D,
    DCNv2Regressor,
    MLPResNet,
    RealPUResNet1D,
    build_4layer_mlp,
    build_complex_mlp,
    explicit_ordinary_context_crosses,
)
from pure_resmlp.paths import RUNS_DIR


TORCH_MODELS = {
    "mlp",
    "resmlp",
    "rv_pure",
    "cv_mlp",
    "cv_pure",
    "mlp_ordinary",
    "mlp_context",
    "mlp_explicit_cross",
    "dcnv2_context",
}
ALL_MODELS = tuple(sorted(TORCH_MODELS))
ALL_DATASETS = (
    "friedman",
    "concrete",
    "california",
    "energy",
    "airfoil",
    "superconductivity",
    "ccpp",
    "yacht",
    "qsar_aquatic",
    "auto_mpg",
    "bike_sharing",
    "parkinsons_telemonitoring",
    "synthetic_pairwise",
    "synthetic_higher_order",
)
CONTEXT_ABLATIONS = {
    "none": (),
    "bike_no_time": ("hr", "mnth"),
    "bike_no_calendar": ("workingday", "holiday", "weekday"),
    "bike_no_weather": ("season", "weathersit"),
}


@dataclass(frozen=True)
class ModelInputs:
    X_train: np.ndarray
    X_validation: np.ndarray
    X_test: np.ndarray
    feature_names: tuple[str, ...]

    @property
    def input_dim(self) -> int:
        return len(self.feature_names)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_prepared_data(args: argparse.Namespace) -> tuple[PreparedData, dict[str, Any]]:
    if args.dataset == "parkinsons_telemonitoring":
        return prepare_parkinsons_telemonitoring(), {
            "source_dataset": "UCI_189",
            "split_protocol": "within_subject_chronological_60_20_20",
            "target": "total_UPDRS",
            "excluded_other_target": "motor_UPDRS",
        }
    if args.dataset == "bike_sharing":
        return prepare_bike_sharing(), {}
    if args.dataset.startswith("synthetic_"):
        synthetic = prepare_synthetic_interactions(
            args.dataset.removeprefix("synthetic_"),
            n_samples=args.synthetic_samples,
            irrelevant_features=args.irrelevant_features,
            noise_fraction=args.noise_fraction,
            data_seed=args.data_seed,
            split_seed=args.split_seed,
        )
        return synthetic.prepared, {
            "true_pairs": synthetic.true_pairs,
            "interaction_groups": synthetic.interaction_groups,
        }
    return (
        prepare_benchmark(
            args.dataset,
            split_seed=args.split_seed,
            energy_target=args.energy_target,
        ),
        {},
    )


def prepare_model_inputs(
    data: PreparedData,
    model_name: str,
    excluded_groups: tuple[str, ...] = (),
) -> ModelInputs:
    train, names = transform_input_matrix(
        data, model_name, data.X_train, excluded_groups
    )
    validation, validation_names = transform_input_matrix(
        data, model_name, data.X_validation, excluded_groups
    )
    test, test_names = transform_input_matrix(
        data, model_name, data.X_test, excluded_groups
    )
    if names != validation_names or names != test_names:
        raise RuntimeError("model feature maps are inconsistent across splits")
    return ModelInputs(train, validation, test, names)


def transform_input_matrix(
    data: PreparedData,
    model_name: str,
    X: np.ndarray,
    excluded_groups: tuple[str, ...] = (),
) -> tuple[np.ndarray, tuple[str, ...]]:
    unknown = set(excluded_groups) - set(data.feature_groups)
    if unknown:
        raise ValueError(f"unknown excluded feature groups: {sorted(unknown)}")
    if model_name == "mlp_ordinary":
        columns = [
            index
            for group, indices in data.feature_groups.items()
            if data.feature_group_roles[group] == "ordinary"
            and group not in excluded_groups
            for index in indices
        ]
        if not columns:
            raise ValueError("mlp_ordinary requires ordinary feature groups")
        names = tuple(data.feature_names[index] for index in columns)
        return X[:, columns], names
    if model_name == "mlp_explicit_cross":
        retained_columns = [
            index
            for index in range(data.input_dim)
            if all(
                index not in data.feature_groups[group] for group in excluded_groups
            )
        ]
        old_to_new = {old: new for new, old in enumerate(retained_columns)}
        groups = {
            group: tuple(old_to_new[index] for index in indices)
            for group, indices in data.feature_groups.items()
            if group not in excluded_groups
        }
        names = tuple(data.feature_names[index] for index in retained_columns)
        return explicit_ordinary_context_crosses(
            X[:, retained_columns],
            names,
            groups,
            data.feature_group_roles,
        )
    retained_columns = [
        index
        for index in range(data.input_dim)
        if all(index not in data.feature_groups[group] for group in excluded_groups)
    ]
    if not retained_columns:
        raise ValueError("context ablation removed every input feature")
    if excluded_groups:
        return X[:, retained_columns], tuple(
            data.feature_names[index] for index in retained_columns
        )
    return X, data.feature_names


def resolve_context_ablation(dataset: str, name: str) -> tuple[str, ...]:
    excluded = CONTEXT_ABLATIONS[name]
    if name.startswith("bike_") and dataset != "bike_sharing":
        raise ValueError(f"{name} is only valid for bike_sharing")
    return excluded


def build_torch_model(model_name: str, input_dim: int, hidden_dim: int) -> nn.Module:
    if model_name in {"mlp", "mlp_ordinary", "mlp_context", "mlp_explicit_cross"}:
        return build_4layer_mlp(input_dim, 1, hidden_dim)
    if model_name == "resmlp":
        return MLPResNet(input_dim, hidden_dim, 1)
    if model_name == "rv_pure":
        return RealPUResNet1D(input_dim, hidden_dim, 1)
    if model_name == "cv_mlp":
        return build_complex_mlp(input_dim, 1, hidden_dim)
    if model_name == "cv_pure":
        return ComplexPUResNet1D(input_dim, hidden_dim, 1)
    if model_name == "dcnv2_context":
        return DCNv2Regressor(input_dim, hidden_dim, 1)
    raise ValueError(f"not a supported PyTorch model: {model_name}")


def _real_degrees_of_freedom(model: nn.Module) -> int:
    return sum(
        parameter.numel() * (2 if torch.is_complex(parameter) else 1)
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def _make_loader(
    X: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(X, dtype=np.float32)),
        torch.from_numpy(np.asarray(y, dtype=np.float32)).unsqueeze(1),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
    )


def _forward(model: nn.Module, X: torch.Tensor, complex_input: bool) -> torch.Tensor:
    if complex_input:
        X = X.to(torch.complex64)
    return model(X)


def train_torch_model(
    model: nn.Module,
    inputs: ModelInputs,
    data: PreparedData,
    *,
    device: torch.device,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    complex_input: bool,
) -> tuple[nn.Module, list[dict[str, float]], np.ndarray, dict[str, Any]]:
    model = model.to(device)
    train_loader = _make_loader(
        inputs.X_train, data.y_train, batch_size=batch_size, shuffle=True, seed=seed
    )
    validation_loader = _make_loader(
        inputs.X_validation,
        data.y_validation,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_total = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            prediction = _forward(model, X_batch, complex_input)
            loss = criterion(prediction, y_batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            train_total += float(loss.item()) * len(X_batch)

        model.eval()
        validation_total = 0.0
        with torch.no_grad():
            for X_batch, y_batch in validation_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                prediction = _forward(model, X_batch, complex_input)
                validation_total += float(criterion(prediction, y_batch).item()) * len(
                    X_batch
                )
        train_loss = train_total / len(train_loader.dataset)
        validation_loss = validation_total / len(validation_loader.dataset)
        history.append(
            {
                "epoch": float(epoch),
                "train_mse": train_loss,
                "validation_mse": validation_loss,
            }
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    predictions = predict_torch(
        model,
        inputs.X_test,
        device=device,
        batch_size=batch_size,
        complex_input=complex_input,
    )
    return model, history, predictions, {
        "best_epoch": best_epoch,
        "best_validation_mse": best_loss,
        "trainable_real_dof": _real_degrees_of_freedom(model),
    }


def predict_torch(
    model: nn.Module,
    X: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    complex_input: bool,
) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.from_numpy(np.asarray(X[start : start + batch_size], dtype=np.float32)).to(
                device
            )
            prediction = _forward(model, batch, complex_input)
            predictions.append(prediction.detach().cpu().numpy().reshape(-1))
    return np.concatenate(predictions)


def regression_metrics(
    data: PreparedData, prediction_scaled: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    prediction_raw = data.y_scaler.inverse_transform(
        np.asarray(prediction_scaled).reshape(-1, 1)
    ).reshape(-1)
    return prediction_raw, {
        "test_mse_standardized": float(
            mean_squared_error(data.y_test, prediction_scaled)
        ),
        "test_mae": float(mean_absolute_error(data.y_test_raw, prediction_raw)),
        "test_rmse": float(
            mean_squared_error(data.y_test_raw, prediction_raw) ** 0.5
        ),
        "test_r2": float(r2_score(data.y_test_raw, prediction_raw)),
    }


def grouped_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    """Report test metrics for each observed batch or gas context."""

    grouped: dict[str, dict[str, float | int]] = {}
    for group in np.unique(groups):
        mask = groups == group
        targets = y_true[mask]
        predictions = y_pred[mask]
        grouped[str(int(group))] = {
            "samples": int(mask.sum()),
            "mae": float(mean_absolute_error(targets, predictions)),
            "rmse": float(mean_squared_error(targets, predictions) ** 0.5),
            "r2": float(r2_score(targets, predictions)),
        }
    return grouped


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=ALL_DATASETS)
    parser.add_argument("--model", required=True, choices=ALL_MODELS)
    parser.add_argument("--experiment", default="jocs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--subset-seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--energy-target", default="Y1", choices=("Y1", "Y2"))
    parser.add_argument("--synthetic-samples", type=int, default=10_000)
    parser.add_argument("--irrelevant-features", type=int, default=20)
    parser.add_argument("--noise-fraction", type=float, default=0.05)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument(
        "--context-ablation",
        choices=tuple(CONTEXT_ABLATIONS),
        default="none",
    )
    parser.add_argument("--output-root", type=Path, default=RUNS_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size) <= 0:
        parser.error("epochs, patience, and batch-size must be positive")
    if not 0.0 < args.train_fraction <= 1.0:
        parser.error("train-fraction must be in (0, 1]")
    if args.subset_seed < 0:
        parser.error("subset-seed must be non-negative")
    learning_rate = args.learning_rate or 1e-3
    if learning_rate <= 0:
        parser.error("learning-rate must be positive")

    set_seed(args.seed)
    data, dataset_metadata = load_prepared_data(args)
    full_training_samples = len(data.y_train)
    subset_group = (
        "subject_id" if args.dataset == "parkinsons_telemonitoring" else None
    )
    data = subset_prepared_training_data(
        data,
        args.train_fraction,
        subset_seed=args.subset_seed,
        group_metadata_key=subset_group,
    )
    try:
        excluded_groups = resolve_context_ablation(
            args.dataset, args.context_ablation
        )
    except ValueError as error:
        parser.error(str(error))
    if excluded_groups and args.model != "rv_pure":
        parser.error("context ablations are restricted to rv_pure")
    inputs = prepare_model_inputs(data, args.model, excluded_groups)
    hidden_dim = args.hidden_dim or (64 if args.model.startswith("cv_") else 128)
    complex_input = args.model in {"cv_mlp", "cv_pure"}
    output_head = (
        "learnable_real_projection_v1" if complex_input else "real_linear"
    )
    parameters = {
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "learning_rate": learning_rate,
        "hidden_dim": hidden_dim,
        "input_dim": inputs.input_dim,
        "feature_names": inputs.feature_names,
        "target_name": data.target_name,
        "context_ablation": args.context_ablation,
        "excluded_groups": excluded_groups,
        "output_head": output_head,
        "energy_target": args.energy_target if args.dataset == "energy" else None,
        "synthetic_samples": (
            args.synthetic_samples if args.dataset.startswith("synthetic_") else None
        ),
        "irrelevant_features": (
            args.irrelevant_features if args.dataset.startswith("synthetic_") else None
        ),
        "noise_fraction": (
            args.noise_fraction if args.dataset.startswith("synthetic_") else None
        ),
        "data_seed": args.data_seed if args.dataset.startswith("synthetic_") else None,
        "train_fraction": args.train_fraction,
        "subset_seed": args.subset_seed,
        "full_training_samples": full_training_samples,
        "training_samples": len(data.y_train),
        "subset_protocol": (
            "fixed_nested_subject_stratified_prefixes"
            if subset_group is not None
            else "fixed_nested_random_prefix"
        ),
        **dataset_metadata,
    }
    config = RunConfig(
        experiment=args.experiment,
        dataset=args.dataset,
        model=args.model,
        seed=args.seed,
        split_seed=args.split_seed,
        parameters=parameters,
    )
    recorder = RunRecorder(config, root=args.output_root, overwrite=args.overwrite)
    recorder.prepare()
    recorder.save_split_indices(**data.split_indices)

    start = time.perf_counter()
    device = resolve_device(args.device)
    model = build_torch_model(args.model, inputs.input_dim, hidden_dim)
    model, history, prediction_scaled, training_info = train_torch_model(
        model,
        inputs,
        data,
        device=device,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=learning_rate,
        patience=args.patience,
        complex_input=complex_input,
    )
    training_seconds = time.perf_counter() - start

    inference_start = time.perf_counter()
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    prediction_scaled = predict_torch(
        model,
        inputs.X_test,
        device=device,
        batch_size=args.batch_size,
        complex_input=complex_input,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - inference_start
    prediction_raw, metrics = regression_metrics(data, prediction_scaled)
    prediction_metadata: dict[str, np.ndarray] = {}
    if data.sample_metadata is not None:
        prediction_metadata = {
            key: np.asarray(values)[data.test_indices]
            for key, values in data.sample_metadata.items()
        }
        if "batch_id" in prediction_metadata:
            metrics["test_by_batch"] = grouped_regression_metrics(
                data.y_test_raw,
                prediction_raw,
                prediction_metadata["batch_id"],
            )
        if "gas_id" in prediction_metadata:
            metrics["test_by_gas"] = grouped_regression_metrics(
                data.y_test_raw,
                prediction_raw,
                prediction_metadata["gas_id"],
            )
    metrics.update(training_info)
    metrics.update(
        {
            "training_seconds": training_seconds,
            "inference_seconds": inference_seconds,
            "test_samples": len(data.y_test_raw),
            "numerical_failure": False,
        }
    )
    recorder.save_history(history)
    recorder.save_predictions(
        data.y_test_raw,
        prediction_raw,
        sample_indices=data.test_indices,
        metadata=prediction_metadata,
    )
    recorder.save_metrics(metrics)
    checkpoint_metadata = {
        "input_dim": inputs.input_dim,
        "feature_names": inputs.feature_names,
        "output_head": output_head,
        "x_scaler_mean": data.x_scaler.mean_,
        "x_scaler_scale": data.x_scaler.scale_,
        "y_scaler_mean": data.y_scaler.mean_,
        "y_scaler_scale": data.y_scaler.scale_,
    }
    recorder.save_checkpoint(
        model,
        epoch=int(training_info["best_epoch"]),
        metadata=checkpoint_metadata,
    )
    recorder.finalize()
    print(json.dumps({"run_dir": str(recorder.run_dir), "metrics": metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
