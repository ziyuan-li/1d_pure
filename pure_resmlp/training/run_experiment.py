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
from pure_resmlp.data import PreparedData, prepare_benchmark, prepare_bike_sharing, prepare_parkinsons_telemonitoring, prepare_synthetic_interactions, subset_prepared_training_data
from pure_resmlp.experiment import RunConfig, RunRecorder
from pure_resmlp.models import ComplexPUResNet1D, DCNv2Regressor, MLPResNet, RealPUResNet1D, build_mlp, build_complex_mlp, explicit_ordinary_context_crosses
from pure_resmlp.paths import RUNS_DIR
CONTEXT_ABLATIONS = {'none': (), 'bike_no_time': ('hr', 'mnth'), 'bike_no_calendar': ('workingday', 'holiday', 'weekday'), 'bike_no_weather': ('season', 'weathersit')}

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

def prepare_model_inputs(data: PreparedData, model_name: str, excluded_groups: tuple[str, ...]=()) -> ModelInputs:
    train, names = transform_input_matrix(data, model_name, data.X_train, excluded_groups)
    validation, validation_names = transform_input_matrix(data, model_name, data.X_validation, excluded_groups)
    test, test_names = transform_input_matrix(data, model_name, data.X_test, excluded_groups)
    if names != validation_names or names != test_names:
        raise RuntimeError('model feature maps are inconsistent across splits')
    return ModelInputs(train, validation, test, names)

def transform_input_matrix(data: PreparedData, model_name: str, X: np.ndarray, excluded_groups: tuple[str, ...]=()) -> tuple[np.ndarray, tuple[str, ...]]:
    unknown = set(excluded_groups) - set(data.feature_groups)
    if unknown:
        raise ValueError(f'unknown excluded feature groups: {sorted(unknown)}')
    if model_name == 'mlp_ordinary':
        columns = [index for group, indices in data.feature_groups.items() if data.feature_group_roles[group] == 'ordinary' and group not in excluded_groups for index in indices]
        if not columns:
            raise ValueError('mlp_ordinary requires ordinary feature groups')
        names = tuple((data.feature_names[index] for index in columns))
        return (X[:, columns], names)
    if model_name == 'mlp_explicit_cross':
        retained_columns = [index for index in range(data.input_dim) if all((index not in data.feature_groups[group] for group in excluded_groups))]
        old_to_new = {old: new for new, old in enumerate(retained_columns)}
        groups = {group: tuple((old_to_new[index] for index in indices)) for group, indices in data.feature_groups.items() if group not in excluded_groups}
        names = tuple((data.feature_names[index] for index in retained_columns))
        return explicit_ordinary_context_crosses(X[:, retained_columns], names, groups, data.feature_group_roles)
    retained_columns = [index for index in range(data.input_dim) if all((index not in data.feature_groups[group] for group in excluded_groups))]
    if not retained_columns:
        raise ValueError('context ablation removed every input feature')
    if excluded_groups:
        return (X[:, retained_columns], tuple((data.feature_names[index] for index in retained_columns)))
    return (X, data.feature_names)

def resolve_context_ablation(dataset: str, name: str) -> tuple[str, ...]:
    excluded = CONTEXT_ABLATIONS[name]
    if name.startswith('bike_') and dataset != 'bike_sharing':
        raise ValueError(f'{name} is only valid for bike_sharing')
    return excluded

def build_torch_model(model_name: str, input_dim: int, hidden_dim: int) -> nn.Module:
    if model_name in {'mlp', 'mlp_ordinary', 'mlp_context', 'mlp_explicit_cross'}:
        return build_mlp(input_dim, 1, hidden_dim)
    if model_name == 'resmlp':
        return MLPResNet(input_dim, hidden_dim, 1)
    if model_name == 'rv_pure':
        return RealPUResNet1D(input_dim, hidden_dim, 1)
    if model_name == 'cv_mlp':
        return build_complex_mlp(input_dim, 1, hidden_dim)
    if model_name == 'cv_pure':
        return ComplexPUResNet1D(input_dim, hidden_dim, 1)
    if model_name == 'dcnv2_context':
        return DCNv2Regressor(input_dim, hidden_dim, 1)
    raise ValueError(f'not a supported PyTorch model: {model_name}')

def _real_degrees_of_freedom(model: nn.Module) -> int:
    return sum((parameter.numel() * (2 if torch.is_complex(parameter) else 1) for parameter in model.parameters() if parameter.requires_grad))

def _make_loader(X: np.ndarray, y: np.ndarray, *, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(torch.from_numpy(np.asarray(X, dtype=np.float32)), torch.from_numpy(np.asarray(y, dtype=np.float32)).unsqueeze(1))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator if shuffle else None)

def _forward(model: nn.Module, X: torch.Tensor, complex_input: bool) -> torch.Tensor:
    if complex_input:
        X = X.to(torch.complex64)
    return model(X)

def train_torch_model(model: nn.Module, inputs: ModelInputs, data: PreparedData, *, device: torch.device, seed: int, epochs: int, batch_size: int, learning_rate: float, patience: int, complex_input: bool, max_grad_norm: float | None=None) -> tuple[nn.Module, list[dict[str, float]], np.ndarray, dict[str, Any]]:
    model = model.to(device)
    train_loader = _make_loader(inputs.X_train, data.y_train, batch_size=batch_size, shuffle=True, seed=seed)
    validation_loader = _make_loader(inputs.X_validation, data.y_validation, batch_size=batch_size, shuffle=False, seed=seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    best_loss = float('inf')
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_total = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = (X_batch.to(device), y_batch.to(device))
            optimizer.zero_grad()
            prediction = _forward(model, X_batch, complex_input)
            loss = criterion(prediction, y_batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'non-finite training loss at epoch {epoch}')
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            train_total += float(loss.item()) * len(X_batch)
        model.eval()
        validation_total = 0.0
        with torch.no_grad():
            for X_batch, y_batch in validation_loader:
                X_batch, y_batch = (X_batch.to(device), y_batch.to(device))
                prediction = _forward(model, X_batch, complex_input)
                validation_total += float(criterion(prediction, y_batch).item()) * len(X_batch)
        train_loss = train_total / len(train_loader.dataset)
        validation_loss = validation_total / len(validation_loader.dataset)
        if not np.isfinite(validation_loss):
            raise FloatingPointError(f'non-finite validation loss at epoch {epoch}')
        history.append({'epoch': float(epoch), 'train_mse': train_loss, 'validation_mse': validation_loss})
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_state is None:
        raise RuntimeError('training did not produce a valid checkpoint')
    model.load_state_dict(best_state)
    predictions = predict_torch(model, inputs.X_test, device=device, batch_size=batch_size, complex_input=complex_input)
    return (model, history, predictions, {'best_epoch': best_epoch, 'best_validation_mse': best_loss, 'trainable_real_dof': _real_degrees_of_freedom(model)})

def predict_torch(model: nn.Module, X: np.ndarray, *, device: torch.device, batch_size: int, complex_input: bool) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.from_numpy(np.asarray(X[start:start + batch_size], dtype=np.float32)).to(device)
            prediction = _forward(model, batch, complex_input)
            predictions.append(prediction.detach().cpu().numpy().reshape(-1))
    return np.concatenate(predictions)

def regression_metrics(data: PreparedData, prediction_scaled: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    prediction_raw = data.y_scaler.inverse_transform(np.asarray(prediction_scaled).reshape(-1, 1)).reshape(-1)
    return (prediction_raw, {'test_mse_standardized': float(mean_squared_error(data.y_test, prediction_scaled)), 'test_mae': float(mean_absolute_error(data.y_test_raw, prediction_raw)), 'test_rmse': float(mean_squared_error(data.y_test_raw, prediction_raw) ** 0.5), 'test_r2': float(r2_score(data.y_test_raw, prediction_raw))})
