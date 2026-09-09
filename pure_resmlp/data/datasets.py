"""Dataset loading, splitting, and training-only preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pure_resmlp.paths import DATA_DIR


RAW_DIR = DATA_DIR / "raw"


@dataclass(frozen=True)
class RawDataset:
    name: str
    X: np.ndarray
    y: np.ndarray
    feature_names: tuple[str, ...]
    target_name: str
    train_fraction: float
    validation_fraction: float


@dataclass(frozen=True)
class PreparedData:
    name: str
    X_train: np.ndarray
    X_validation: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_validation: np.ndarray
    y_test: np.ndarray
    y_train_raw: np.ndarray
    y_validation_raw: np.ndarray
    y_test_raw: np.ndarray
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    feature_names: tuple[str, ...]
    target_name: str
    feature_groups: Mapping[str, tuple[int, ...]]
    feature_group_roles: Mapping[str, str]
    x_scaler: StandardScaler
    y_scaler: StandardScaler
    encoder: Any | None = None
    sample_metadata: Mapping[str, np.ndarray] | None = None

    @property
    def input_dim(self) -> int:
        return len(self.feature_names)

    @property
    def split_indices(self) -> dict[str, np.ndarray]:
        return {
            "train": self.train_indices,
            "validation": self.validation_indices,
            "test": self.test_indices,
        }


def _as_float_arrays(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if X.ndim != 2 or X.shape[0] != y.size:
        raise ValueError("X and y must be aligned two-dimensional tabular data")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("dataset contains missing or non-finite values")
    return X, y


def _load_friedman() -> RawDataset:
    X, y = make_friedman1(
        n_samples=10_000,
        n_features=10,
        noise=0.0,
        random_state=0,
    )
    return RawDataset(
        name="friedman",
        X=X,
        y=y,
        feature_names=tuple(f"x{i}" for i in range(10)),
        target_name="y",
        train_fraction=0.70,
        validation_fraction=0.15,
    )


def _load_concrete() -> RawDataset:
    path = DATA_DIR / "concrete+compressive+strength" / "Concrete_Data.xls"
    frame = pd.read_excel(path)
    X, y = _as_float_arrays(frame.iloc[:, :8].to_numpy(), frame.iloc[:, 8].to_numpy())
    return RawDataset(
        name="concrete",
        X=X,
        y=y,
        feature_names=tuple(str(name).strip() for name in frame.columns[:8]),
        target_name=str(frame.columns[8]).strip(),
        train_fraction=0.60,
        validation_fraction=0.20,
    )


def _load_california() -> RawDataset:
    path = RAW_DIR / "california_housing" / "CaliforniaHousing" / "cal_housing.data"
    raw = np.loadtxt(path, delimiter=",")
    households = raw[:, 6]
    if np.any(households <= 0):
        raise ValueError("California Housing contains non-positive household counts")
    X = np.column_stack(
        (
            raw[:, 7],
            raw[:, 2],
            raw[:, 3] / households,
            raw[:, 4] / households,
            raw[:, 5],
            raw[:, 5] / households,
            raw[:, 1],
            raw[:, 0],
        )
    )
    y = raw[:, 8] / 100_000.0
    X, y = _as_float_arrays(X, y)
    return RawDataset(
        name="california",
        X=X,
        y=y,
        feature_names=(
            "MedInc",
            "HouseAge",
            "AveRooms",
            "AveBedrms",
            "Population",
            "AveOccup",
            "Latitude",
            "Longitude",
        ),
        target_name="MedHouseVal",
        train_fraction=0.80,
        validation_fraction=0.10,
    )


def _load_energy(target: str) -> RawDataset:
    if target not in {"Y1", "Y2"}:
        raise ValueError("Energy target must be Y1 or Y2")
    path = RAW_DIR / "energy_efficiency" / "ENB2012_data.xlsx"
    frame = pd.read_excel(path).dropna(axis=0, how="all").dropna(axis=1, how="all")
    X, y = _as_float_arrays(frame.iloc[:, :8].to_numpy(), frame[target].to_numpy())
    return RawDataset(
        name="energy",
        X=X,
        y=y,
        feature_names=tuple(str(name) for name in frame.columns[:8]),
        target_name=target,
        train_fraction=0.70,
        validation_fraction=0.15,
    )


def _load_yacht_hydrodynamics() -> RawDataset:
    path = RAW_DIR / "yacht_hydrodynamics" / "yacht_hydrodynamics.data"
    frame = pd.read_csv(path, sep=r"\s+", header=None)
    if frame.shape != (308, 7):
        raise ValueError("Yacht Hydrodynamics must contain 308 rows and 7 columns")
    feature_names = (
        "longitudinal_position_center_buoyancy",
        "prismatic_coefficient",
        "length_displacement_ratio",
        "beam_draught_ratio",
        "length_beam_ratio",
        "froude_number",
    )
    X, y = _as_float_arrays(frame.iloc[:, :6].to_numpy(), frame.iloc[:, 6].to_numpy())
    return RawDataset(
        name="yacht",
        X=X,
        y=y,
        feature_names=feature_names,
        target_name="residuary_resistance",
        train_fraction=0.70,
        validation_fraction=0.15,
    )


def _load_auto_mpg() -> RawDataset:
    path = RAW_DIR / "auto_mpg" / "auto-mpg.data"
    columns = (
        "mpg",
        "cylinders",
        "displacement",
        "horsepower",
        "weight",
        "acceleration",
        "model_year",
        "origin",
        "car_name",
    )
    frame = pd.read_csv(
        path,
        sep=r"\s+",
        names=columns,
        na_values="?",
        quotechar='"',
    ).dropna(subset=["horsepower"])
    if frame.shape[0] != 392:
        raise ValueError("Auto MPG must contain 392 complete rows")
    numeric = frame[
        [
            "cylinders",
            "displacement",
            "horsepower",
            "weight",
            "acceleration",
            "model_year",
        ]
    ].to_numpy()
    origin = frame["origin"].to_numpy(dtype=np.int64)
    if set(np.unique(origin)) != {1, 2, 3}:
        raise ValueError("Auto MPG contains an unexpected origin category")
    origin_one_hot = np.column_stack([origin == category for category in (1, 2, 3)])
    X, y = _as_float_arrays(
        np.column_stack((numeric, origin_one_hot)),
        frame["mpg"].to_numpy(),
    )
    return RawDataset(
        name="auto_mpg",
        X=X,
        y=y,
        feature_names=(
            "cylinders",
            "displacement",
            "horsepower",
            "weight",
            "acceleration",
            "model_year",
            "origin_1",
            "origin_2",
            "origin_3",
        ),
        target_name="mpg",
        train_fraction=0.70,
        validation_fraction=0.15,
    )


def load_raw_dataset(name: str, *, energy_target: str = "Y1") -> RawDataset:
    loaders = {
        "friedman": _load_friedman,
        "concrete": _load_concrete,
        "california": _load_california,
        "yacht": _load_yacht_hydrodynamics,
        "auto_mpg": _load_auto_mpg,
    }
    normalized = name.lower().replace("-", "_")
    if normalized == "energy":
        return _load_energy(energy_target)
    try:
        return loaders[normalized]()
    except KeyError as error:
        raise ValueError(f"unknown benchmark dataset: {name}") from error


def _random_split_indices(
    size: int,
    *,
    train_fraction: float,
    validation_fraction: float,
    split_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    test_fraction = 1.0 - train_fraction - validation_fraction
    if min(train_fraction, validation_fraction, test_fraction) <= 0:
        raise ValueError("train, validation, and test fractions must all be positive")
    indices = np.arange(size)
    train, remainder = train_test_split(
        indices,
        train_size=train_fraction,
        random_state=split_seed,
        shuffle=True,
    )
    validation_share = validation_fraction / (validation_fraction + test_fraction)
    validation, test = train_test_split(
        remainder,
        train_size=validation_share,
        random_state=split_seed,
        shuffle=True,
    )
    return np.sort(train), np.sort(validation), np.sort(test)


def _prepare_from_arrays(
    *,
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: tuple[str, ...],
    target_name: str,
    indices: tuple[np.ndarray, np.ndarray, np.ndarray],
    feature_groups: Mapping[str, tuple[int, ...]],
    feature_group_roles: Mapping[str, str],
    encoder: Any | None = None,
    sample_metadata: Mapping[str, np.ndarray] | None = None,
) -> PreparedData:
    train, validation, test = indices
    if sample_metadata is not None:
        for key, values in sample_metadata.items():
            if len(values) != len(y):
                raise ValueError(f"sample metadata {key!r} is not aligned with y")
    x_scaler = StandardScaler().fit(X[train])
    y_scaler = StandardScaler().fit(y[train].reshape(-1, 1))
    X_scaled = x_scaler.transform(X).astype(np.float32)
    y_scaled = y_scaler.transform(y.reshape(-1, 1)).reshape(-1).astype(np.float32)
    return PreparedData(
        name=name,
        X_train=X_scaled[train],
        X_validation=X_scaled[validation],
        X_test=X_scaled[test],
        y_train=y_scaled[train],
        y_validation=y_scaled[validation],
        y_test=y_scaled[test],
        y_train_raw=y[train].astype(np.float64),
        y_validation_raw=y[validation].astype(np.float64),
        y_test_raw=y[test].astype(np.float64),
        train_indices=train,
        validation_indices=validation,
        test_indices=test,
        feature_names=feature_names,
        target_name=target_name,
        feature_groups=dict(feature_groups),
        feature_group_roles=dict(feature_group_roles),
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        encoder=encoder,
        sample_metadata=(
            {key: np.asarray(values) for key, values in sample_metadata.items()}
            if sample_metadata is not None
            else None
        ),
    )


def subset_prepared_training_data(
    data: PreparedData,
    fraction: float,
    *,
    subset_seed: int = 0,
    group_metadata_key: str | None = None,
) -> PreparedData:
    """Select nested training subsets while keeping validation and test rows fixed.

    Refit feature and target scalers on the selected rows. With
    group_metadata_key, retain every training group and take deterministic
    per-group prefixes to keep fractions nested.
    """

    if not 0.0 < fraction <= 1.0:
        raise ValueError("training fraction must be in (0, 1]")
    if subset_seed < 0:
        raise ValueError("subset seed must be non-negative")
    if fraction == 1.0:
        return data

    training_size = len(data.train_indices)
    rng = np.random.default_rng(subset_seed)
    if group_metadata_key is None:
        selected_positions = np.sort(
            rng.permutation(training_size)[: max(1, int(np.ceil(fraction * training_size)))]
        )
    else:
        if data.sample_metadata is None or group_metadata_key not in data.sample_metadata:
            raise ValueError(f"missing sample metadata group {group_metadata_key!r}")
        training_groups = np.asarray(data.sample_metadata[group_metadata_key])[
            data.train_indices
        ]
        selected_parts: list[np.ndarray] = []
        for group in np.unique(training_groups):
            positions = np.flatnonzero(training_groups == group)
            take = max(1, int(np.ceil(fraction * len(positions))))
            selected_parts.append(rng.permutation(positions)[:take])
        selected_positions = np.sort(np.concatenate(selected_parts))

    raw_train_X = data.x_scaler.inverse_transform(data.X_train)
    raw_validation_X = data.x_scaler.inverse_transform(data.X_validation)
    raw_test_X = data.x_scaler.inverse_transform(data.X_test)
    selected_raw_X = raw_train_X[selected_positions]
    selected_raw_y = data.y_train_raw[selected_positions]
    x_scaler = StandardScaler().fit(selected_raw_X)
    y_scaler = StandardScaler().fit(selected_raw_y.reshape(-1, 1))

    def scale_X(values: np.ndarray) -> np.ndarray:
        return x_scaler.transform(values).astype(np.float32)

    def scale_y(values: np.ndarray) -> np.ndarray:
        return y_scaler.transform(values.reshape(-1, 1)).reshape(-1).astype(np.float32)

    return PreparedData(
        name=data.name,
        X_train=scale_X(selected_raw_X),
        X_validation=scale_X(raw_validation_X),
        X_test=scale_X(raw_test_X),
        y_train=scale_y(selected_raw_y),
        y_validation=scale_y(data.y_validation_raw),
        y_test=scale_y(data.y_test_raw),
        y_train_raw=selected_raw_y.copy(),
        y_validation_raw=data.y_validation_raw.copy(),
        y_test_raw=data.y_test_raw.copy(),
        train_indices=data.train_indices[selected_positions].copy(),
        validation_indices=data.validation_indices.copy(),
        test_indices=data.test_indices.copy(),
        feature_names=data.feature_names,
        target_name=data.target_name,
        feature_groups=dict(data.feature_groups),
        feature_group_roles=dict(data.feature_group_roles),
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        encoder=data.encoder,
        sample_metadata=data.sample_metadata,
    )


def prepare_benchmark(
    name: str,
    *,
    split_seed: int = 0,
    energy_target: str = "Y1",
) -> PreparedData:
    raw = load_raw_dataset(name, energy_target=energy_target)
    X, y = _as_float_arrays(raw.X, raw.y)
    indices = _random_split_indices(
        len(y),
        train_fraction=raw.train_fraction,
        validation_fraction=raw.validation_fraction,
        split_seed=split_seed,
    )
    groups = {feature: (index,) for index, feature in enumerate(raw.feature_names)}
    roles = {feature: "ordinary" for feature in raw.feature_names}
    return _prepare_from_arrays(
        name=raw.name,
        X=X,
        y=y,
        feature_names=raw.feature_names,
        target_name=raw.target_name,
        indices=indices,
        feature_groups=groups,
        feature_group_roles=roles,
    )


def prepare_bike_sharing() -> PreparedData:
    path = RAW_DIR / "bike_sharing" / "hour.csv"
    frame = pd.read_csv(path)
    required = {
        "instant",
        "dteday",
        "season",
        "yr",
        "mnth",
        "hr",
        "holiday",
        "weekday",
        "workingday",
        "weathersit",
        "temp",
        "atemp",
        "hum",
        "windspeed",
        "casual",
        "registered",
        "cnt",
    }
    if set(frame.columns) != required:
        raise ValueError("Bike Sharing columns do not match the documented schema")
    if frame.isna().any().any():
        raise ValueError("Bike Sharing contains missing values")
    if not (frame["casual"] + frame["registered"] == frame["cnt"]).all():
        raise ValueError("Bike Sharing target identity check failed")
    timestamps = pd.to_datetime(frame["dteday"]) + pd.to_timedelta(frame["hr"], unit="h")
    if not timestamps.is_monotonic_increasing:
        raise ValueError("Bike Sharing rows are not chronological")

    size = len(frame)
    train_end = int(size * 0.60)
    validation_end = int(size * 0.80)
    indices = (
        np.arange(0, train_end),
        np.arange(train_end, validation_end),
        np.arange(validation_end, size),
    )

    blocks: list[np.ndarray] = []
    feature_names: list[str] = []
    feature_groups: dict[str, tuple[int, ...]] = {}
    roles: dict[str, str] = {}
    offset = 0

    def add_block(group: str, names: list[str], values: np.ndarray, role: str) -> None:
        nonlocal offset
        values = np.asarray(values, dtype=np.float64)
        blocks.append(values)
        feature_names.extend(names)
        feature_groups[group] = tuple(range(offset, offset + values.shape[1]))
        roles[group] = role
        offset += values.shape[1]

    ordinary = ["temp", "atemp", "hum", "windspeed"]
    for name in ordinary:
        add_block(name, [name], frame[[name]].to_numpy(), "ordinary")

    for name, period in (("hr", 24.0), ("mnth", 12.0)):
        angle = 2.0 * np.pi * frame[name].to_numpy(dtype=np.float64) / period
        add_block(
            name,
            [f"{name}_sin", f"{name}_cos"],
            np.column_stack((np.sin(angle), np.cos(angle))),
            "context",
        )

    categorical = ["season", "weathersit", "weekday"]
    encoder = OneHotEncoder(
        categories=[np.arange(1, 5), np.arange(1, 5), np.arange(0, 7)],
        handle_unknown="ignore",
        sparse_output=False,
        dtype=np.float64,
    )
    encoder.fit(frame.iloc[indices[0]][categorical])
    encoded = encoder.transform(frame[categorical])
    encoded_names = list(encoder.get_feature_names_out(categorical))
    start = 0
    for name, categories in zip(categorical, encoder.categories_, strict=True):
        width = len(categories)
        add_block(
            name,
            encoded_names[start : start + width],
            encoded[:, start : start + width],
            "context",
        )
        start += width

    for name in ("workingday", "holiday", "yr"):
        add_block(name, [name], frame[[name]].to_numpy(), "context")

    X = np.column_stack(blocks)
    y = frame["cnt"].to_numpy(dtype=np.float64)
    return _prepare_from_arrays(
        name="bike_sharing",
        X=X,
        y=y,
        feature_names=tuple(feature_names),
        target_name="cnt",
        indices=indices,
        feature_groups=feature_groups,
        feature_group_roles=roles,
        encoder=encoder,
    )


def prepare_parkinsons_telemonitoring() -> PreparedData:
    """Prepare future-within-patient total-UPDRS regression from UCI 189."""

    path = RAW_DIR / "parkinsons_telemonitoring" / "parkinsons_updrs.data"
    frame = pd.read_csv(path)
    if frame.shape != (5_875, 22) or frame.isna().any().any():
        raise ValueError("Parkinsons Telemonitoring does not match UCI 189")
    frame = frame.sort_values(["subject#", "test_time"], kind="stable").reset_index(
        drop=True
    )
    subjects = frame["subject#"].to_numpy(dtype=np.int64)
    if set(np.unique(subjects)) != set(range(1, 43)):
        raise ValueError("Parkinsons Telemonitoring must contain subjects 1 through 42")

    train_parts: list[np.ndarray] = []
    validation_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for subject in range(1, 43):
        subject_rows = np.flatnonzero(subjects == subject)
        train_end = int(len(subject_rows) * 0.60)
        validation_end = int(len(subject_rows) * 0.80)
        train_parts.append(subject_rows[:train_end])
        validation_parts.append(subject_rows[train_end:validation_end])
        test_parts.append(subject_rows[validation_end:])
    indices = tuple(
        np.sort(np.concatenate(parts))
        for parts in (train_parts, validation_parts, test_parts)
    )

    ordinary = [
        name
        for name in frame.columns
        if name
        not in {
            "subject#",
            "age",
            "sex",
            "test_time",
            "motor_UPDRS",
            "total_UPDRS",
        }
    ]
    encoder = OneHotEncoder(
        categories=[np.arange(1, 43)],
        handle_unknown="error",
        sparse_output=False,
        dtype=np.float64,
    )
    subject_column = subjects.reshape(-1, 1)
    encoder.fit(subject_column[indices[0]])
    subject_context = encoder.transform(subject_column)

    blocks: list[np.ndarray] = []
    feature_names: list[str] = []
    feature_groups: dict[str, tuple[int, ...]] = {}
    roles: dict[str, str] = {}
    offset = 0

    def add_block(group: str, names: list[str], values: np.ndarray, role: str) -> None:
        nonlocal offset
        values = np.asarray(values, dtype=np.float64)
        blocks.append(values)
        feature_names.extend(names)
        feature_groups[group] = tuple(range(offset, offset + values.shape[1]))
        roles[group] = role
        offset += values.shape[1]

    for name in ordinary:
        add_block(name, [name], frame[[name]].to_numpy(), "ordinary")
    add_block(
        "subject",
        [f"subject_{subject}" for subject in range(1, 43)],
        subject_context,
        "context",
    )
    for name in ("age", "sex", "test_time"):
        add_block(name, [name], frame[[name]].to_numpy(), "context")

    X, y = _as_float_arrays(np.column_stack(blocks), frame["total_UPDRS"].to_numpy())
    return _prepare_from_arrays(
        name="parkinsons_telemonitoring",
        X=X,
        y=y,
        feature_names=tuple(feature_names),
        target_name="total_UPDRS",
        indices=indices,
        feature_groups=feature_groups,
        feature_group_roles=roles,
        encoder=encoder,
        sample_metadata={
            "subject_id": subjects,
            "test_time": frame["test_time"].to_numpy(dtype=np.float64),
        },
    )
