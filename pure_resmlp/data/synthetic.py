"""Controlled interaction-recovery datasets for the JoCS extension."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from pure_resmlp.data.datasets import (
    PreparedData,
    _prepare_from_arrays,
    _random_split_indices,
)


@dataclass(frozen=True)
class SyntheticInteractionData:
    prepared: PreparedData
    true_pairs: tuple[tuple[str, str], ...]
    interaction_groups: tuple[tuple[str, ...], ...]


def prepare_synthetic_interactions(
    kind: str,
    *,
    n_samples: int = 10_000,
    irrelevant_features: int = 20,
    noise_fraction: float = 0.05,
    data_seed: int = 0,
    split_seed: int = 0,
) -> SyntheticInteractionData:
    """Generate pairwise or higher-order data with known interaction groups."""

    if n_samples < 100:
        raise ValueError("n_samples must be at least 100")
    if irrelevant_features < 0 or noise_fraction < 0:
        raise ValueError("irrelevant_features and noise_fraction must be non-negative")
    normalized = kind.lower().replace("-", "_")
    signal_features = 4 if normalized == "pairwise" else 7
    if normalized not in {"pairwise", "higher_order"}:
        raise ValueError("kind must be pairwise or higher_order")

    rng = np.random.default_rng(data_seed)
    X = rng.uniform(-1.0, 1.0, size=(n_samples, signal_features + irrelevant_features))
    names = tuple(f"x{i + 1}" for i in range(X.shape[1]))

    if normalized == "pairwise":
        signal = X[:, 0] * X[:, 1] + X[:, 2] * X[:, 3]
        group_indices = ((0, 1), (2, 3))
    else:
        signal = X[:, 0] * X[:, 1] * X[:, 2] + X[:, 3] * X[:, 4] * X[:, 5] * X[:, 6]
        group_indices = ((0, 1, 2), (3, 4, 5, 6))

    noise_scale = noise_fraction * float(np.std(signal))
    y = signal + rng.normal(0.0, noise_scale, size=n_samples)
    interaction_groups = tuple(
        tuple(names[index] for index in group) for group in group_indices
    )
    true_pairs = tuple(
        pair
        for group in interaction_groups
        for pair in combinations(group, 2)
    )
    indices = _random_split_indices(
        n_samples,
        train_fraction=0.70,
        validation_fraction=0.15,
        split_seed=split_seed,
    )
    feature_groups = {name: (index,) for index, name in enumerate(names)}
    roles = {name: "ordinary" for name in names}
    prepared = _prepare_from_arrays(
        name=f"synthetic_{normalized}",
        X=X,
        y=y,
        feature_names=names,
        target_name="y",
        indices=indices,
        feature_groups=feature_groups,
        feature_group_roles=roles,
    )
    return SyntheticInteractionData(
        prepared=prepared,
        true_pairs=true_pairs,
        interaction_groups=interaction_groups,
    )
