from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Mapping, Sequence
import numpy as np
PredictFunction = Callable[[np.ndarray], np.ndarray]

@dataclass(frozen=True)
class InteractionResult:
    group_names: tuple[str, ...]
    matrix: np.ndarray

    def ranked_pairs(self) -> list[tuple[str, str, float]]:
        pairs = [(self.group_names[i], self.group_names[j], float(self.matrix[i, j])) for i, j in combinations(range(len(self.group_names)), 2)]
        return sorted(pairs, key=lambda item: item[2], reverse=True)

def finite_difference_interactions(predict_fn: PredictFunction, X_evaluation: np.ndarray, X_reference: np.ndarray, feature_groups: Mapping[str, Sequence[int]], *, n_samples: int=256, seed: int=0) -> InteractionResult:
    X_evaluation = np.asarray(X_evaluation)
    X_reference = np.asarray(X_reference)
    if X_evaluation.ndim != 2 or X_reference.ndim != 2:
        raise ValueError('evaluation and reference inputs must be matrices')
    if X_evaluation.shape[1] != X_reference.shape[1]:
        raise ValueError('evaluation and reference inputs must have equal width')
    if n_samples <= 0:
        raise ValueError('n_samples must be positive')
    group_names = tuple(feature_groups)
    if len(group_names) < 2:
        raise ValueError('at least two feature groups are required')
    for name, indices in feature_groups.items():
        if not indices:
            raise ValueError(f'feature group is empty: {name}')
        if min(indices) < 0 or max(indices) >= X_evaluation.shape[1]:
            raise ValueError(f'feature group index is out of range: {name}')
    rng = np.random.default_rng(seed)
    count = min(n_samples, len(X_evaluation))
    evaluation_rows = rng.choice(len(X_evaluation), size=count, replace=False)
    reference_rows = rng.choice(len(X_reference), size=count, replace=True)
    evaluation = X_evaluation[evaluation_rows]
    references = X_reference[reference_rows]
    baseline = np.asarray(predict_fn(evaluation)).reshape(-1)
    if baseline.size != count:
        raise ValueError('predict_fn must return one prediction per row')
    matrix = np.zeros((len(group_names), len(group_names)), dtype=np.float64)
    for left, right in combinations(range(len(group_names)), 2):
        left_columns = list(feature_groups[group_names[left]])
        right_columns = list(feature_groups[group_names[right]])
        left_replaced = evaluation.copy()
        right_replaced = evaluation.copy()
        both_replaced = evaluation.copy()
        left_replaced[:, left_columns] = references[:, left_columns]
        right_replaced[:, right_columns] = references[:, right_columns]
        both_replaced[:, left_columns] = references[:, left_columns]
        both_replaced[:, right_columns] = references[:, right_columns]
        delta = baseline - np.asarray(predict_fn(left_replaced)).reshape(-1) - np.asarray(predict_fn(right_replaced)).reshape(-1) + np.asarray(predict_fn(both_replaced)).reshape(-1)
        score = float(np.mean(np.abs(delta)))
        matrix[left, right] = score
        matrix[right, left] = score
    return InteractionResult(group_names=group_names, matrix=matrix)

def interaction_recovery_metrics(result: InteractionResult, true_pairs: Sequence[tuple[str, str]], *, k: int | None=None) -> dict[str, float]:
    truth = {frozenset(pair) for pair in true_pairs}
    if not truth:
        raise ValueError('true_pairs cannot be empty')
    ranked = result.ranked_pairs()
    known_groups = set(result.group_names)
    if any((not pair.issubset(known_groups) for pair in truth)):
        raise ValueError('true_pairs reference unknown feature groups')
    if k is None:
        k = len(truth)
    if k <= 0:
        raise ValueError('k must be positive')
    selected = {frozenset(pair[:2]) for pair in ranked[:k]}
    true_positive = len(selected & truth)
    ranks = [rank for rank, (left, right, _score) in enumerate(ranked, start=1) if frozenset((left, right)) in truth]
    return {'precision_at_k': true_positive / k, 'recall_at_k': true_positive / len(truth), 'true_interaction_mean_rank': float(np.mean(ranks))}
