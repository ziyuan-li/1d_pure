from __future__ import annotations
from typing import Mapping, Sequence
import numpy as np
import torch
import torch.nn as nn

class CrossLayerV2(nn.Module):

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, input_dim)

    def forward(self, x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return x0 * self.linear(x) + x

class DCNv2Regressor(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int=128, output_dim: int=1, num_cross_layers: int=3, num_deep_layers: int=3) -> None:
        super().__init__()
        if min(input_dim, hidden_dim, output_dim, num_cross_layers, num_deep_layers) <= 0:
            raise ValueError('all DCNv2 dimensions and layer counts must be positive')
        self.cross_layers = nn.ModuleList((CrossLayerV2(input_dim) for _ in range(num_cross_layers)))
        deep_layers: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(num_deep_layers):
            deep_layers.extend((nn.Linear(current_dim, hidden_dim), nn.ReLU()))
            current_dim = hidden_dim
        self.deep = nn.Sequential(*deep_layers)
        self.output = nn.Linear(input_dim + hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        crossed = x
        for layer in self.cross_layers:
            crossed = layer(x, crossed)
        deep = self.deep(x)
        return self.output(torch.cat((crossed, deep), dim=-1))

def explicit_ordinary_context_crosses(X: np.ndarray, feature_names: Sequence[str], feature_groups: Mapping[str, Sequence[int]], feature_group_roles: Mapping[str, str]) -> tuple[np.ndarray, tuple[str, ...]]:
    X = np.asarray(X)
    ordinary = [index for group, indices in feature_groups.items() if feature_group_roles[group] == 'ordinary' for index in indices]
    context = [index for group, indices in feature_groups.items() if feature_group_roles[group] == 'context' for index in indices]
    if not ordinary or not context:
        raise ValueError('explicit crosses require ordinary and context feature groups')
    products = (X[:, ordinary, None] * X[:, None, context]).reshape(X.shape[0], -1)
    cross_names = tuple((f'{feature_names[left]}*{feature_names[right]}' for left in ordinary for right in context))
    return (np.concatenate((X, products), axis=1), tuple(feature_names) + cross_names)
