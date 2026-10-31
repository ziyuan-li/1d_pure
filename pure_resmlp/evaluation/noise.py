from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd

def continuous_ordinary_columns(data: Any) -> np.ndarray:
    ordinary = {index for group, indices in data.feature_groups.items() if data.feature_group_roles[group] == 'ordinary' for index in indices}
    raw_train = data.x_scaler.inverse_transform(data.X_train)
    return np.asarray([index for index in sorted(ordinary) if np.unique(raw_train[:, index]).size > 2], dtype=np.int64)

def add_gaussian_feature_noise(X: np.ndarray, columns: np.ndarray, noise_std: float, seed: int) -> np.ndarray:
    if noise_std < 0:
        raise ValueError('noise standard deviation must be non-negative')
    result = np.asarray(X, dtype=np.float32).copy()
    if noise_std == 0 or len(columns) == 0:
        return result
    noise = np.random.default_rng(seed).normal(0.0, noise_std, size=(len(result), len(columns)))
    result[:, columns] += noise.astype(np.float32)
    return result


def _bootstrap_interval(values: np.ndarray, seed: int=0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    return (float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)))

def summarize_noise_rows(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_rows = raw.groupby(['experiment', 'dataset', 'model', 'training_seed', 'noise_std'], as_index=False)[['test_mae', 'test_rmse', 'test_r2']].mean().sort_values(['dataset', 'model', 'training_seed', 'noise_std'])
    clean = seed_rows.loc[seed_rows['noise_std'] == 0].rename(columns={'test_mae': 'clean_mae', 'test_rmse': 'clean_rmse', 'test_r2': 'clean_r2'})[['experiment', 'dataset', 'model', 'training_seed', 'clean_mae', 'clean_rmse', 'clean_r2']]
    seed_rows = seed_rows.merge(clean, on=['experiment', 'dataset', 'model', 'training_seed'], validate='many_to_one')
    seed_rows['mae_increase_pct'] = 100.0 * (seed_rows['test_mae'] - seed_rows['clean_mae']) / seed_rows['clean_mae']
    seed_rows['rmse_increase_pct'] = 100.0 * (seed_rows['test_rmse'] - seed_rows['clean_rmse']) / seed_rows['clean_rmse']
    seed_rows['r2_drop'] = seed_rows['clean_r2'] - seed_rows['test_r2']
    seed_rows['test_target_sd'] = seed_rows['clean_rmse'] / np.sqrt(np.maximum(1.0 - seed_rows['clean_r2'], np.finfo(np.float64).eps))
    seed_rows['test_nrmse'] = seed_rows['test_rmse'] / seed_rows['test_target_sd']
    seed_rows['degradation_nrmse'] = (seed_rows['test_rmse'] - seed_rows['clean_rmse']) / seed_rows['test_target_sd']
    summaries: list[dict[str, Any]] = []
    group_columns = ['experiment', 'dataset', 'model', 'noise_std']
    for keys, values in seed_rows.groupby(group_columns, sort=True):
        row = dict(zip(group_columns, keys, strict=True))
        row['training_seeds'] = int(values['training_seed'].nunique())
        for metric in ('test_mae', 'test_rmse', 'test_r2', 'mae_increase_pct', 'rmse_increase_pct', 'r2_drop', 'test_nrmse', 'degradation_nrmse'):
            numbers = values[metric].to_numpy(dtype=np.float64)
            low, high = _bootstrap_interval(numbers)
            row[f'{metric}_mean'] = float(numbers.mean())
            row[f'{metric}_std'] = float(numbers.std(ddof=1))
            row[f'{metric}_ci_2.5'] = low
            row[f'{metric}_ci_97.5'] = high
        summaries.append(row)
    return (seed_rows, pd.DataFrame.from_records(summaries))

def _auc_by_training_seed(seed_rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    group_columns = ['experiment', 'dataset', 'model', 'training_seed']
    for keys, curve in seed_rows.groupby(group_columns, sort=True):
        curve = curve.sort_values('noise_std')
        noise = curve['noise_std'].to_numpy(dtype=np.float64)
        noise_range = float(noise[-1] - noise[0])
        if noise_range <= 0:
            raise ValueError('AUC requires at least two distinct noise levels')
        row = dict(zip(group_columns, keys, strict=True))
        row['performance_auc'] = float(np.trapz(curve['test_nrmse'].to_numpy(), noise) / noise_range)
        row['degradation_auc'] = float(np.trapz(curve['degradation_nrmse'].to_numpy(), noise) / noise_range)
        records.append(row)
    return pd.DataFrame.from_records(records)

def summarize_auc(seed_rows: pd.DataFrame) -> pd.DataFrame:
    per_seed = _auc_by_training_seed(seed_rows)
    rows: list[dict[str, Any]] = []
    group_columns = ['experiment', 'dataset', 'model']
    for keys, values in per_seed.groupby(group_columns, sort=True):
        row = dict(zip(group_columns, keys, strict=True))
        row['training_seeds'] = int(values['training_seed'].nunique())
        for metric in ('performance_auc', 'degradation_auc'):
            numbers = values[metric].to_numpy(dtype=np.float64)
            low, high = _bootstrap_interval(numbers)
            row[f'{metric}_mean'] = float(numbers.mean())
            row[f'{metric}_std'] = float(numbers.std(ddof=1))
            row[f'{metric}_ci_2.5'] = low
            row[f'{metric}_ci_97.5'] = high
        rows.append(row)
    return pd.DataFrame.from_records(rows)

def summarize_overall_auc(seed_rows: pd.DataFrame) -> pd.DataFrame:
    per_seed = _auc_by_training_seed(seed_rows)
    suite_definitions = {'scientific': {'datasets': {'friedman', 'concrete', 'california', 'energy', 'yacht', 'auto_mpg'}, 'models': {'mlp', 'resmlp', 'rv_pure', 'cv_mlp', 'cv_pure'}}, 'context': {'datasets': {'bike_sharing', 'parkinsons_telemonitoring'}, 'models': {'mlp_context', 'mlp_explicit_cross', 'dcnv2_context', 'rv_pure', 'cv_mlp', 'cv_pure'}}}
    rows: list[dict[str, Any]] = []
    for suite, definition in suite_definitions.items():
        selected = per_seed.loc[per_seed['dataset'].isin(definition['datasets']) & per_seed['model'].isin(definition['models'])]
        within_seed = selected.groupby(['model', 'training_seed'], as_index=False)[['performance_auc', 'degradation_auc']].mean()
        for model, values in within_seed.groupby('model', sort=True):
            model_dataset_count = int(selected.loc[selected['model'] == model, 'dataset'].nunique())
            row: dict[str, Any] = {'suite': suite, 'model': model, 'datasets': model_dataset_count, 'training_seeds': int(values['training_seed'].nunique()), 'dataset_weighting': 'equal'}
            for metric in ('performance_auc', 'degradation_auc'):
                numbers = values[metric].to_numpy(dtype=np.float64)
                low, high = _bootstrap_interval(numbers)
                row[f'{metric}_mean'] = float(numbers.mean())
                row[f'{metric}_std'] = float(numbers.std(ddof=1))
                row[f'{metric}_ci_2.5'] = low
                row[f'{metric}_ci_97.5'] = high
            rows.append(row)
    return pd.DataFrame.from_records(rows)

def summarize_auc_effects(seed_rows: pd.DataFrame) -> pd.DataFrame:
    per_seed = _auc_by_training_seed(seed_rows)
    definitions = {'scientific': {'datasets': {'friedman', 'concrete', 'california', 'energy', 'yacht', 'auto_mpg'}, 'baselines': ('mlp', 'resmlp', 'rv_pure', 'cv_mlp')}, 'context': {'datasets': {'bike_sharing', 'parkinsons_telemonitoring'}, 'baselines': ('mlp_context', 'mlp_explicit_cross', 'dcnv2_context', 'rv_pure', 'cv_mlp')}}
    rows: list[dict[str, Any]] = []
    for suite, definition in definitions.items():
        selected = per_seed.loc[per_seed['dataset'].isin(definition['datasets'])]
        for metric in ('performance_auc', 'degradation_auc'):
            pivot = selected.pivot(index=['dataset', 'training_seed'], columns='model', values=metric)
            for baseline in definition['baselines']:
                paired = (pivot[baseline] - pivot['cv_pure']).rename('improvement')
                within_seed = paired.groupby('training_seed').mean().to_numpy()
                low, high = _bootstrap_interval(within_seed)
                rows.append({'suite': suite, 'metric': metric, 'baseline': baseline, 'candidate': 'cv_pure', 'datasets': len(definition['datasets']), 'training_seed_pairs': len(within_seed), 'mean_improvement': float(within_seed.mean()), 'ci_2.5': low, 'ci_97.5': high})
    return pd.DataFrame.from_records(rows)
