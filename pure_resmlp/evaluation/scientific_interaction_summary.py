from __future__ import annotations
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
DATASETS = ('friedman', 'concrete', 'california')
MODELS = ('mlp', 'resmlp', 'rv_pure', 'cv_mlp', 'cv_pure')
SEEDS = tuple(range(5))
TOP_K = 10
ZERO_TOLERANCE = 1e-12
PAIR_COUNTS = {'friedman': 45, 'concrete': 28, 'california': 28}
KEYS = ['dataset', 'model', 'training_seed', 'sampling_seed']

def concentration(scores: np.ndarray, k: int=TOP_K) -> dict[str, float]:
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or len(scores) < k or k <= 1:
        raise ValueError('Need at least k strengths and k > 1')
    if not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError('Interaction strengths must be finite and nonnegative')
    total = float(scores.sum())
    top = np.sort(scores)[-k:]
    if total <= ZERO_TOLERANCE:
        return dict(total_strength=total, top10_mass=np.nan, entropy_top10=np.nan, hhi_top10=np.nan)
    p = top / top.sum()
    positive = p[p > 0]
    return dict(total_strength=total, top10_mass=float(top.sum() / total), entropy_top10=float(-(positive * np.log(positive)).sum() / np.log(k)), hhi_top10=float(np.square(p).sum()))

def top_pairs(frame: pd.DataFrame) -> frozenset[tuple[int, int]]:
    if frame.strength.sum() <= ZERO_TOLERANCE:
        return frozenset()
    ordered = frame.sort_values(['strength', 'i', 'j'], ascending=[False, True, True])
    return frozenset(ordered.head(TOP_K)[['i', 'j']].itertuples(index=False, name=None))

def jaccard(left: frozenset, right: frozenset) -> float:
    return len(left & right) / len(left | right) if left and right else np.nan

def validate_raw(raw: pd.DataFrame) -> None:
    expected = {(d, m, t, s) for d in DATASETS for m in MODELS for t in SEEDS for s in SEEDS}
    if set(raw[KEYS].itertuples(index=False, name=None)) != expected:
        raise ValueError('Incomplete dataset/model/training/sampling grid')
    if raw.duplicated(KEYS + ['i', 'j']).any() or not (raw.i < raw.j).all():
        raise ValueError('Duplicate, diagonal, or reversed pairs')
    if not np.isfinite(raw.strength).all() or (raw.strength < 0).any():
        raise ValueError('Invalid interaction strengths')
    for key, group in raw.groupby(KEYS):
        n = 10 if key[0] == 'friedman' else 8
        if set(group[['i', 'j']].itertuples(index=False, name=None)) != set(combinations(range(n), 2)):
            raise ValueError(f'Incomplete feature pairs: {key}')
        expected_samples = 206 if key[0] == 'concrete' else 256
        if not group.n_samples.eq(expected_samples).all():
            raise ValueError(f'Unexpected evaluation row count: {key}')

def summarize(raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    validate_raw(raw)
    metric_records, sampling_pairs = ([], [])
    metric_names = ['total_strength', 'top10_mass', 'entropy_top10', 'hhi_top10']
    for key, group in raw.groupby(KEYS, sort=True):
        record = dict(zip(KEYS, key))
        record.update(concentration(group.strength.to_numpy()))
        strengths = group.sort_values('strength', ascending=False).strength.to_numpy()
        record['top10_boundary_tie'] = bool(abs(strengths[9] - strengths[10]) <= ZERO_TOLERANCE)
        if key[0] == 'friedman':
            true_score = float(group.loc[(group.i == 0) & (group.j == 1), 'strength'].item())
            others = group.loc[~((group.i == 0) & (group.j == 1)), 'strength']
            record.update(true_pair_strength=true_score, true_pair_rank=1 + int((others > true_score + ZERO_TOLERANCE).sum()), true_pair_top1=int(true_score > float(others.max()) + ZERO_TOLERANCE), true_pair_mass=true_score / record['total_strength'] if record['total_strength'] > ZERO_TOLERANCE else np.nan)
        metric_records.append(record)
    per_resample = pd.DataFrame(metric_records)
    metric_names += ['true_pair_strength', 'true_pair_rank', 'true_pair_top1', 'true_pair_mass']
    per_training = per_resample.groupby(KEYS[:3], as_index=False)[metric_names].mean()
    for key, group in raw.groupby(KEYS[:3], sort=True):
        sets = {s: top_pairs(g) for s, g in group.groupby('sampling_seed')}
        for a, b in combinations(SEEDS, 2):
            sampling_pairs.append(dict(zip(KEYS[:3], key)) | dict(sampling_seed_a=a, sampling_seed_b=b, jaccard_top10=jaccard(sets[a], sets[b])))
    sampling = pd.DataFrame(sampling_pairs)
    sample_mean = sampling.groupby(KEYS[:3], as_index=False).jaccard_top10.mean().rename(columns={'jaccard_top10': 'sampling_jaccard_top10'})
    per_training = per_training.merge(sample_mean, on=KEYS[:3], validate='one_to_one')
    metric_names += ['sampling_jaccard_top10']
    mean_maps = raw.groupby(['dataset', 'model', 'training_seed', 'i', 'j', 'feature_i', 'feature_j'], as_index=False).strength.mean()
    training_pairs = []
    for key, group in mean_maps.groupby(['dataset', 'model'], sort=True):
        sets = {s: top_pairs(g) for s, g in group.groupby('training_seed')}
        for a, b in combinations(SEEDS, 2):
            training_pairs.append(dict(dataset=key[0], model=key[1], training_seed_a=a, training_seed_b=b, jaccard_top10=jaccard(sets[a], sets[b])))
    stability = pd.DataFrame(training_pairs)
    summary_records = []
    for key, group in per_training.groupby(['dataset', 'model'], sort=True):
        record = dict(dataset=key[0], model=key[1], training_seeds=len(group), sampling_seeds=5)
        for metric in metric_names:
            record[metric + '_mean'] = group[metric].mean()
            record[metric + '_std'] = group[metric].std(ddof=1)
        js = stability.loc[(stability.dataset == key[0]) & (stability.model == key[1]), 'jaccard_top10']
        record.update(training_jaccard_top10_mean=js.mean(), training_jaccard_top10_min=js.min(), training_jaccard_top10_max=js.max(), training_pair_comparisons=len(js))
        summary_records.append(record)
    summary = pd.DataFrame(summary_records)
    pooled_maps = mean_maps.groupby(['dataset', 'model', 'i', 'j', 'feature_i', 'feature_j'], as_index=False).agg(strength_mean=('strength', 'mean'), strength_std=('strength', 'std'))
    pooled_maps['rank'] = pooled_maps.groupby(['dataset', 'model']).strength_mean.rank(ascending=False, method='min').astype(int)
    effect_records = []
    for dataset in DATASETS:
        metrics = ['entropy_top10', 'hhi_top10', 'top10_mass', 'sampling_jaccard_top10']
        if dataset == 'friedman':
            metrics += ['true_pair_mass']
        for baseline in ('mlp', 'cv_mlp'):
            for metric in metrics:
                pivot = per_training.loc[per_training.dataset == dataset].pivot(index='training_seed', columns='model', values=metric)
                delta = (pivot.cv_pure - pivot[baseline]).to_numpy()
                draws = np.random.default_rng(0).choice(delta, size=(10000, 5), replace=True).mean(1)
                effect_records.append(dict(dataset=dataset, baseline=baseline, candidate='cv_pure', metric=metric, candidate_minus_baseline_mean=delta.mean(), ci_lower=np.quantile(draws, 0.025), ci_upper=np.quantile(draws, 0.975), training_seed_pairs=5))
    return dict(per_resample=per_resample, per_training=per_training, summary=summary, sampling_stability_pairs=sampling, training_stability_pairs=stability, mean_maps=pooled_maps, paired_effects=pd.DataFrame(effect_records))

def write_tables(raw: pd.DataFrame, output: Path) -> dict[str, pd.DataFrame]:
    tables = summarize(raw)
    output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output / f'{name}.csv', index=False)
    return tables
