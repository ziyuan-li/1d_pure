import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pure_resmlp.paths import PROJECT_ROOT
from pure_resmlp.evaluation.noise import (
    _bootstrap_interval, summarize_noise_rows, summarize_auc,
    summarize_overall_auc, summarize_auc_effects,
)
from pure_resmlp.evaluation.scientific_interaction_summary import write_tables


METRICS = ['test_mse_standardized', 'test_mae', 'test_rmse', 'test_r2']
SCIENCE = ['friedman', 'concrete', 'california', 'energy', 'yacht', 'auto_mpg']


def summary(frame, keys):
    values = frame.groupby(keys)[METRICS].agg(['mean', 'std', 'count'])
    values.columns = ['_'.join(c) for c in values.columns]
    return values.reset_index()


def effects(frame, keys):
    rows = []
    for key, group in frame.groupby(keys):
        key = key if isinstance(key, tuple) else (key,)
        for metric in METRICS:
            pivot = group.pivot(index='seed', columns='model', values=metric)
            if 'cv_pure' not in pivot:
                continue
            for baseline in pivot.columns.drop('cv_pure'):
                pair = pivot[[baseline, 'cv_pure']].dropna()
                if len(pair) < 3:
                    continue
                delta = pair[baseline] - pair.cv_pure
                if metric == 'test_r2':
                    delta = -delta
                low, high = _bootstrap_interval(delta.to_numpy())
                rows.append(dict(zip(keys, key)) | dict(baseline=baseline, candidate='cv_pure',
                    metric=metric, paired_seeds=len(pair), improvement_mean=delta.mean(),
                    ci_lower=low, ci_upper=high))
    return pd.DataFrame(rows)


def build(source, output):
    output.mkdir(parents=True, exist_ok=True)
    if source.resolve() == output.resolve():
        raise ValueError('Use a separate output directory to preserve released records.')
    def read(name):
        return pd.read_csv(source / (name + '.csv'))
    def save(name, frame):
        frame.to_csv(output / (name + '.csv'), index=False)
    selected = read('selected_runs')
    save('selected_runs', selected)
    primary = selected[selected.kind == 'primary']
    save('primary_summary', summary(primary, ['dataset', 'model']))
    save('primary_paired_effects', effects(primary, ['dataset']))
    split = selected[selected.dataset.isin(SCIENCE) & selected.kind.isin(['primary', 'split_robustness'])]
    per_split = summary(split, ['dataset', 'split_seed', 'model'])
    if len(per_split) != 150 or not per_split.test_rmse_count.eq(5).all():
        raise ValueError('Expected six datasets, five models, five splits, and five training seeds.')
    save('random_split_by_split', per_split)
    save('random_split_robustness', per_split.groupby(['dataset', 'model']).agg(
        split_mean_rmse=('test_rmse_mean', 'mean'), between_split_sd=('test_rmse_mean', 'std'),
        mean_within_split_training_sd=('test_rmse_std', 'mean'),
        completed_splits=('split_seed', 'nunique')).reset_index())
    save('random_split_paired_effects', effects(split, ['dataset', 'split_seed']))
    ranked = per_split.copy()
    ranked['rank'] = ranked.groupby(['dataset', 'split_seed']).test_rmse_mean.rank(method='min')
    ranked['best_in_split'] = ranked['rank'].eq(1)
    save('random_split_model_ranks', ranked)
    save('table3_summary', summary(read('table3_per_seed'), ['model', 'noise_std']))
    write_tables(read('pairs_raw'), output / 'scientific_interactions')
    recovery = read('recovery_raw')
    recovery['exact'] = recovery.precision_at_k.eq(1) & recovery.recall_at_k.eq(1)
    save('synthetic_recovery_summary', recovery.groupby(['dataset', 'model']).agg(
        precision=('precision_at_k', 'mean'), recall=('recall_at_k', 'mean'),
        mean_rank=('true_interaction_mean_rank', 'mean'), rank_sd=('true_interaction_mean_rank', 'std'),
        exact_count=('exact', 'sum'), repetitions=('exact', 'size')).reset_index())
    ablations = read('table7_per_seed').sort_values(['model', 'ablation', 'seed'])
    rows = []
    for (model, ablation), group in ablations.groupby(['model', 'ablation']):
        low, high = _bootstrap_interval(group.delta_mae.to_numpy())
        rows.append(dict(model=model, ablation=ablation, seeds=len(group),
            mae_mean=group.test_mae.mean(), mae_sd=group.test_mae.std(ddof=1),
            delta_mae_mean=group.delta_mae.mean(), ci_lower=low, ci_upper=high))
    save('table7_summary', pd.DataFrame(rows))
    full = primary[(primary.dataset == 'bike_sharing') & primary.model.isin(['rv_pure', 'cv_pure']) & primary.seed.isin([0, 1, 2])]
    save('table7_full_context', full.groupby('model').agg(
        mae_mean=('test_mae', 'mean'), mae_sd=('test_mae', 'std'), seeds=('seed', 'size')).reset_index())
    noise, noise_summary = summarize_noise_rows(read('noise_raw'))
    save('noise_per_training_seed', noise)
    save('noise_summary', noise_summary)
    save('noise_auc', summarize_auc(noise))
    save('noise_suite_auc', summarize_overall_auc(noise))
    save('noise_paired_effects', summarize_auc_effects(noise))
    curves = selected[(selected.kind == 'learning') | ((selected.kind == 'primary') & selected.dataset.isin(['friedman', 'concrete', 'parkinsons_telemonitoring']))]
    save('learning_curves', summary(curves, ['dataset', 'fraction', 'model']))
    rows = []
    for (dataset, model, seed), group in curves.groupby(['dataset', 'model', 'seed']):
        group = group.sort_values('fraction')
        np.testing.assert_array_equal(group.fraction, [.05, .10, .25, .50, 1.])
        rows.append(dict(dataset=dataset, model=model, seed=seed, aulc=np.trapz(group.test_rmse, group.fraction)))
    aulc = pd.DataFrame(rows)
    save('learning_aulc_per_seed', aulc)
    save('learning_aulc_summary', aulc.groupby(['dataset', 'model']).aulc.agg(['mean', 'std', 'count']).reset_index())
    rows = []
    for dataset, group in aulc.groupby('dataset'):
        pivot = group.pivot(index='seed', columns='model', values='aulc')
        for baseline in pivot.columns.drop('cv_pure'):
            delta = (pivot[baseline] - pivot.cv_pure).to_numpy()
            low, high = _bootstrap_interval(delta)
            rows.append(dict(dataset=dataset, baseline=baseline, candidate='cv_pure',
                paired_seeds=len(delta), aulc_improvement_mean=delta.mean(), ci_lower=low, ci_upper=high))
    save('learning_aulc_paired_effects', pd.DataFrame(rows))
    save('latency_summary', read('latency_per_seed').groupby(['dataset', 'model']).median_ms.agg(['mean', 'std', 'count']).reset_index())
    save('context_interactions', read('context_interactions'))
    print('Recomputed manuscript summaries from per-seed records in', output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--record-dir', type=Path, default=PROJECT_ROOT / 'records')
    parser.add_argument('--output-dir', type=Path, default=PROJECT_ROOT / 'generated/reports')
    args = parser.parse_args()
    build(args.record_dir, args.output_dir)
