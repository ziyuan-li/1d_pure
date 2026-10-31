from pathlib import Path
import argparse
import json
import pandas as pd
from pure_resmlp.paths import PROJECT_ROOT

parser = argparse.ArgumentParser()
parser.add_argument('--report-dir', type=Path, default=PROJECT_ROOT/'records')
parser.add_argument('--output-dir', type=Path, default=PROJECT_ROOT/'generated/tables')
args = parser.parse_args()
REPORT = args.report_dir
models = ['mlp', 'resmlp', 'rv_pure', 'cv_mlp', 'cv_pure']
context_models = ['mlp_ordinary', 'mlp_context', 'mlp_explicit_cross', 'dcnv2_context', 'rv_pure', 'cv_mlp', 'cv_pure']
science = ['friedman', 'concrete', 'california', 'energy', 'yacht', 'auto_mpg']
labels = dict(zip(models, ['MLP', 'ResMLP', 'RV-PURe', 'CV-MLP', 'CV-PURe'])) | {'mlp_ordinary': 'MLP-ordinary', 'mlp_context': 'MLP-context', 'mlp_explicit_cross': 'MLP-explicit-cross', 'dcnv2_context': 'DCNv2-context', 'rv_pu': 'RV-PU', 'cv_pu': 'CV-PU'}
names = dict(zip(science, ['Friedman', 'Concrete', 'California', 'Energy', 'Yacht', 'Auto MPG'])) | {'synthetic_pairwise': 'Pairwise', 'synthetic_higher_order': 'Higher-order', 'bike_sharing': 'Bike Sharing', 'parkinsons_telemonitoring': 'Parkinsons'}
primary = pd.read_csv(REPORT / 'primary_summary.csv')
tables = {}

def number(mean, sd=None, *, digits=4, bold=False):
    value = f'{mean:.{digits}f}'
    if sd is not None:
        value += f'\\pm{sd:.{digits}f}'
    return '$' + ('\\mathbf{' + value + '}' if bold else value) + '$'

def table(label, caption, columns, header, rows, *, wide=True, size='\\small', spacing=4):
    env = 'table*' if wide else 'table'
    body = '\n'.join(('\\midrule' if row is None else ' & '.join(row) + ' \\\\' for row in rows))
    value = f'\\begin{{{env}}}[pos=tp]\\revisioncolor' + '\n' + '\\caption{' + caption + '}\n' + '\\label{' + label + '}\\centering' + size + '\n' + f'\\setlength{{\\tabcolsep}}{{{spacing}pt}}' + '\n' + '\\begin{tabular}{@{}' + columns + '@{}}' + '\n' + '\\toprule' + '\n' + ' & '.join(header) + ' \\\\ \\midrule' + '\n' + body + '\n' + '\\bottomrule' + '\n' + '\\end{tabular}' + '\n' + f'\\end{{{env}}}' + '\n'
    tables[label] = value
    return value
dataset_rows = [['Friedman', '10,000', '10', '7,000 / 1,499 / 1,501', 'Random'], ['Concrete', '1,030', '8', '618 / 206 / 206', 'Random'], ['California Housing', '20,640', '8', '16,512 / 2,064 / 2,064', 'Random'], ['Energy Efficiency', '768', '8', '537 / 115 / 116', 'Random'], ['Yacht Hydrodynamics', '308', '6', '215 / 46 / 47', 'Random'], ['Auto MPG', '392', '9', '274 / 58 / 60', 'Random'], ['Synthetic pairwise', '10,000', '24', '7,000 / 1,499 / 1,501', 'Random'], ['Synthetic higher-order', '10,000', '27', '7,000 / 1,499 / 1,501', 'Random'], ['Bike Sharing', '17,379', '26', '10,427 / 3,476 / 3,476', 'Chronological'], ['Parkinsons', '5,875', '61', '3,573 / 1,221 / 1,081', 'Within-subject chronological']]
t1 = table('tab:datasets', 'Datasets and actual partition sizes. $p$ is the encoded input dimension, including full context where applicable. Random partitions use split seed 0 for the fixed-split analyses; the six scientific benchmarks also use split seeds 1--4.', 'lrrll', ['Dataset', 'Observations', '$p$', 'Train / validation / test', 'Partition'], dataset_rows)
data = pd.read_csv(REPORT / 'random_split_robustness.csv')
rows = []
for ds in science:
    g = data[data.dataset == ds].set_index('model')
    rows.append([names[ds]] + [number(g.loc[m, 'split_mean_rmse'], g.loc[m, 'between_split_sd'], bold=g.loc[m, 'split_mean_rmse'] == g.split_mean_rmse.min()) for m in models])
t2 = table('tab:scientific', 'Cross-split test RMSE in original target units. Each of five split seeds uses five training seeds. Values are the mean and sample SD of the five split-level means; bold marks the lowest mean per dataset. California values are in units of \\$100,000.', 'lrrrrr', ['Dataset'] + [labels[m] for m in models], rows)
data = pd.read_csv(REPORT / 'table3_summary.csv')
rows = []
for m in ['rv_pu', 'rv_pure', 'cv_pu', 'cv_pure']:
    row = [labels[m], 'Yes' if m.endswith('pure') else 'No']
    for sigma in [0.0, 0.05]:
        g = data[data.noise_std == sigma].set_index('model')
        row.append(number(g.loc[m, 'test_mse_standardized_mean'], g.loc[m, 'test_mse_standardized_std'], bold=g.loc[m, 'test_mse_standardized_mean'] == g.test_mse_standardized_mean.min()))
    rows.append(row)
t3 = table('tab:skip-ablation', 'Residual-connection ablation on Concrete: standardized-target test MSE (mean $\\pm$ sample SD, five paired training seeds). PU removes only the identity shortcuts. Bold marks the lowest mean in each condition.', 'lcrr', ['Model', 'Shortcuts', '$\\sigma=0$', '$\\sigma=0.05$'], rows, wide=False, size='\\footnotesize', spacing=3)
data = pd.read_csv(REPORT / 'scientific_interactions/summary.csv')
rows = []
for ds in science[:3]:
    if rows:
        rows.append(None)
    g = data[data.dataset == ds].set_index('model')
    for k, m in enumerate(models):
        row = [names[ds] if k == 0 else '', labels[m]]
        for col, sd, digits, scale, direction in [('entropy_top10_mean', 'entropy_top10_std', 3, 1, 'min'), ('top10_mass_mean', 'top10_mass_std', 2, 100, 'max'), ('true_pair_mass_mean', 'true_pair_mass_std', 2, 100, 'max'), ('training_jaccard_top10_mean', None, 3, 1, 'max'), ('sampling_jaccard_top10_mean', 'sampling_jaccard_top10_std', 3, 1, 'max')]:
            x = g.loc[m, col]
            row.append(number(x * scale, g.loc[m, sd] * scale if sd else None, digits=digits, bold=x == getattr(g[col], direction)()) if pd.notna(x) else '---')
        rows.append(row)
t4 = table('tab:scientific-interactions', 'Interaction concentration, Friedman true-pair share, and Top-10 Jaccard stability. Except for mean between-run training stability, entries are mean $\\pm$ sample SD over five training seeds after averaging five sampling repetitions. Bold marks the most concentrated or stable values within each dataset; true-pair share uses the known Friedman coupling.', 'llrrrrr', ['Dataset', 'Model', '$H_{10}$', '$C_{10}$ (\\%)', 'True pair (\\%)', '\\shortstack{Jaccard\\\\Training}', '\\shortstack{Jaccard\\\\Sampling}'], rows, size='\\footnotesize', spacing=3)
data = pd.read_csv(REPORT / 'synthetic_recovery_summary.csv')
rows = []
for ds in ['synthetic_pairwise', 'synthetic_higher_order']:
    if rows:
        rows.append(None)
    g = data[data.dataset == ds].set_index('model')
    p = primary[primary.dataset == ds].set_index('model')
    for k, m in enumerate([m for m in models if m in g.index]):
        z = g.loc[m]
        exact = f'{int(z.exact_count)}/{int(z.repetitions)}'
        rows.append([names[ds] if k == 0 else '', labels[m], number(p.loc[m, 'test_rmse_mean'], p.loc[m, 'test_rmse_std'], bold=p.loc[m, 'test_rmse_mean'] == p.test_rmse_mean.min()), number(z.precision, digits=3, bold=z.precision == g.precision.max()), number(z.mean_rank, z.rank_sd, digits=1, bold=z.mean_rank == g.mean_rank.min()), '\\textbf{' + exact + '}' if z.exact_count == g.exact_count.max() else exact])
t5 = table('tab:recovery', 'Synthetic prediction and interaction recovery. RMSE is mean $\\pm$ sample SD over three training seeds; ranks summarize $3\\times5$ training/sampling combinations. $K=2$ for the pairwise task and $K=9$ for the higher-order task. Exact gives complete recoveries out of 15 combinations. Bold marks the best values.', 'llrrrr', ['Task', 'Model', 'Test RMSE', 'Precision@$K$ = Recall@$K$', 'Mean rank', 'Exact'], rows)
g = primary[primary.dataset == 'bike_sharing'].set_index('model')
rows = []
for m in context_models:
    rows.append([labels[m]] + [number(g.loc[m, c + '_mean'], g.loc[m, c + '_std'], bold=g.loc[m, c + '_mean'] == (g[c + '_mean'].max() if c == 'test_r2' else g[c + '_mean'].min())) for c in ['test_mae', 'test_rmse', 'test_r2']])
t6 = table('tab:context', 'Bike Sharing prediction (mean $\\pm$ sample SD, five training seeds). MAE and RMSE are in rental counts. Bold marks the lowest mean errors and highest mean $R^2$.', 'lrrr', ['Model', 'MAE', 'RMSE', '$R^2$'], rows)
data = pd.read_csv(REPORT / 'table7_summary.csv')
full = pd.read_csv(REPORT / 'table7_full_context.csv').set_index('model')
rows = []
for m in ['rv_pure', 'cv_pure']:
    z = full.loc[m]
    rows.append(['None', labels[m], number(z.mae_mean, z.mae_sd, bold=z.mae_mean == full.mae_mean.min()), '---'])
for ab, label in [('bike_no_time', 'Hour, month'), ('bike_no_calendar', 'Calendar'), ('bike_no_weather', 'Season, weather category')]:
    rows.append(None)
    g = data[data.ablation == ab].set_index('model')
    for m in ['rv_pure', 'cv_pure']:
        z = g.loc[m]
        rows.append([label, labels[m], number(z.mae_mean, z.mae_sd, bold=z.mae_mean == g.mae_mean.min()), f'${z.delta_mae_mean:.4f}\\ [{z.ci_lower:.4f},{z.ci_upper:.4f}]$'])
t7 = table('tab:ablations', 'RV-PURe and CV-PURe context ablations on Bike Sharing (mean $\\pm$ sample SD, training seeds 0--2). $\\Delta$MAE is ablated minus full-context MAE, paired by seed. Calendar comprises weekday, holiday, and working day. Bold marks the lower mean MAE for each input condition.', 'llrr', ['Removed group', 'Model', 'MAE', '$\\Delta$MAE [95\\% CI]'], rows)
data = pd.read_csv(REPORT / 'noise_suite_auc.csv')
rows = []
for suite, ms in [('scientific', models), ('context', context_models[1:])]:
    if rows:
        rows.append(None)
    rows.append(['\\multicolumn{3}{@{}l}{' + ('Scientific suite (six datasets)' if suite == 'scientific' else 'Context suite (two datasets)') + '}'])
    g = data[data.suite == suite].set_index('model')
    for m in ms:
        z = g.loc[m]
        rows.append([labels[m], number(z.performance_auc_mean, z.performance_auc_std, bold=z.performance_auc_mean == g.performance_auc_mean.min()), f"$[{z['performance_auc_ci_2.5']:.4f},{z['performance_auc_ci_97.5']:.4f}]$"])
t8 = table('tab:noise', 'Noise-performance AUC over $\\sigma=0$--$0.10$, normalized by test-target SD and averaged equally across datasets in each suite. Mean $\\pm$ sample SD and bootstrap intervals summarize five training seeds. Lower is better; bold marks the lowest mean per suite.', 'lrr', ['Model', 'AUC', '95\\% CI'], rows, wide=False, size='\\footnotesize', spacing=3)
data = pd.read_csv(REPORT / 'learning_aulc_summary.csv')
rows = []
for ds in ['friedman', 'concrete', 'parkinsons_telemonitoring']:
    if rows:
        rows.append(None)
    g = data[data.dataset == ds].set_index('model')
    for k, m in enumerate(context_models if ds.startswith('parkinsons') else models):
        z = g.loc[m]
        rows.append([names[ds] if k == 0 else '', labels[m], number(z['mean'], z['std'], bold=z['mean'] == g['mean'].min())])
t9 = table('tab:sample', 'RMSE area under the learning curve over training fractions 0.05--1.00 (mean $\\pm$ sample SD, five training seeds). Lower is better; bold marks the lowest mean within each dataset.', 'llr', ['Dataset', 'Model', 'AULC'], rows, wide=False, size='\\footnotesize', spacing=3)
selected = pd.read_csv(REPORT / 'selected_runs.csv')
selected = selected[selected.kind == 'primary']
latency = pd.read_csv(REPORT / 'latency_summary.csv')
rows = []
for k, m in enumerate(models):
    counts = selected[selected.dataset.isin(science) & (selected.model == m)].trainable_real_dof
    row = ['Scientific' if k == 0 else '', labels[m], f'{int(counts.min()):,}--{int(counts.max()):,}']
    for ds in ['friedman', 'yacht', 'auto_mpg']:
        g = latency[latency.dataset == ds].set_index('model')
        z = g.loc[m]
        row.append(number(z['mean'], z['std'], bold=z['mean'] == g['mean'].min()))
    rows.append(row)
for ds in ['synthetic_higher_order', 'bike_sharing', 'parkinsons_telemonitoring']:
    rows.append(None)
    for k, m in enumerate(['cv_mlp', 'cv_pure']):
        count = int(selected[(selected.dataset == ds) & (selected.model == m)].trainable_real_dof.iloc[0])
        rows.append([names[ds] if k == 0 else '', labels[m], f'{count:,}', '---', '---', '---'])
t10 = table('tab:efficiency', 'Model capacity and full-test inference latency on an RTX 5080. Complex parameters count as two real scalars; complex-model totals include three real projection parameters. Scientific parameter ranges reflect input dimensions. Latency is in milliseconds (mean $\\pm$ sample SD of five checkpoint-level medians); bold marks the lowest mean in each timed column.', 'llrrrr', ['Task', 'Model', 'Real parameters', 'Friedman (1,501)', 'Yacht (47)', 'Auto MPG (60)'], rows, spacing=3)
args.output_dir.mkdir(parents=True, exist_ok=True)
for i, (label, value) in enumerate(tables.items(), 1):
    (args.output_dir/f'table{i:02d}.tex').write_text(value.replace(r'\revisioncolor', ''), encoding='utf8')
(args.output_dir/'tables.json').write_text(json.dumps(tables,indent=2),encoding='utf8')
print('Exported all 10 manuscript tables to',args.output_dir)
