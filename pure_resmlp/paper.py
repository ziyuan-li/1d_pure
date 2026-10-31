from __future__ import annotations

import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import argparse
import hashlib
import json
import time
from pathlib import Path
from itertools import combinations
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from pure_resmlp.paths import PROJECT_ROOT
from pure_resmlp.data import (
    prepare_benchmark, prepare_bike_sharing, prepare_parkinsons_telemonitoring,
    prepare_synthetic_interactions, subset_prepared_training_data,
)
from pure_resmlp.models.shortcut import ShortcutAblation
from pure_resmlp.training.run_experiment import (
    build_torch_model, prepare_model_inputs, transform_input_matrix, set_seed,
    predict_torch, regression_metrics, train_torch_model,
)


DEFAULT_RUN = 'primary/synthetic_higher_order/cv_pure/split_0/f1000/none/seed_0'
SYNTHETIC = ('friedman', 'synthetic_pairwise', 'synthetic_higher_order')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf8')


def runs():
    return read_json(PROJECT_ROOT / 'protocol/runs.json')


def get_run(ident):
    matches = [r for r in runs() if r['id'] == ident]
    if len(matches) != 1:
        raise ValueError(f'Unknown run: {ident}. Use the list command.')
    return matches[0]


def configure():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def make_model(run):
    name = {'rv_pu': 'rv_pure', 'cv_pu': 'cv_pure'}.get(run['model'], run['model'])
    model = build_torch_model(name, run['input_dim'], run['hidden_dim'])
    if run['model'] in ('rv_pu', 'cv_pu'):
        model = ShortcutAblation(model, complex_input=run['model'].startswith('cv_'))
    return model


def checkpoint_path(run, checkpoint=None):
    return PROJECT_ROOT / run['checkpoint'] if checkpoint is None else Path(checkpoint).resolve()


def load_model(run, device='cpu', checkpoint=None):
    path = checkpoint_path(run, checkpoint)
    if not path.exists():
        raise FileNotFoundError(f'{path}: extract the checkpoint archive at the repository root.')
    if checkpoint is None and sha256(path) != run['checkpoint_sha256']:
        raise ValueError(f'Checkpoint checksum mismatch: {run["id"]}')
    if checkpoint is not None and (path.parent / 'config.json').exists():
        saved = read_json(path.parent / 'config.json')
        if saved.get('source_run_id', saved.get('id')) != run['id']:
            raise ValueError('The checkpoint configuration does not match --run-id.')
        if saved.get('checkpoint') == path.name and saved.get('checkpoint_sha256') != sha256(path):
            raise ValueError('The checkpoint checksum does not match its saved configuration.')
    model = make_model(run)
    state = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def preprocessing(run):
    return read_json(PROJECT_ROOT / run['preprocessing'])


def predict_encoded(run, X, device='cpu', model=None):
    metadata = preprocessing(run)
    values = np.asarray(X, dtype=np.float64)
    expected = len(metadata['feature_names'])
    if values.ndim != 2 or values.shape[1] != expected or not np.isfinite(values).all():
        raise ValueError(f'Expected finite encoded inputs with shape (n, {expected}).')
    values = ((values - metadata['x_scaler_mean']) / metadata['x_scaler_scale']).astype(np.float32)
    description = SimpleNamespace(input_dim=expected, feature_names=tuple(metadata['feature_names']),
                                  feature_groups=metadata['feature_groups'], feature_group_roles=metadata['feature_group_roles'])
    values, _ = transform_input_matrix(description, run['model'], values, tuple(run['excluded_groups']))
    model = load_model(run, device) if model is None else model
    scaled = predict_torch(model, values, device=torch.device(device), batch_size=run['batch_size'],
                           complex_input=run['model'].startswith('cv_'))
    return scaled.astype(np.float64) * metadata['y_scaler_scale'][0] + metadata['y_scaler_mean'][0]


def prepare(run):
    dataset = run['dataset']
    if dataset == 'bike_sharing':
        data = prepare_bike_sharing(window=(.6, .8, 1.0))
    elif dataset == 'parkinsons_telemonitoring':
        data = prepare_parkinsons_telemonitoring(window=(.6, .8, 1.0), strict_time_groups=True)
    elif dataset.startswith('synthetic_'):
        data = prepare_synthetic_interactions(dataset.removeprefix('synthetic_'), n_samples=10000,
                  irrelevant_features=20, noise_fraction=.05, data_seed=0, split_seed=run['split_seed']).prepared
    else:
        data = prepare_benchmark(dataset, split_seed=run['split_seed'])
    data = subset_prepared_training_data(data, run['fraction'], subset_seed=0,
               group_metadata_key='subject_id' if dataset == 'parkinsons_telemonitoring' else None)
    metadata = preprocessing(run)
    with np.load(PROJECT_ROOT / metadata['split_indices'], allow_pickle=False) as indices:
        for key, value in data.split_indices.items():
            np.testing.assert_array_equal(value, indices[key])
    for prefix, scaler in [('x', data.x_scaler), ('y', data.y_scaler)]:
        np.testing.assert_allclose(scaler.mean_, metadata[f'{prefix}_scaler_mean'], rtol=0, atol=1e-10)
        np.testing.assert_allclose(scaler.scale_, metadata[f'{prefix}_scaler_scale'], rtol=0, atol=1e-10)
    return data


def generate_synthetic():
    from sklearn.datasets import make_friedman1
    folder = PROJECT_ROOT / 'data/synthetic'
    folder.mkdir(parents=True, exist_ok=True)
    for name in SYNTHETIC:
        if name == 'friedman':
            X, y = make_friedman1(n_samples=10000, n_features=10, noise=0.0, random_state=0)
            signal = y.copy()
        else:
            rng = np.random.default_rng(0)
            higher = name == 'synthetic_higher_order'
            X = rng.uniform(-1., 1., size=(10000, 27 if higher else 24))
            signal = X[:, 0]*X[:, 1]*X[:, 2] + X[:, 3]*X[:, 4]*X[:, 5]*X[:, 6] if higher else X[:, 0]*X[:, 1] + X[:, 2]*X[:, 3]
            y = signal + rng.normal(0.0, .05*float(np.std(signal)), size=10000)
        np.savez_compressed(folder / f'{name}.npz', X=X, y=y, signal=signal)
    print('Generated three synthetic datasets in', folder)


def demo(run, device, checkpoint=None):
    if run['dataset'] not in SYNTHETIC:
        raise ValueError('The demo uses only the three included synthetic datasets.')
    path = PROJECT_ROOT / 'data/synthetic' / f'{run["dataset"]}.npz'
    with np.load(path, allow_pickle=False) as data:
        metadata = preprocessing(run)
        with np.load(PROJECT_ROOT / metadata['split_indices'], allow_pickle=False) as split:
            indices = split['test']
        prediction = predict_encoded(run, data['X'][indices], device, load_model(run, device, checkpoint))
        rmse = float(np.sqrt(np.mean((prediction-data['y'][indices])**2)))
        expected = run['metrics']['test_rmse']
        if checkpoint is None:
            np.testing.assert_allclose(rmse, expected, rtol=5e-4, atol=5e-5)
        result = dict(run=run['id'], device=device, test_rmse=rmse, recorded_rmse=expected,
                      first_predictions=prediction[:5].tolist(), targets=data['y'][indices][:5].tolist())
        print(json.dumps(result, indent=2))
        return result


def verify(device):
    all_runs = runs()
    checked_preprocessors = set()
    rng = np.random.default_rng(123)
    for i, run in enumerate(all_runs, 1):
        metadata = preprocessing(run)
        if run['preprocessing'] not in checked_preprocessors:
            path = PROJECT_ROOT / metadata['split_indices']
            assert sha256(path) == metadata['split_sha256']
            with np.load(path, allow_pickle=False) as split:
                groups = [set(split[k].tolist()) for k in ('train','validation','test')]
                assert not(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
            checked_preprocessors.add(run['preprocessing'])
        model = load_model(run, device)
        assert all(torch.isfinite(p).all() for p in model.parameters())
        count = sum(p.numel()*(2 if p.is_complex() else 1) for p in model.parameters())
        assert count == run['trainable_real_parameters']
        values = rng.normal(0, .2, size=(4, len(metadata['feature_names'])))
        values = values*np.asarray(metadata['x_scaler_scale']) + np.asarray(metadata['x_scaler_mean'])
        result = predict_encoded(run, values, device, model)
        assert result.shape == (4,) and np.isfinite(result).all(), run['id']
        if i % 200 == 0:
            print(f'Checked {i}/{len(all_runs)} weights, dimensions, and forward passes', flush=True)
    print(json.dumps(dict(checkpoints=len(all_runs), preprocessing_records=len(checked_preprocessors), status='passed')))


def evaluate(run, device, output, *, noise=False, interactions=False, checkpoint=None):
    from pure_resmlp.evaluation.noise import continuous_ordinary_columns, add_gaussian_feature_noise
    from pure_resmlp.evaluation.finite_difference_interactions import finite_difference_interactions, interaction_recovery_metrics
    data = prepare(run)
    model = load_model(run, device, checkpoint)
    common = dict(dataset=run['dataset'], model=run['model'], training_seed=run['seed'])
    def predict(X, batch=512):
        values, _ = transform_input_matrix(data, run['model'], X, tuple(run['excluded_groups']))
        return predict_torch(model, values, device=torch.device(device), batch_size=batch,
                             complex_input=run['model'].startswith('cv_'))
    _, metrics = regression_metrics(data, predict(data.X_test, run['batch_size']))
    if checkpoint is None:
        for key in ('test_rmse', 'test_mae', 'test_r2'):
            np.testing.assert_allclose(metrics[key], run['metrics'][key], rtol=5e-4, atol=5e-5)
    folder = output / run['id']
    write_json(folder / 'metrics.json', metrics)
    write_json(folder / 'evaluation.json', dict(run_id=run['id'], **common,
        checkpoint_sha256=sha256(checkpoint_path(run, checkpoint)),
        checkpoint_source='released' if checkpoint is None else 'custom',
        preprocessing=run['preprocessing'], split_seed=run['split_seed'], fraction=run['fraction']))
    if noise:
        rows = []
        columns = continuous_ordinary_columns(data)
        for sigma in (0., .02, .05, .1):
            for seed in ((-1,) if sigma == 0 else range(5)):
                X = add_gaussian_feature_noise(data.X_test, columns, sigma, max(0, seed))
                _, scores = regression_metrics(data, predict(X, run['batch_size'] if sigma == 0 and run['kind']=='shortcut_ablation' else 512))
                rows.append(dict(**common, experiment='protocol_v3', noise_std=sigma,
                    corruption_seed=seed, perturbed_features=len(columns), **scores))
        pd.DataFrame(rows).to_csv(folder / 'noise.csv', index=False)
    if interactions:
        context = run['dataset'] in ('bike_sharing', 'parkinsons_telemonitoring')
        rows, recovery, context_rows, sampling = [], [], [], []
        for seed in ((0,) if context else range(5)):
            score = finite_difference_interactions(predict, data.X_test, data.X_train, data.feature_groups, n_samples=256, seed=seed)
            count = min(256, len(data.X_test))
            rng = np.random.default_rng(seed)
            test = rng.choice(len(data.X_test), size=count, replace=False)
            refs = rng.choice(len(data.X_train), size=count, replace=True)
            sampling.append(dict(sampling_seed=seed, test_indices=data.test_indices[test].tolist(),
                                 reference_indices=data.train_indices[refs].tolist()))
            for i, j in combinations(range(len(score.group_names)), 2):
                a, b = score.group_names[i], score.group_names[j]
                strength = float(score.matrix[i, j])
                rows.append(dict(**common, sampling_seed=seed, n_samples=count, i=i, j=j,
                                 feature_i=a, feature_j=b, strength=strength))
                if context and data.feature_group_roles[a] != data.feature_group_roles[b]:
                    ordinary, ctx = (a,b) if data.feature_group_roles[a] == 'ordinary' else (b,a)
                    context_rows.append(dict(**common, sampling_seed=seed, ordinary_feature=ordinary,
                                             context_feature=ctx, interaction_strength=strength))
            truth = preprocessing(run)['true_pairs']
            if truth:
                recovery.append(dict(**common, sampling_seed=seed, **interaction_recovery_metrics(score, truth)))
        frame = pd.DataFrame(rows)
        frame.to_csv(folder / 'pairs.csv', index=False)
        frame.to_csv(folder / 'interactions.csv', index=False)
        write_json(folder / 'sampling_indices.json', sampling)
        if context_rows:
            pd.DataFrame(context_rows).to_csv(folder / 'context.csv', index=False)
        if recovery:
            pd.DataFrame(recovery).to_csv(folder / 'recovery.csv', index=False)
    print(json.dumps(dict(run=run['id'], **metrics), indent=2))


def train(run, device, output, smoke_epochs=None):
    data = prepare(run)
    inputs = prepare_model_inputs(data, run['model'], tuple(run['excluded_groups']))
    assert inputs.input_dim == run['input_dim']
    set_seed(run['seed'])
    model = make_model(run)
    folder = output / ('smoke' if smoke_epochs else 'full') / run['id']
    folder.mkdir(parents=True, exist_ok=False)
    model, history, prediction, info = train_torch_model(model, inputs, data,
        device=torch.device(device), seed=run['seed'], epochs=smoke_epochs or run['epochs'],
        batch_size=run['batch_size'], learning_rate=run['learning_rate'], patience=run['patience'],
        complex_input=run['model'].startswith('cv_'), max_grad_norm=run['max_grad_norm'])
    _, metrics = regression_metrics(data, prediction)
    torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()}, folder / 'checkpoint.pt')
    pd.DataFrame(history).to_csv(folder / 'history.csv', index=False)
    write_json(folder / 'metrics.json', dict(**metrics, **info, epochs_run=len(history), smoke=bool(smoke_epochs)))
    config = {k:v for k,v in run.items() if k not in ('metrics', 'checkpoint_sha256', 'source_checkpoint_sha256', 'state_sha256')}
    config.update(checkpoint='checkpoint.pt', checkpoint_sha256=sha256(folder / 'checkpoint.pt'),
                  epochs=smoke_epochs or run['epochs'], source_run_id=run['id'], smoke=bool(smoke_epochs))
    write_json(folder / 'config.json', config)
    print(json.dumps(dict(run=run['id'], smoke=bool(smoke_epochs), **metrics, **info), indent=2))


def benchmark(run, device, output, checkpoint=None):
    data = prepare(run)
    model = load_model(run, device, checkpoint)
    target = torch.device(device)
    def synchronize():
        if target.type == 'cuda':
            torch.cuda.synchronize(target)
    def forward():
        values, _ = transform_input_matrix(data, run['model'], data.X_test, tuple(run['excluded_groups']))
        return predict_torch(model, values, device=target, batch_size=128, complex_input=run['model'].startswith('cv_'))
    start, warmups = time.perf_counter(), 0
    while warmups < 20 or time.perf_counter()-start < .5:
        forward(); synchronize(); warmups += 1
    timings = []
    for _ in range(100):
        synchronize(); start = time.perf_counter()
        forward(); synchronize()
        timings.append((time.perf_counter()-start)*1000)
    result = dict(run=run['id'], median_ms=float(np.median(timings)),
                  test_rows=len(data.X_test), batch_size=128, passes=100, warmups=warmups,
                  device=torch.cuda.get_device_name(target) if target.type=='cuda' else 'CPU', timings_ms=timings)
    write_json(output / run['id'] / 'latency.json', result)
    print(json.dumps({k:v for k,v in result.items() if k!='timings_ms'}, indent=2))


def figures(output):
    from pure_resmlp.evaluation import style_paper_figures as f
    f.apply_publication_style()
    f.plt.rcParams.update({'mathtext.fontset':'dejavusans', 'axes.titleweight':'bold'})
    f.block_diagram(output)
    f.architecture_diagram(output)
    frames = [pd.read_csv(PROJECT_ROOT / 'records' / n) for n in f.SOURCES]
    f.scientific_maps(frames[0], output)
    f.context_maps(frames[1], output)
    f.noise_curves(frames[2], output)
    f.learning_curves(frames[3], output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['list', 'demo', 'verify', 'train', 'evaluate', 'benchmark', 'figures', 'generate-data'])
    parser.add_argument('--run-id', default=DEFAULT_RUN)
    parser.add_argument('--dataset')
    parser.add_argument('--model')
    parser.add_argument('--kind')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output-dir', type=Path, default=PROJECT_ROOT / 'generated')
    parser.add_argument('--smoke-epochs', type=int)
    parser.add_argument('--noise', action='store_true')
    parser.add_argument('--interactions', action='store_true')
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    if args.checkpoint is not None and args.command not in ('demo', 'evaluate', 'benchmark'):
        parser.error('--checkpoint applies to demo, evaluate, and benchmark.')
    configure()
    if args.command == 'list':
        for r in runs():
            if all(value is None or r[key] == value for key,value in [('dataset',args.dataset),('model',args.model),('kind',args.kind)]):
                print(r['id'])
    elif args.command == 'generate-data':
        generate_synthetic()
    elif args.command == 'verify':
        verify(args.device)
    elif args.command == 'figures':
        figures(args.output_dir / 'figures')
    elif args.command == 'demo':
        demo(get_run(args.run_id), args.device, args.checkpoint)
    elif args.command == 'evaluate':
        evaluate(get_run(args.run_id), args.device, args.output_dir / 'evaluations', noise=args.noise, interactions=args.interactions, checkpoint=args.checkpoint)
    elif args.command == 'benchmark':
        benchmark(get_run(args.run_id), args.device, args.output_dir / 'latency', args.checkpoint)
    else:
        if args.smoke_epochs is not None and args.smoke_epochs < 1:
            parser.error('--smoke-epochs must be positive')
        train(get_run(args.run_id), args.device, args.output_dir / 'runs', args.smoke_epochs)


if __name__ == '__main__':
    main()
