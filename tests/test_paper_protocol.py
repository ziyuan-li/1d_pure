import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from pure_resmlp.data import prepare_parkinsons_telemonitoring, subset_prepared_training_data
from pure_resmlp.evaluation.finite_difference_interactions import finite_difference_interactions
from pure_resmlp.evaluation.noise import summarize_noise_rows, summarize_overall_auc
from pure_resmlp.models.pure import ProductUnit1D
from pure_resmlp.models.complex_pure import ComplexProductUnit1D
from pure_resmlp.paper import DEFAULT_RUN, evaluate, get_run, make_model, read_json, sha256


class PaperProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_real_and_complex_product_formulas(self):
        torch.manual_seed(7)
        x = torch.randn(11, 4)
        real = ProductUnit1D(4, 3)
        factors = torch.maximum(x, torch.zeros_like(x)) + 1
        expected = torch.prod(factors[:, None, :] ** real.linear.weight[None, :, :], dim=2)
        torch.testing.assert_close(real(x), expected)
        complex_unit = ComplexProductUnit1D(4, 3)
        z = torch.complex(x, torch.randn_like(x))
        safe = torch.complex(torch.relu(z.real) + 2 ** -.5, torch.relu(z.imag) + 2 ** -.5)
        weight = complex_unit.linear.weight
        log_amplitude, phase = torch.log(torch.abs(safe)), torch.angle(safe)
        q = log_amplitude @ weight.real.T - phase @ weight.imag.T
        omega = log_amplitude @ weight.imag.T + phase @ weight.real.T
        expected_complex = torch.exp(q) * torch.complex(torch.cos(omega), torch.sin(omega))
        torch.testing.assert_close(complex_unit(z), expected_complex)

    def test_mixed_difference_matches_analytic_interaction(self):
        rng = np.random.default_rng(8)
        test, reference = rng.normal(size=(80, 3)), rng.normal(size=(120, 3))
        result = finite_difference_interactions(lambda x: x[:, 0] * x[:, 1] + x[:, 2] ** 2,
            test, reference, {'a': (0,), 'b': (1,), 'c': (2,)}, n_samples=32, seed=3)
        sampling = np.random.default_rng(3)
        x = test[sampling.choice(80, 32, replace=False)]
        r = reference[sampling.choice(120, 32, replace=True)]
        expected = np.abs((x[:, 0] - r[:, 0]) * (x[:, 1] - r[:, 1])).mean()
        self.assertAlmostEqual(result.matrix[0, 1], expected)
        np.testing.assert_allclose(result.matrix[[0, 1], 2], 0, atol=1e-14)

    def test_default_chronological_groups_and_nested_subsets(self):
        subjects = np.concatenate([np.full(140 if s <= 37 else 139, s) for s in range(1, 43)])
        positions = np.concatenate([np.arange(140 if s <= 37 else 139) for s in range(1, 43)])
        frame = pd.DataFrame({'subject#': subjects, 'age': 40 + subjects, 'sex': subjects % 2,
            'test_time': positions // 3, 'motor_UPDRS': positions / 10,
            'total_UPDRS': 10 + subjects + positions / 5})
        rng = np.random.default_rng(9)
        for i in range(16):
            frame[f'voice{i}'] = rng.normal(size=len(frame))
        with patch('pure_resmlp.data.datasets.pd.read_csv', return_value=frame):
            data = prepare_parkinsons_telemonitoring()
            explicit = prepare_parkinsons_telemonitoring(strict_time_groups=True)
        groups = []
        for key, indices in data.split_indices.items():
            np.testing.assert_array_equal(indices, explicit.split_indices[key])
            groups.append(set(zip(data.sample_metadata['subject_id'][indices], data.sample_metadata['test_time'][indices])))
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        previous = set()
        for fraction in (.05, .10, .25, .50, 1.):
            subset = subset_prepared_training_data(data, fraction, subset_seed=0, group_metadata_key='subject_id')
            self.assertTrue(previous.issubset(set(subset.train_indices)))
            self.assertEqual(len(np.unique(data.sample_metadata['subject_id'][subset.train_indices])), 42)
            np.testing.assert_array_equal(subset.test_indices, data.test_indices)
            np.testing.assert_array_equal(subset.validation_indices, data.validation_indices)
            self.assertAlmostEqual(subset.y_scaler.mean_[0], subset.y_train_raw.mean())
            previous = set(subset.train_indices)

    def test_noise_auc_averages_corruptions_then_datasets(self):
        rows, expected = [], []
        datasets = ('friedman', 'concrete', 'california', 'energy', 'yacht', 'auto_mpg')
        sigma = (0., .02, .05, .10)
        for ds_index, dataset in enumerate(datasets):
            target_sd = 2. + ds_index
            for seed in range(5):
                curve = []
                for level in sigma:
                    center = .2 + ds_index * .01 + seed * .003 + 2 * level ** 2
                    curve.append(center)
                    offsets = [0.] if level == 0 else [-.01, .01, 0., -.02, .02]
                    for corruption, offset in enumerate(offsets):
                        rmse = (center + offset) * target_sd
                        rows.append(dict(experiment='test', dataset=dataset, model='cv_pure',
                            training_seed=seed, noise_std=level, corruption_seed=corruption,
                            test_mae=rmse / 2, test_rmse=rmse, test_r2=1 - (rmse / target_sd) ** 2))
                area = sum((curve[i] + curve[i+1]) * (sigma[i+1]-sigma[i]) / 2 for i in range(3)) / .1
                expected.append(dict(dataset=dataset, seed=seed, area=area))
        per_seed, _ = summarize_noise_rows(pd.DataFrame(rows))
        actual = summarize_overall_auc(per_seed).iloc[0]
        values = pd.DataFrame(expected).groupby('seed').area.mean().to_numpy()
        self.assertAlmostEqual(actual.performance_auc_mean, values.mean())
        self.assertAlmostEqual(actual.performance_auc_std, values.std(ddof=1))
        draws = np.random.default_rng(0).choice(values, size=(10000, 5)).mean(1)
        self.assertAlmostEqual(actual['performance_auc_ci_2.5'], np.quantile(draws, .025))
        self.assertAlmostEqual(actual['performance_auc_ci_97.5'], np.quantile(draws, .975))

    def test_evaluate_explicit_checkpoint_and_complete_output_schema(self):
        run = get_run(DEFAULT_RUN)
        torch.manual_seed(0)
        model = make_model(run)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / 'fresh.pt'
            torch.save(model.state_dict(), weights)
            with contextlib.redirect_stdout(io.StringIO()):
                evaluate(run, 'cpu', root / 'evaluation', checkpoint=weights, noise=True)
            folder = root / 'evaluation' / run['id']
            meta = read_json(folder / 'evaluation.json')
            self.assertEqual(meta['checkpoint_source'], 'custom')
            self.assertEqual(meta['checkpoint_sha256'], sha256(weights))
            scores = read_json(folder / 'metrics.json')
            self.assertGreater(scores['test_rmse'], run['metrics']['test_rmse'] * 2)
            noise = pd.read_csv(folder / 'noise.csv')
            self.assertEqual(len(noise), 16)
            self.assertTrue({'dataset', 'model', 'training_seed', 'corruption_seed', 'experiment'}.issubset(noise.columns))
            seed_rows, _ = summarize_noise_rows(pd.concat([noise, noise.assign(training_seed=1)], ignore_index=True))
            self.assertEqual(len(seed_rows), 8)


if __name__ == '__main__':
    unittest.main()
