"""Tests for the scientific CV-MLP comparison protocol."""

import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pure_resmlp.training.run_scientific_cv_mlp import (
    BASELINE_EXPERIMENT, DATASETS, EXPERIMENT, PARAMETERS,
    check_config, check_splits, command, jobs, run_path,
)


class ScientificCVMLPTests(unittest.TestCase):
    def test_grid_contains_exactly_six_datasets_and_five_seeds(self):
        grid = jobs()
        self.assertEqual(len(grid), 30)
        self.assertEqual(len(set(grid)), 30)
        for dataset in DATASETS:
            self.assertEqual({seed for name, seed in grid if name == dataset}, set(range(5)))

    def test_paths_do_not_modify_the_formal_benchmark(self):
        root = Path("artifacts/runs")
        self.assertEqual(run_path(root, "yacht", 2),
                         root / EXPERIMENT / "yacht/cv_mlp/seed_2")
        self.assertEqual(run_path(root, "yacht", 2, "cv_pure"),
                         root / BASELINE_EXPERIMENT / "yacht/cv_pure/seed_2")

    def test_command_is_frozen_and_never_overwrites(self):
        args = command(Path("artifacts/runs"), "energy", 4)
        flags = dict(zip(args[3::2], args[4::2]))
        self.assertNotIn("--overwrite", args)
        self.assertEqual(flags["--model"], "cv_mlp")
        self.assertEqual(flags["--split-seed"], "0")
        self.assertEqual(flags["--epochs"], "200")
        self.assertEqual(flags["--batch-size"], "128")
        self.assertEqual(flags["--patience"], "20")
        self.assertEqual(flags["--hidden-dim"], "64")
        self.assertEqual(flags["--learning-rate"], "0.001")
        self.assertEqual(flags["--energy-target"], "Y1")

    def test_protocol_rejects_context_settings_and_other_splits(self):
        config = dict(experiment=EXPERIMENT, dataset="energy", model="cv_mlp",
                      seed=4, split_seed=0, parameters=dict(PARAMETERS))
        check_config(config, "energy", 4, "cv_mlp")
        bad = copy.deepcopy(config)
        bad["parameters"]["epochs"] = 100
        with self.assertRaises(ValueError):
            check_config(bad, "energy", 4, "cv_mlp")
        bad = copy.deepcopy(config)
        bad["split_seed"] = 1
        with self.assertRaises(ValueError):
            check_config(bad, "energy", 4, "cv_mlp")

    def test_split_comparison_checks_indices_not_just_lengths(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "split_indices.npz"
            expected = dict(train=np.array([0, 2]), validation=np.array([1]),
                            test=np.array([3]))
            np.savez(path, **expected)
            check_splits(path, expected)
            changed = dict(expected, train=np.array([2, 0]))
            with self.assertRaises(ValueError):
                check_splits(path, changed)


if __name__ == "__main__":
    unittest.main()
