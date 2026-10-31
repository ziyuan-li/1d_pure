import unittest

import numpy as np
import torch

from pure_resmlp.paper import DEFAULT_RUN, get_run, make_model, load_model, preprocessing, demo, runs
from pure_resmlp.training.run_experiment import set_seed


class ReleaseTests(unittest.TestCase):
    def test_paper_grid(self):
        records = runs()
        self.assertEqual(len(records), 1215)
        self.assertEqual(len({r['id'] for r in records}), 1215)
        self.assertFalse(any(r['kind'] in ('temporal_robustness', 'environment_control') for r in records))
        primary = [r for r in records if r['kind']=='primary']
        self.assertEqual(len(primary),247)
        self.assertEqual({r['seed'] for r in primary if r['dataset'].startswith('synthetic_')},{0,1,2})
        for dataset in ('friedman','concrete','california','energy','yacht','auto_mpg'):
            selected = [r for r in records if r['dataset']==dataset and r['kind'] in ('primary','split_robustness')]
            self.assertEqual(len(selected),125)
            self.assertEqual({r['split_seed'] for r in selected},set(range(5)))

    def test_signed_readout_and_independent_blocks(self):
        run = get_run(DEFAULT_RUN)
        set_seed(0)
        model = make_model(run)
        self.assertTrue(torch.count_nonzero(model.block2.fc1.weight)>0)
        self.assertFalse(torch.equal(model.block1.fc1.weight,model.block2.fc1.weight))
        self.assertEqual(torch.count_nonzero(model.out_fc.weight),0)
        values = torch.tensor([[-1+2j],[1-2j]],dtype=torch.complex64)
        prediction = model.readout(values)
        self.assertLess(prediction[0].item(),0)
        self.assertGreater(prediction[1].item(),0)

    def test_shortcut_pairing(self):
        for prefix in ('rv','cv'):
            full = get_run(f'primary/concrete/{prefix}_pure/split_0/f1000/none/seed_0')
            plain = get_run(f'shortcut_ablation/concrete/{prefix}_pu/split_0/f1000/none/seed_0')
            set_seed(0); a=make_model(full); state_a=torch.get_rng_state()
            set_seed(0); b=make_model(plain); state_b=torch.get_rng_state()
            self.assertTrue(torch.equal(state_a,state_b))
            for k,v in a.state_dict().items():
                self.assertTrue(torch.equal(v,b.network.state_dict()[k]))

    def test_synthetic_checkpoint_metrics(self):
        for dataset in ('friedman','synthetic_pairwise','synthetic_higher_order'):
            result = demo(get_run(f'primary/{dataset}/cv_pure/split_0/f1000/none/seed_0'),'cpu')
            self.assertTrue(np.isfinite(result['test_rmse']))

    def test_projection_in_checkpoint(self):
        run = get_run(DEFAULT_RUN)
        model = load_model(run)
        values = model(torch.ones(8,run['input_dim'],dtype=torch.complex64))
        self.assertEqual(values.shape,(8,1))
        self.assertTrue(torch.isfinite(values).all())
        self.assertEqual(len(preprocessing(run)['feature_names']),27)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
