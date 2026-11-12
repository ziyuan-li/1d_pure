# PURe

PyTorch implementation of real- and complex-valued product-unit residual networks for nonlinear regression and feature-interaction analysis.

Code for **Learning Nonlinear Interactions: Prediction and Interpretation with Product-Unit Residual Networks**, by Ziyuan Li, Uwe Jaekel, and Babette Dellen.

The repository includes model architectures, baselines, pretrained checkpoints, experiment configurations, and analysis scripts.

## Quick start

Use Python 3.12 and run the following commands from the repository root:

```bash
python -m pip install -r requirements.txt
python -m pure_resmlp.paper demo
```

The demo evaluates a bundled CV-PURe checkpoint on synthetic data. Add `--device cuda:0` to run it on a GPU. Recorded package versions are listed in [requirements-reproduction.txt](requirements-reproduction.txt).

## Experiments

List the available runs, then use a run ID to train or evaluate a model:

```bash
python -m pure_resmlp.paper list
python -m pure_resmlp.paper train --run-id RUN_ID --device cuda:0
python -m pure_resmlp.paper evaluate --run-id RUN_ID --noise --interactions --device cuda:0
```

Replace `RUN_ID` with an ID from `list`. Evaluation uses the bundled checkpoint by default; add `--checkpoint PATH` to evaluate a newly trained model. Use `--device cpu` for CPU execution. Outputs are saved under `generated/`.

Rebuild summaries, tables, and figures from the saved records:

```bash
python -m pure_resmlp.summarize
python -m pure_resmlp.tables --report-dir generated/reports
python -m pure_resmlp.paper figures
```

## Files and data

| Contents | Location |
| --- | --- |
| Architectures, training, and interaction analysis | [pure_resmlp/](pure_resmlp/) |
| Run settings, seeds, scalers, and split indices | [protocol/](protocol/) |
| Trained model weights | [checkpoints/](checkpoints/) |
| Saved experiment results | [records/](records/) |
| Result visualizations | [figures/](figures/) |

Synthetic datasets are included in [data/synthetic/](data/synthetic/). Obtain third-party datasets from their original sources using [the data guide](data/README.md).

## Verification

Check the bundled checkpoints and run the tests:

```bash
python -m pure_resmlp.paper verify
python -m unittest discover -s tests -v
```

## License

Code and checkpoints are provided under the [MIT License](LICENSE).
