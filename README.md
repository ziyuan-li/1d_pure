# PURe: prediction and interaction recovery

Code and checkpoints for *Learning Nonlinear Interactions: Prediction and
Interpretation with Product-Unit Residual Networks* (journal revision,
31 October 2026), by Ziyuan Li, Uwe Jaekel, and Babette Dellen.

This release contains the experiments used by the manuscript's **10 tables
and 6 figures**, including 1,215 trained checkpoints. CV-PURe and CV-MLP use
the signed readout `alpha * Re(o) + beta * Im(o) + b`. Both PURe blocks use
independent default initialization; the final output affine layer starts at zero.

See [paper–code correspondence](PAPER_CODE_MAP.md) for the methods, seeds,
initialization, and sources for every table and figure.

## Quick start

Use Python 3.12. Install a PyTorch build appropriate for your CPU or GPU,
then run these commands from the repository root:

```bash
python -m pip install -r requirements.txt
python -m pure_resmlp.paper demo
python -m pure_resmlp.paper verify
python -m unittest discover -s tests -v
```

The demo loads the higher-order CV-PURe checkpoint (training seed 0), predicts
the included synthetic test data, and checks its RMSE against the saved result.
`verify` loads every checkpoint, checks checksums and dimensions, and runs
generated-input forward passes. Add `--device cuda:0` to use a GPU.

If checkpoints are distributed separately, extract `pure_paper_checkpoints.zip`
at the repository root to create `checkpoints/`. Each file contains a PyTorch
state dictionary and is loaded with `weights_only=True`.

## Reproduce experiments and figures

```bash
python -m pure_resmlp.paper list --dataset friedman --model cv_pure
python -m pure_resmlp.paper train --device cuda:0
python -m pure_resmlp.paper evaluate --noise --interactions --device cuda:0
python -m pure_resmlp.summarize
python -m pure_resmlp.tables
python -m pure_resmlp.paper figures
```

`train` and `evaluate` default to the higher-order CV-PURe seed-0 run; select
another listed configuration with `--run-id ID`. Training starts from scratch
using the recorded settings. `--smoke-epochs 2` runs a short training check.
Outputs go to `generated/`; released weights and records are preserved.
Evaluation uses released weights by default. Pass `--checkpoint PATH` to
`evaluate`, `demo`, or `benchmark` to use a newly trained weight file with the
same `--run-id`; the evaluation records its actual checkpoint hash.
Tables and figures rebuild from saved results without datasets or a GPU.
`summarize` recomputes statistics from per-seed records; use
`python -m pure_resmlp.tables --report-dir generated/reports` to use those summaries.
Use `python -m pure_resmlp.paper benchmark --run-id ID --device cuda:0` for
the synchronized full-test latency protocol.

| Contents | Location |
|---|---|
| Architectures, preprocessing, training, interaction scoring | `pure_resmlp/` |
| Run configurations, seeds, scalers, exact split indices | `protocol/` |
| Model weights | `checkpoints/` |
| Per-seed metrics and table/figure source records | `records/` |
| Friedman, pairwise, and higher-order synthetic data | `data/synthetic/` |
| Six manuscript figures | `figures/` |

Scientific benchmarks use split seeds **0–4**, each with training seeds **0–4**.
Other random-split analyses use split **0**. Synthetic recovery and Bike context
ablations use training seeds **0–2**; other experiments use **0–4**. Bike and
Parkinsons use chronological partitions. Full settings and checkpoint provenance
are in `protocol/`, with table/figure sources in `protocol/manuscript_index.json`.
The recorded environment is PyTorch 2.14.0/CUDA 13.0 on an
RTX 5080. Hardware or package changes can affect training and timing results.

Only generated synthetic observations are included. Obtain third-party data
from their original sources using [the data guide](data/README.md). No third-party
sample rows, targets, or individual predictions are distributed.

Code and checkpoints are provided under the [MIT License](LICENSE).
