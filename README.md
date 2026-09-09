# PU-ResMLP

Code for **Modeling Nonlinear Feature Interactions with Product-Unit Residual Networks**.

The repository compares real- and complex-valued MLP and Product-Unit Residual
Network (PURe) models on six scientific regression benchmarks: Friedman 1,
Concrete Compressive Strength, California Housing, Energy Efficiency, Yacht
Hydrodynamics, and Auto MPG.

The context-aware main study uses Bike Sharing and Parkinsons Telemonitoring.
Every dataset uses a leakage-safe split appropriate to its sampling structure
and training-only preprocessing statistics.

## Repository layout

```text
PU-ResMLP/
|-- pure_resmlp/
|   |-- models/       # RV-MLP, RV-PURe, CV-MLP, CV-PURe, and Res-MLP
|   |-- training/     # Main and sample-efficiency training entry points
|   |-- evaluation/   # Publication tables, figures, robustness, and SHAP
|   `-- legacy/       # Older single-purpose scripts retained for reference
|-- experiments/
|   `-- superconductivity/  # Historical experiment, not in the journal benchmark
|-- tests/
|-- data/
|   `-- raw/          # UCI datasets added for the JoCS extension
`-- artifacts/        # Generated runs and selected result summaries
```

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Run all commands below from the repository root.

Run the lightweight architecture checks with:

```bash
pytest
```

## Main experiments

The supported unified runner records config, environment, split indices, loss
history, predictions, metrics, checkpoint, and a completion manifest:

```bash
python -m pure_resmlp.training.run_experiment \
  --experiment jocs_benchmark_v1 --dataset friedman --model cv_pure \
  --seed 0 --split-seed 0
```

Evaluate the saved main models under the frozen Gaussian feature-noise protocol:

```bash
python -m pure_resmlp.evaluation.evaluate_noise_robustness
```

Run the 10%/25%/50% sample-efficiency additions on two GPUs, then combine them
with the existing 100% main runs:

```bash
python -m pure_resmlp.training.run_sample_efficiency_suite --gpus 0 1
python -m pure_resmlp.evaluation.summarize_sample_efficiency
```

Run the CV-MLP baselines used for the higher-order synthetic task, Bike, and
Parkinsons, including their interaction analyses:

```bash
python -m pure_resmlp.training.run_cv_baselines --gpus 0 1
python -m pure_resmlp.evaluation.summarize_jocs --runs-root artifacts/runs --output-dir artifacts/reports/jocs_v1
```

Generate the main paper summary and figures:

```bash
python -m pure_resmlp.evaluation.summarize_jocs --runs-root artifacts/runs --output-dir artifacts/reports/jocs_v1
python -m pure_resmlp.evaluation.plot_jocs --runs-root artifacts/runs --report-dir artifacts/reports/jocs_v1
python -m pure_resmlp.evaluation.audit_checkpoints --runs-root artifacts/runs --strict
```

Rebenchmark full-test inference latency with CUDA synchronization and repeated
measurements:

```bash
python -m pure_resmlp.evaluation.benchmark_inference --device cuda:0
```

## Current reproducibility status

- The journal main result set contains 226 training runs and 106 interaction
  analyses. The scientific benchmark contributes 120 checkpoints: four models,
  six datasets, five training seeds, and fixed split seed zero. CV-MLP is also
  included for higher-order interaction recovery, Bike, and Parkinsons.
- Gaussian robustness reuses 190 main checkpoints for 3,040 inference
  evaluations at four noise levels and five corruption seeds. Its primary
  comparison is target-SD-normalized noisy-performance AUC over sigma 0--0.1.
- Sample efficiency adds 210 low-data checkpoints on Friedman, Concrete, and
  Parkinsons and reuses 70 full-data runs, yielding 280 curve observations.
- The server artifact audit loads all 226 formal checkpoints and 210 new
  low-data checkpoints successfully. The 13 CV-MLP baselines on higher-order
  synthetic data, Bike, and Parkinsons are part of that formal audit; all 48
  automated tests pass.
- Complex-valued models use a learnable real projection initialized as the
  real component. It supports signed standardized targets and passes gradients
  through a zero-initialized output layer.

## Results and repository contents

The [result index](artifacts/reports/jocs_v1/README.md) lists the versioned
experiment summaries and evaluation protocols. Training and evaluation write
their outputs to `artifacts/`; only the selected CSV tables and protocol JSON
files are included in Git. Checkpoints, per-run artifacts, and generated figures
remain local. Dataset sources and loading notes are in
[data/raw/README.md](data/raw/README.md).

The repository contains experiment code, tests, dependencies, data instructions,
and result summaries. Manuscripts, extension plans, writing notes, cover letters,
highlights, compiled documents, and temporary files remain local and are excluded
by `.gitignore`.

## Legacy scripts

Files in `pure_resmlp/legacy/` are preserved to document earlier analysis steps.
The supported publication entry points are the modules in
`pure_resmlp/evaluation/`.
