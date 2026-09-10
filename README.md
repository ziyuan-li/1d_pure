# 1d_pure

Code and experiment results for the journal-extension manuscript:

**Beyond Prediction: Learning and Interpreting Nonlinear Interactions with
Product-Unit Residual Networks**

Ziyuan Li, Uwe Jaekel, and Babette Dellen

Product-Unit Residual Networks (PURe) combine learned multiplicative features
with residual connections. This study evaluates nonlinear prediction together
with input-level interaction interpretation: synthetic functions test recovery
of known interactions, while real-data maps describe the feature relationships
learned by the models. The PyTorch implementation includes real-valued PURe
(RV-PURe), complex-valued PURe (CV-PURe), MLP, ResMLP, CV-MLP, explicit-cross
MLP, and DCNv2.

The manuscript extends our ICCS 2026 paper,
[Modeling Nonlinear Feature Interactions with Product-Unit Residual Networks](https://doi.org/10.1007/978-3-032-29921-5_6).
The [conference preprint](https://arxiv.org/abs/2606.06861) is also available.

[Experiment results](artifacts/reports/jocs_v1/README.md) ·
[Scientific CV-MLP comparison](artifacts/reports/jocs_scientific_cv_mlp_v1/summary.md) ·
[Dataset setup](data/raw/README.md) · [Citation](#citation)

## Experiments

| Task | Datasets | Training seeds | Evaluation |
|---|---|---|---|
| Interaction recovery | Pairwise and higher-order synthetic functions | 0–2 | RMSE, Precision@K, Recall@K, mean rank of true interacting pairs |
| Scientific regression | Friedman, Concrete, California Housing, Energy Efficiency, Yacht Hydrodynamics, Auto MPG | 0–4 | Test RMSE |
| Context-aware prediction | Bike Sharing, Parkinsons Telemonitoring | 0–4 | MAE, RMSE, R-squared, interaction maps |
| Input-noise robustness | Six scientific and two context datasets | 0–4, saved models | Normalized RMSE curves and noisy-performance AUC |
| Sample efficiency | Friedman, Concrete, Parkinsons | 0–4 | Learning curves and RMSE AULC |
| Inference efficiency | Friedman, Yacht, Auto MPG | 0–4, saved models | Repeated full-test inference latency |

Scientific and synthetic tasks use split seed 0. Bike uses a chronological
split; Parkinsons uses chronological splits within each subject. Preprocessing
statistics are fitted on training data. Synthetic attribution uses sampling
seeds 0–4 for each checkpoint; these are separate from training seeds. Bike
context ablations use training seeds 0–2.

## Installation

Use Python 3.12, the version used for the repository checks. Run commands from
the repository root.

```bash
git clone https://github.com/ziyuan-li/1d_pure.git
cd 1d_pure
python -m venv .venv
```

Activate the environment on Linux/macOS:

```bash
source .venv/bin/activate
```

Or in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Then install dependencies:

```bash
python -m pip install -r requirements.txt
```

The single-run trainer supports CPU and CUDA. GPU runs require a CUDA-enabled
PyTorch installation. The suite launchers use CUDA devices selected with
`--gpus`; use `--gpus 0` for one GPU or `--gpus 0 1` for two.

## Data

Friedman and the interaction-recovery datasets are generated in code. Concrete
is included under `data/concrete+compressive+strength/`. Download the remaining
datasets to the paths in the [data guide](data/raw/README.md) before training.
The loaders read local files; they do not download datasets automatically.

## Training

### Synthetic interaction recovery

For example, train on the higher-order task:

```bash
python -m pure_resmlp.training.run_experiment --experiment jocs_synthetic_v1 --dataset synthetic_higher_order --model cv_pure --seed 0 --split-seed 0 --hidden-dim 64 --epochs 100 --patience 15 --batch-size 256
```

Use `synthetic_pairwise` for the pairwise task. Both tasks compare `mlp`,
`resmlp`, `rv_pure`, and `cv_pure` at width 64 with training seeds 0–2. The
higher-order task also includes CV-MLP through the complex-baseline launcher
below. Data-generation settings are in
[`synthetic.py`](pure_resmlp/data/synthetic.py).

### Scientific regression

For example, train CV-PURe on Friedman:

```bash
python -m pure_resmlp.training.run_experiment --experiment jocs_benchmark_v1 --dataset friedman --model cv_pure --seed 0 --split-seed 0 --hidden-dim 64 --epochs 200 --patience 20 --batch-size 128
```

The four-model grid uses `mlp`, `resmlp`, `rv_pure`, and `cv_pure` on
`friedman`, `concrete`, `california`, `energy`, `yacht`, and `auto_mpg`, each with
training seeds 0–4. Real models use width 128; CV-PURe uses width 64. Energy
uses heating load (`--energy-target Y1`). The default learning rate is 0.001.

Scientific CV-MLP is a separate addition at width 64 under the same scientific
training settings. Its launcher first checks the existing CV-PURe baseline
configurations and split indices:

```bash
python -m pure_resmlp.training.run_scientific_cv_mlp --gpus 0
```

This adds 30 CV-MLP runs. Its five-model results are in
[`jocs_scientific_cv_mlp_v1`](artifacts/reports/jocs_scientific_cv_mlp_v1/summary.md);
the original four-model tables remain in `jocs_v1`.

### Context-aware prediction

For example, train RV-PURe on Bike Sharing:

```bash
python -m pure_resmlp.training.run_experiment --experiment jocs_bike_context_v1 --dataset bike_sharing --model rv_pure --seed 0 --hidden-dim 64 --epochs 100 --patience 15 --batch-size 256
```

Use `jocs_parkinsons_context_v1` and `parkinsons_telemonitoring` for Parkinsons.
Both context grids use `mlp_ordinary`, `mlp_context`, `mlp_explicit_cross`,
`dcnv2_context`, `rv_pure`, and `cv_pure`, with width 64 and training seeds 0–4.
The complex-baseline launcher adds CV-MLP for higher-order recovery and both
context datasets, and runs or reuses the Bike CV-PURe comparisons:

```bash
python -m pure_resmlp.training.run_cv_baselines --gpus 0
```

Bike ablations use RV-PURe with the same context settings and training seeds
0–2. Set the following experiment name and `--context-ablation` value:

| Experiment | Ablation |
|---|---|
| `jocs_bike_ablation_no_time_v1` | `bike_no_time` |
| `jocs_bike_ablation_no_calendar_v1` | `bike_no_calendar` |
| `jocs_bike_ablation_no_weather_v1` | `bike_no_weather` |

The examples above run one dataset/model/seed combination. Repeat them over
the indicated grids to regenerate the complete studies. The `--experiment`
names identify the directories expected by the summary scripts.

## Evaluation

Evaluation requires the saved run artifacts and source datasets. A fresh clone
contains result summaries, but not trained checkpoints.

### Interaction analysis

Evaluate a synthetic checkpoint, then repeat attribution sampling:

```bash
python -m pure_resmlp.evaluation.run_interactions artifacts/runs/jocs_synthetic_v1/synthetic_higher_order/cv_pure/seed_0 --n-samples 256 --seed 0
python -m pure_resmlp.evaluation.run_interactions artifacts/runs/jocs_synthetic_v1/synthetic_higher_order/cv_pure/seed_0 --n-samples 256 --resample-seeds 0 1 2 3 4
```

The same evaluator computes grouped interactions for context checkpoints.
Use sampling seed 0 for the context maps. Recovery resampling applies to
synthetic tasks with known interacting pairs.

### Input-noise robustness

```bash
python -m pure_resmlp.evaluation.evaluate_noise_robustness
```

Noise evaluation uses levels 0, 0.02, 0.05, and 0.10 without retraining.
It evaluates four scientific models and all seven context models. The
suite-level AUC compares the six models with context inputs; MLP-ordinary
appears in the dataset-level outputs. Scientific CV-MLP is not included in
this evaluation. See the
[suite-level AUC results](artifacts/reports/jocs_v1/robustness/noise_robustness_overall_auc.csv).

### Sample efficiency

```bash
python -m pure_resmlp.training.run_sample_efficiency_suite --gpus 0
python -m pure_resmlp.evaluation.summarize_sample_efficiency
```

Training adds nested 10%, 25%, and 50% subsets and reuses the existing 100%
runs. The comparison uses four models on Friedman and Concrete and six on
Parkinsons, including MLP-ordinary; CV-MLP is not part of the low-data study.
See the [AULC results](artifacts/reports/jocs_v1/sample_efficiency/sample_efficiency_aulc.csv).

### Inference efficiency

```bash
python -m pure_resmlp.evaluation.benchmark_inference --device cuda:0
```

The benchmark measures synchronized, repeated full-test inference on Friedman,
Yacht, and Auto MPG. The committed
[latency results](artifacts/reports/jocs_v1/inference_efficiency/inference_timings_summary.csv)
were measured on a single NVIDIA Tesla V100-SXM2-32GB.

### Summaries, figures, and checkpoint checks

After completing all required training runs and interaction analyses:

```bash
python -m pure_resmlp.evaluation.summarize_jocs --runs-root artifacts/runs --output-dir artifacts/reports/jocs_v1
python -m pure_resmlp.evaluation.redraw_jocs_figures
python -m pure_resmlp.evaluation.audit_checkpoints --runs-root artifacts/runs --strict
python -m pure_resmlp.evaluation.audit_scientific_cv_mlp --device cuda:0
```

The main summarizer expects the complete 226-run grid and its interaction
outputs. Figure redrawing also requires the noise and sample-efficiency
summaries. Scientific CV-MLP results and their checkpoint audit are separate.
Use `--help` on any entry point for its options.

## Tests

After dataset setup, run the full test suite:

```bash
python -m pytest tests -q
```

Architecture, synthetic interaction, and recording checks can run without
external datasets:

```bash
python -m pytest tests/test_zero_initialization.py tests/test_interactions.py tests/test_run_recording.py tests/test_scientific_cv_mlp.py -q
```

## Repository layout and outputs

```text
pure_resmlp/
  data/          Dataset loading, splits, and synthetic functions
  models/        Shared network implementations
  training/      Unified trainer and suite launchers
  evaluation/    Interaction analysis, robustness, summaries, and figures
  experiment/    Run configuration and artifact recording
tests/           Model, data, protocol, and recording tests
data/            Included Concrete data and local dataset downloads
artifacts/
  runs/          Local checkpoints and per-run outputs
  reports/       Versioned result summaries and evaluation protocols
```

Each run saves configuration, environment information, split indices, training
history, predictions, metrics, a checkpoint, and a completion manifest under
`artifacts/runs/<experiment>/<dataset>/<model>/seed_<seed>/`.

Git includes the experiment code, dependencies, tests, data instructions, and
selected result summaries. Downloaded datasets, checkpoints, generated figures,
and temporary files remain local. Manuscripts and submission materials are
outside the published repository scope.

## Citation

For the original ICCS 2026 formulation, cite:

```bibtex
@inproceedings{li2026modeling,
  author    = {Li, Ziyuan and Jaekel, Uwe and Dellen, Babette},
  title     = {Modeling Nonlinear Feature Interactions with Product-Unit Residual Networks},
  booktitle = {Computational Science -- ICCS 2026},
  year      = {2026},
  series    = {Lecture Notes in Computer Science},
  volume    = {16783},
  pages     = {80--94},
  publisher = {Springer Nature Switzerland},
  doi       = {10.1007/978-3-032-29921-5_6}
}
```
