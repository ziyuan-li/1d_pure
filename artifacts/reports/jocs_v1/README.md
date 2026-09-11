# Experiment results

Results for **Learning Nonlinear Interactions: Prediction and Interpretation
with Product-Unit Residual Networks**. The entries below follow the experiment
order in the manuscript. CSV filenames are stable identifiers, not manuscript
table numbers.

| Experiment | Summary | Underlying records |
|---|---|---|
| Ground-truth interaction recovery | [Prediction](table3_synthetic.csv), [recovery across attribution seeds](table3_synthetic_interaction_robustness.csv) | [Training runs](all_runs.csv), [attribution resamples](interaction_resamples.csv) |
| Scientific regression | [Five-model benchmark](table1_benchmark.csv) | [Training runs](all_runs.csv) |
| Context-aware prediction | [Seven-model comparison](table2_context_main.csv), [Bike ablations](table2_context_ablations.csv), [ablation effects](table2_context_ablation_paired_effects.csv) | [Training runs](all_runs.csv), [context-map scores](context_interactions.csv) |
| Input-noise robustness | [Noise curves](robustness/noise_robustness_summary.csv), [suite AUC](robustness/noise_robustness_overall_auc.csv), [paired effects](robustness/noise_robustness_auc_effects.csv) | [Corruption realizations](robustness/noise_robustness_raw.csv), [per-training-seed averages](robustness/noise_robustness_training_seed.csv) |
| Sample efficiency | [Learning curves](sample_efficiency/sample_efficiency_summary.csv), [AULC](sample_efficiency/sample_efficiency_aulc.csv), [paired AULC effects](sample_efficiency/sample_efficiency_aulc_effects.csv) | [Per-fraction training results](sample_efficiency/sample_efficiency_raw.csv) |
| Inference efficiency | [V100 latency](inference_efficiency/inference_timings_summary.csv) | [Per-checkpoint timings](inference_efficiency/inference_timings_raw.csv) |

[Prediction-error comparisons](paired_effects_bootstrap.csv) and
[direct CV-MLP/CV-PURe comparisons](cv_architecture_effects.csv) report paired
bootstrap intervals. The [combined summary](summary.md) provides readable
prediction and recovery tables. Core software versions and hardware are
recorded in [environment.json](environment.json).

## Models and repetitions

- Scientific prediction: MLP, ResMLP, RV-PURe, CV-MLP, and CV-PURe on six
  datasets, training seeds 0–4, split seed 0.
- Synthetic recovery: four models on the pairwise task and five on the
  higher-order task, with CV-MLP in the latter. Training seeds are 0–2;
  attribution sampling seeds are 0–4 for each checkpoint.
- Context prediction: seven models on Bike Sharing and Parkinsons, training
  seeds 0–4. Bike ablations retrain RV-PURe with seeds 0–2.
- Noise: four scientific models, without CV-MLP, and all seven context models.
  Suite-level context AUC includes the six models receiving context inputs;
  MLP-ordinary remains in dataset-level results.
- Sample efficiency: four scientific models on Friedman and Concrete, and
  six context models on Parkinsons, without CV-MLP.
- Inference: four scientific models on Friedman, Yacht, and Auto MPG.

`all_runs.csv` contains 256 distinct main/ablation training runs. The 30
scientific CV-MLP runs are included in this total. Their original launcher
records remain under `../jocs_scientific_cv_mlp_v1/`; the table in this directory
is the consolidated scientific comparison.

`interaction_resamples.csv` contains 135 attribution evaluations: 27 synthetic
checkpoints, each evaluated with five sampling seeds. The recovery summary uses
all these samples. `table3_synthetic.csv` retains sampling-seed-0 interaction
statistics used in the synthetic figure, alongside predictive metrics across
three training seeds. Repeated attribution samples are not independent
training runs.

`context_interactions.csv` contains 1,440 ordinary-feature/context scores from
30 checkpoints: three displayed models, two datasets, and five training seeds.
Sampling seed 0 is used throughout. Each displayed map averages the five
checkpoints of one model on one dataset.

## Metrics

MAE and RMSE use original target units unless the column identifies normalized
error. Standard deviations are sample SD across training seeds. Noise AUC is
interval-averaged, target-SD-normalized RMSE with equal dataset weighting within
each suite. `performance_auc` is the paper's primary noise summary;
`degradation_auc` and percentage-change columns are secondary diagnostics.
Learning-curve AULC uses original-unit RMSE and is compared within each dataset.
The three evaluation subdirectories include `protocol.json` with their settings.

Inference records contain each checkpoint's median, mean, and sample SD over
100 timed evaluations. The paper summarizes the checkpoint medians across five
training seeds; individual evaluation durations are not included.

## Regeneration

From the repository root, after installing dependencies:

```bash
python -m pure_resmlp.evaluation.rebuild_reports
python -m pure_resmlp.evaluation.redraw_jocs_figures --report-dir artifacts/reports/reproduced
```

These commands use only the included result records and write to a separate
directory. They do not train or reevaluate models. To rerun the experiments
themselves, follow the training and evaluation commands in the repository
README. Checkpoints, prediction arrays, manuscript files, and rendered figures
remain outside the Git allowlist.
