# Experiment results

This directory contains summaries generated from the saved experiment runs.
The CSV files retain the values and columns of the existing reports.

The [scientific CV-MLP addition](../jocs_scientific_cv_mlp_v1/summary.md)
reports all five scientific models separately. `table1_benchmark.csv` below
contains the original four-model comparison. The addition includes 30 new
CV-MLP runs and reuses the 120 scientific baseline runs; it does not add
CV-MLP to the scientific noise or sample-efficiency suites.

| Files | Contents |
|---|---|
| `table1_benchmark.csv` | Scientific regression metrics |
| `table2_context_main.csv` | Bike Sharing and Parkinsons prediction metrics |
| `table2_context_ablations.csv` | Bike Sharing context ablations |
| `table2_context_ablation_paired_effects.csv` | Paired ablation effects |
| `table3_synthetic.csv` | Synthetic prediction and interaction recovery |
| `table3_synthetic_interaction_robustness.csv` | Repeated attribution sampling |
| `paired_effects_bootstrap.csv` | Paired prediction-error comparisons |
| `cv_architecture_effects.csv` | Direct complex-valued model comparisons |
| `robustness/*.csv` | Input-noise summaries, AUC, and paired effects |
| `sample_efficiency/*.csv` | Learning curves, AULC, and paired effects |
| `inference_efficiency/*.csv` | Repeated full-test inference latency |

The three evaluation subdirectories include `protocol.json` with their settings.
MAE and RMSE use original target units unless a column identifies a
normalized error. Noise-performance AUC uses target-SD-normalized RMSE and equal
dataset weighting within each suite. Learning-curve AULC uses original-unit RMSE
and is compared within each dataset.

Scientific and context prediction summaries use five training seeds. Synthetic
training and Bike ablations use three. Attribution resamples are reported
separately from training repetitions.

## Regeneration

Run the training and evaluation commands in the repository README to produce the
underlying artifacts. Then regenerate the main and sample-efficiency summaries:

```bash
python -m pure_resmlp.evaluation.summarize_jocs --runs-root artifacts/runs --output-dir artifacts/reports/jocs_v1
python -m pure_resmlp.evaluation.summarize_sample_efficiency
```

The noise evaluator and inference benchmark write their summaries directly.
Checkpoints, per-run logs, raw prediction files, and rendered figures are not
included in this directory's Git allowlist.
