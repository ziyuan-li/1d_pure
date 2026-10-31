# Paper–code correspondence

This file maps the 31 October 2026 journal revision of *Learning Nonlinear
Interactions: Prediction and Interpretation with Product-Unit Residual Networks*
to the released implementation. Paths below are relative to the repository root.

## Architecture and methods

| Manuscript | Implementation | Correspondence |
|---|---|---|
| Section 3.1: real product unit and residual block | [models/pure.py](pure_resmlp/models/pure.py): `ProductUnit1D`, `PUResidualBlock1D` | `exp(W log(ReLU(a) + 1))`; bias-free exponent matrix; `h + FC2(PU(FC1(h)))`. |
| Section 3.2: RV-PURe | [models/pure.py](pure_resmlp/models/pure.py): `RealPUResNet1D` | Input affine + ReLU, two residual blocks, scalar affine output. |
| Section 3.3: complex product unit and CV-PURe | [models/complex_pure.py](pure_resmlp/models/complex_pure.py) | Real/imaginary ReLU with offset `1/sqrt(2)`, principal complex logarithm, complex exponent matrix, two residual blocks; no activation after the input affine map. |
| Section 3.3: signed readout | [models/complex_readout.py](pure_resmlp/models/complex_readout.py): `ComplexRealProjection` | `alpha * Re(o) + beta * Im(o) + b`; all three parameters are learnable. CV-PURe and CV-MLP use this readout in every experiment. |
| Section 3.4: mixed finite-difference scores | [evaluation/finite_difference_interactions.py](pure_resmlp/evaluation/finite_difference_interactions.py) | Four-point contrasts, absolute averaging, grouped encoded columns, shared test/reference sampling, recovery metrics. |
| Section 4.1: MLP and CV-MLP | [models/mlp.py](pure_resmlp/models/mlp.py), [models/complex_mlp.py](pure_resmlp/models/complex_mlp.py) | Seven hidden affine layers; ReLU or componentwise complex ReLU; scalar output. |
| Section 4.1: ResMLP | [models/residual_mlp.py](pure_resmlp/models/residual_mlp.py) | Input ReLU, two three-affine residual branches, final ReLU and scalar output. |
| Section 4.1: training and metrics | [training/run_experiment.py](pure_resmlp/training/run_experiment.py) | Adam, MSE loss, seeded shuffled minibatches, validation-based early stopping, restoration of the best checkpoint, inverse-scaled MAE/RMSE. |
| Section 4.3: shortcut ablation | [models/shortcut.py](pure_resmlp/models/shortcut.py): `ShortcutAblation` | Removes only the identity additions; retains the same layers, initialization draws and optimizer settings. |
| Section 5.1: concentration and stability | [evaluation/scientific_interaction_summary.py](pure_resmlp/evaluation/scientific_interaction_summary.py) | Top-10 entropy/share, Friedman true-pair share, training/sampling Jaccard overlaps, paired bootstrap intervals. |
| Section 5.2: synthetic response functions | [data/synthetic.py](pure_resmlp/data/synthetic.py) | Two pairwise products or third-/fourth-order products, 20 distractors, generation seed 0 and 5% response noise. |
| Section 5.3: context inputs and DCNv2 | [models/context.py](pure_resmlp/models/context.py), [training/run_experiment.py](pure_resmlp/training/run_experiment.py): `transform_input_matrix` | Ordinary/context selection, all explicit ordinary–context crosses, three cross layers and a parallel three-layer deep branch. |
| Section 6.1: feature noise | [evaluation/noise.py](pure_resmlp/evaluation/noise.py) | Gaussian corruption of eligible standardized ordinary columns, recomputed explicit crosses, corruption averaging and normalized AUC. |
| Section 6.2: learning curves | [data/datasets.py](pure_resmlp/data/datasets.py): `subset_prepared_training_data`; [summarize.py](pure_resmlp/summarize.py) | Nested subsets, refitted training scalers, fixed validation/test partitions, trapezoidal RMSE AULC. |
| Section 6.3: capacity and timing | [paper.py](pure_resmlp/paper.py): `verify`, `benchmark` | Complex parameters count as two real scalars; projection contributes three. Synchronized full-test timing uses batch size 128 and 100 timed passes after warm-up. |

## Initialization, preprocessing and seeds

- Both PURe blocks use independent PyTorch default initialization. Their final
  network output affine map is zero-initialized; the projection starts at
  `(alpha, beta, b) = (1, 0, 0)`. There is no zero-initialization switch for PU blocks.
- ResMLP separately zero-initializes the last affine map in its second residual
  branch and its network output layer. MLP, CV-MLP and DCNv2 retain default
  affine initialization. PURe and its ablations clip gradient norm at 1.0.
- Scientific tasks use real width 128 / complex width 64, at most 200 epochs,
  batch size 128 and patience 20. Synthetic/context tasks use width 64, at most
  100 epochs, batch size 256 and patience 15. All use Adam at learning rate 0.001.
- [data/datasets.py](pure_resmlp/data/datasets.py) fits input and target scalers
  using training rows only. Bike uses chronological 60/20/20 partitions;
  Parkinsons uses within-subject chronological groups without splitting equal
  timestamps. Context encodings and excluded target-related columns are defined here.
- Scientific cross-split performance uses split seeds 0–4 × training seeds 0–4.
  Fixed random partitions use split seed 0. Synthetic recovery and Bike context
  ablations use training seeds 0–2; other experiments use 0–4.
- Generation and nested-subset selection use seed 0. Interaction sampling and
  corruption use seeds 0–4; context maps use sampling seed 0. Bootstrap intervals
  use 10,000 resamples with seed 0. [paper.py](pure_resmlp/paper.py) enables
  deterministic algorithms, disables TF32 and selects highest float32 precision.

Exact per-run settings are in [protocol/runs.json](protocol/runs.json), seed roles
in [protocol/protocol.json](protocol/protocol.json), and saved scaler statistics
and partition indices in `protocol/preprocessing/` and `protocol/splits/`.
`window_2` in context run IDs denotes the final 60/20/20 partition.

## Tables and figures

| Item | Published source records | Implementation |
|---|---|---|
| Table 1: datasets and partitions | `protocol/preprocessing/`, `protocol/splits/` | `data/datasets.py`, `data/synthetic.py` |
| Table 2: cross-split performance | `records/random_split_by_split.csv`, `records/random_split_robustness.csv` | `summarize.py`: five training means per split, then five split means |
| Table 3: residual ablation | `records/table3_per_seed.csv`, `records/table3_summary.csv` | `models/shortcut.py`, `summarize.py` |
| Table 4: interaction structure/stability | `records/pairs_raw.csv`, `records/scientific_interactions/` | `evaluation/scientific_interaction_summary.py` |
| Table 5: synthetic recovery | `records/primary_summary.csv`, `records/recovery_raw.csv`, `records/synthetic_recovery_summary.csv` | `evaluation/finite_difference_interactions.py`, `summarize.py` |
| Table 6: Bike prediction | `records/primary_summary.csv` | `summarize.py` |
| Table 7: RV/CV Bike ablations | `records/table7_per_seed.csv`, `records/table7_full_context.csv`, `records/table7_summary.csv` | `training/run_experiment.py`, `summarize.py`; three paired training seeds |
| Table 8: noise AUC | `records/noise_raw.csv`, `records/noise_suite_auc.csv` | `evaluation/noise.py`, `summarize.py` |
| Table 9: learning AULC | `records/learning_curves.csv`, `records/learning_aulc_per_seed.csv`, `records/learning_aulc_summary.csv` | `summarize.py` |
| Table 10: capacity/latency | `protocol/runs.json`, `records/latency_per_seed.csv`, `records/latency_summary.csv` | `paper.py`, `summarize.py` |
| Figure 1: real PU block | `figures/pure_real_block_vertical.pdf` | `evaluation/style_paper_figures.py`: `block_diagram` |
| Figure 2: architectures | `figures/pure_network_architectures_vertical.pdf` | `evaluation/style_paper_figures.py`: `architecture_diagram` |
| Figure 3: scientific maps | `records/scientific_interactions/mean_maps.csv` | `evaluation/style_paper_figures.py`: `scientific_maps` |
| Figure 4: context maps | `records/context_interactions.csv` | `evaluation/style_paper_figures.py`: `context_maps` |
| Figure 5: noise curves | `records/noise_summary.csv` | `evaluation/style_paper_figures.py`: `noise_curves` |
| Figure 6: learning curves | `records/learning_curves.csv` | `evaluation/style_paper_figures.py`: `learning_curves` |

Implementation paths in the last column are under `pure_resmlp/`.
[tables.py](pure_resmlp/tables.py) writes all ten LaTeX tables, including bold best
values. The [README](README.md) gives the training, evaluation and regeneration
commands. Explanations are kept in Markdown; Python source contains no comments
or docstrings.

The release includes 1,215 distinct weight dictionaries, with repeated uses
across analyses sharing the same checkpoint. Each run records its released
checkpoint hash, original checkpoint hash, preprocessing reference and metrics.
[protocol/verification.json](protocol/verification.json) records the release
checks; `SHA256SUMS` covers the distributed files. Only generated synthetic
observations are included; see the [data guide](data/README.md) for external data.
