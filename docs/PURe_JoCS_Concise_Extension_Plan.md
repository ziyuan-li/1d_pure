# Modeling Nonlinear Feature Interactions with Product-Unit Residual Networks

## Task 1: Scientific Regression Benchmark

**Plan.** Evaluate MLP, ResMLP, RV-PURe, and CV-PURe on Friedman, Concrete, California Housing, Energy Efficiency, Yacht Hydrodynamics, and Auto MPG. Use one fixed data split and five training seeds, and report test RMSE as mean ± sample SD.

**Previous version.** **Partly included.** It evaluated Friedman, Concrete, and California Housing, but did not include the other three datasets or the finalized four-model comparison.


## Task 2: Ground-Truth Interaction Recovery

**Plan.** Use pairwise and higher-order synthetic tasks with known interactions. Report Precision@K, Recall@K, and mean rank across repeated attribution samples. Include a direct CV-MLP versus CV-PURe control for the higher-order task.

**Previous version.** **Not included.**


## Task 3: Context-Aware Prediction

**Plan.** Study Bike Sharing and Parkinsons Telemonitoring using ordinary MLP, context-aware MLP, explicit-cross MLP, DCNv2, RV-PURe, CV-MLP, and CV-PURe. Report MAE, RMSE, and \(R^2\) over five training seeds. Use targeted context ablations for Bike Sharing and direct CV-MLP versus CV-PURe controls for both datasets.

**Previous version.** **Not included.**

## Task 4: Interpretability

**Plan.** Use the synthetic tasks to evaluate whether the correct interactions are recovered, and use finite-difference interaction maps to examine feature-context associations in the two real context datasets. These analyses answer different questions and are reported separately.

**Previous version.** **Partly included.** It used SHAP maps, entropy, and concentration statistics, but did not separate ground-truth interaction correctness from real-data association analysis.

## Task 5: Robustness

**Plan.** Apply Gaussian noise to ordinary continuous features at \(\sigma\in\{0,0.02,0.05,0.10\}\). Use target-SD-normalized noisy-performance AUC as the main summary metric and report the full noise curves.

**Previous version.** **Partly included.** It compared clean performance with only one noise level, \(\sigma=0.05\), and did not report a multi-level AUC.

## Task 6: Sample Efficiency

**Plan.** Evaluate nested 10%, 25%, 50%, and 100% training subsets on Friedman, Concrete, and Parkinsons. Report performance at each fraction and summarize the complete learning curve with RMSE AULC.

**Previous version.** **Partly included.** It studied only Concrete, using 5%, 10%, 20%, 40%, 60%, and 80% of the training data.

## Task 7: Computational Efficiency

**Plan.** Report synchronized, repeated full-test inference latency on a single NVIDIA V100. Discuss model capacity separately in the text so that latency and parameterization are not conflated.

**Previous version.** **Partly included.** It discussed parameter budgets but did not provide a dedicated synchronized inference-latency benchmark.
