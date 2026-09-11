# PURe experiment results

Summary of 256 saved training runs.

## Synthetic interaction recovery (sampling seed 0)

| Dataset | Model | RMSE | Precision@K | Recall@K | Mean rank |
| --- | --- | --- | --- | --- | --- |
| synthetic_higher_order | cv_mlp | 0.1377 ± 0.0023 | 0.3333 ± 0.0000 | 0.3333 ± 0.0000 | 117.5556 ± 39.1824 |
| synthetic_higher_order | cv_pure | 0.0735 ± 0.0071 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 5.0000 ± 0.0000 |
| synthetic_higher_order | mlp | 0.1346 ± 0.0025 | 0.3333 ± 0.0000 | 0.3333 ± 0.0000 | 86.3704 ± 39.6516 |
| synthetic_higher_order | resmlp | 0.1344 ± 0.0044 | 0.3333 ± 0.0000 | 0.3333 ± 0.0000 | 81.9630 ± 21.0708 |
| synthetic_higher_order | rv_pure | 0.1368 ± 0.0025 | 0.3333 ± 0.0000 | 0.3333 ± 0.0000 | 78.2963 ± 11.3493 |
| synthetic_pairwise | cv_pure | 0.0415 ± 0.0013 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.5000 ± 0.0000 |
| synthetic_pairwise | mlp | 0.0442 ± 0.0005 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.5000 ± 0.0000 |
| synthetic_pairwise | resmlp | 0.0449 ± 0.0017 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.5000 ± 0.0000 |
| synthetic_pairwise | rv_pure | 0.0390 ± 0.0010 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.5000 ± 0.0000 |

## Synthetic interaction recovery (sampling seeds 0–4)

Five reference-sampling seeds per saved model; these are attribution resamples, not additional training runs.

| Dataset | Model | Resamples | Precision@K | Recall@K | Mean rank | Exact recovery rate |
| --- | --- | --- | --- | --- | --- | --- |
| synthetic_higher_order | cv_mlp | 15 | 0.3333 ± 0.0000 | 0.3333 ± 0.0000 | 115.3481 ± 22.7327 | 0.0000 |
| synthetic_higher_order | cv_pure | 15 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 5.0000 ± 0.0000 | 1.0000 |
| synthetic_higher_order | mlp | 15 | 0.3333 ± 0.0000 | 0.3333 ± 0.0000 | 91.2815 ± 34.3714 | 0.0000 |
| synthetic_higher_order | resmlp | 15 | 0.3333 ± 0.0000 | 0.3333 ± 0.0000 | 78.1185 ± 13.7610 | 0.0000 |
| synthetic_higher_order | rv_pure | 15 | 0.3333 ± 0.0000 | 0.3333 ± 0.0000 | 82.5630 ± 13.9483 | 0.0000 |
| synthetic_pairwise | cv_pure | 15 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.5000 ± 0.0000 | 1.0000 |
| synthetic_pairwise | mlp | 15 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.5000 ± 0.0000 | 1.0000 |
| synthetic_pairwise | resmlp | 15 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.5000 ± 0.0000 | 1.0000 |
| synthetic_pairwise | rv_pure | 15 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.5000 ± 0.0000 | 1.0000 |

## Scientific regression

| Dataset | Model | MAE | RMSE | R² |
| --- | --- | --- | --- | --- |
| auto_mpg | cv_mlp | 2.1058 ± 0.0684 | 3.3135 ± 0.0657 | 0.8266 ± 0.0069 |
| auto_mpg | cv_pure | 2.0882 ± 0.0554 | 3.2390 ± 0.0555 | 0.8343 ± 0.0057 |
| auto_mpg | mlp | 2.1343 ± 0.0890 | 3.3808 ± 0.1562 | 0.8192 ± 0.0165 |
| auto_mpg | resmlp | 2.1290 ± 0.0468 | 3.3479 ± 0.0972 | 0.8229 ± 0.0104 |
| auto_mpg | rv_pure | 2.1373 ± 0.0551 | 3.2756 ± 0.0402 | 0.8305 ± 0.0041 |
| california | cv_mlp | 0.3337 ± 0.0074 | 0.5048 ± 0.0105 | 0.8017 ± 0.0083 |
| california | cv_pure | 0.3383 ± 0.0071 | 0.5055 ± 0.0079 | 0.8012 ± 0.0062 |
| california | mlp | 0.3387 ± 0.0056 | 0.5072 ± 0.0068 | 0.7999 ± 0.0054 |
| california | resmlp | 0.3293 ± 0.0098 | 0.5053 ± 0.0090 | 0.8013 ± 0.0071 |
| california | rv_pure | 0.3327 ± 0.0040 | 0.4996 ± 0.0051 | 0.8058 ± 0.0040 |
| concrete | cv_mlp | 4.0168 ± 0.0719 | 5.8196 ± 0.1231 | 0.8768 ± 0.0053 |
| concrete | cv_pure | 3.9903 ± 0.2120 | 5.6590 ± 0.2066 | 0.8834 ± 0.0084 |
| concrete | mlp | 4.0300 ± 0.1118 | 5.7824 ± 0.0833 | 0.8784 ± 0.0035 |
| concrete | resmlp | 3.9847 ± 0.1632 | 5.7020 ± 0.1478 | 0.8817 ± 0.0061 |
| concrete | rv_pure | 3.9379 ± 0.1416 | 5.6500 ± 0.1301 | 0.8839 ± 0.0053 |
| energy | cv_mlp | 0.3918 ± 0.0383 | 0.5741 ± 0.0551 | 0.9969 ± 0.0006 |
| energy | cv_pure | 0.3707 ± 0.0107 | 0.5407 ± 0.0186 | 0.9973 ± 0.0002 |
| energy | mlp | 0.3981 ± 0.0435 | 0.5792 ± 0.0606 | 0.9968 ± 0.0007 |
| energy | resmlp | 0.3881 ± 0.0280 | 0.5615 ± 0.0277 | 0.9970 ± 0.0003 |
| energy | rv_pure | 0.4175 ± 0.0165 | 0.6082 ± 0.0248 | 0.9965 ± 0.0003 |
| friedman | cv_mlp | 0.1389 ± 0.0079 | 0.1824 ± 0.0114 | 0.9986 ± 0.0002 |
| friedman | cv_pure | 0.0629 ± 0.0078 | 0.0829 ± 0.0092 | 0.9997 ± 0.0001 |
| friedman | mlp | 0.1483 ± 0.0115 | 0.1956 ± 0.0137 | 0.9984 ± 0.0002 |
| friedman | resmlp | 0.1252 ± 0.0116 | 0.1704 ± 0.0171 | 0.9988 ± 0.0002 |
| friedman | rv_pure | 0.1239 ± 0.0144 | 0.1625 ± 0.0193 | 0.9989 ± 0.0003 |
| yacht | cv_mlp | 0.5228 ± 0.0993 | 0.7877 ± 0.1501 | 0.9962 ± 0.0013 |
| yacht | cv_pure | 0.4326 ± 0.0806 | 0.5762 ± 0.1199 | 0.9979 ± 0.0008 |
| yacht | mlp | 0.6325 ± 0.0627 | 0.9214 ± 0.0960 | 0.9948 ± 0.0011 |
| yacht | resmlp | 0.7588 ± 0.2109 | 1.1027 ± 0.1568 | 0.9926 ± 0.0021 |
| yacht | rv_pure | 0.4679 ± 0.0464 | 0.7516 ± 0.1291 | 0.9965 ± 0.0013 |

## Context-aware prediction

| Dataset | Model | MAE | RMSE | R² | R²<0 runs |
| --- | --- | --- | --- | --- | --- |
| bike_sharing | cv_mlp | 65.9166 ± 8.1121 | 95.7014 ± 11.3234 | 0.8095 ± 0.0453 | 0/5 |
| bike_sharing | cv_pure | 61.3939 ± 5.4136 | 88.4508 ± 6.9630 | 0.8383 ± 0.0243 | 0/5 |
| bike_sharing | dcnv2_context | 84.0661 ± 22.8344 | 111.1614 ± 27.2730 | 0.7336 ± 0.1314 | 0/5 |
| bike_sharing | mlp_context | 70.6384 ± 6.3405 | 101.0326 ± 9.2380 | 0.7886 ± 0.0391 | 0/5 |
| bike_sharing | mlp_explicit_cross | 77.9887 ± 4.0459 | 115.9961 ± 6.5596 | 0.7225 ± 0.0319 | 0/5 |
| bike_sharing | mlp_ordinary | 145.7148 ± 1.0937 | 207.4167 ± 1.9811 | 0.1150 ± 0.0168 | 0/5 |
| bike_sharing | rv_pure | 65.6039 ± 8.4749 | 93.6130 ± 14.0155 | 0.8165 ± 0.0553 | 0/5 |
| parkinsons_telemonitoring | cv_mlp | 3.2353 ± 0.0783 | 4.3832 ± 0.1532 | 0.8414 ± 0.0110 | 0/5 |
| parkinsons_telemonitoring | cv_pure | 3.2082 ± 0.0557 | 4.1932 ± 0.0803 | 0.8549 ± 0.0055 | 0/5 |
| parkinsons_telemonitoring | dcnv2_context | 3.7143 ± 0.2688 | 5.0823 ± 0.3917 | 0.7859 ± 0.0319 | 0/5 |
| parkinsons_telemonitoring | mlp_context | 3.4560 ± 0.1384 | 4.5896 ± 0.1489 | 0.8261 ± 0.0112 | 0/5 |
| parkinsons_telemonitoring | mlp_explicit_cross | 3.4705 ± 0.0864 | 4.4101 ± 0.1337 | 0.8395 ± 0.0098 | 0/5 |
| parkinsons_telemonitoring | mlp_ordinary | 7.6034 ± 0.1325 | 9.7073 ± 0.0964 | 0.2227 ± 0.0154 | 0/5 |
| parkinsons_telemonitoring | rv_pure | 3.4633 ± 0.1172 | 4.5058 ± 0.0716 | 0.8325 ± 0.0054 | 0/5 |

## Context ablations

| Dataset | Ablation | MAE | RMSE | R² |
| --- | --- | --- | --- | --- |
| bike_sharing | bike_no_calendar | 96.3218 ± 13.4011 | 136.8392 ± 13.4746 | 0.6123 ± 0.0777 |
| bike_sharing | bike_no_time | 141.6587 ± 1.8504 | 187.8750 ± 1.5250 | 0.2739 ± 0.0118 |
| bike_sharing | bike_no_weather | 62.9419 ± 3.6709 | 92.9997 ± 5.6063 | 0.8217 ± 0.0211 |
