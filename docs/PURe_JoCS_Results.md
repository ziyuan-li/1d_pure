# PURe JoCS 正式实验结果与研发结论

更新时间：2026-09-02

## 1. 完整性

当前论文主结果包括 **226 个神经网络运行**：

| 实验组 | 完整运行 | 交互分析 | 状态 |
|---|---:|---:|---|
| 6 个 scientific datasets × 4 models × 5 training seeds，固定 split 0 | 120 | 不要求 | 已完成 |
| 2 个 synthetic tasks，原 4 models × 3 seeds；higher-order 新增 CV-MLP × 3 seeds | 27 | 27；另有 135 次 attribution resamples | 已完成 |
| Bike Sharing：7 models × 5 seeds | 35 | 35 | 已完成 |
| Parkinsons Telemonitoring：7 models × 5 seeds | 35 | 35 | 已完成 |
| Bike 三组消融 × 3 seeds | 9 | 9 | 已完成 |
| **合计** | **226** | **106** | **完整** |

此外完成两项确认性实验：复用 190 个主模型完成 **3,040 次 Gaussian feature-noise 推理**；在 Friedman、Concrete 和 Parkinsons 上新增 **210 个 low-data 训练**，并复用 70 个 100% 主实验运行形成 280 条样本效率结果。Higher-order、Bike 和 Parkinsons 共包含 13 个 CV-MLP 运行，均保存完整 checkpoint、预测、交互分析和运行工件。

所有运行均保存 config、environment、split、history、prediction、metrics、checkpoint 和 manifest；需要解释的 106 个主结果运行均保存 interaction matrix、ranked pairs 和 interaction metrics。原 24 个 synthetic 运行和新增的 3 个 higher-order CV-MLP 运行均保存五个 reference-sampling seeds 的 `interaction_robustness.json`，共 135 次 attribution resamples。未计算哈希。

## 2. Scientific benchmark 结论

以下均为固定 `split_seed=0`、training seeds 0–4 的 test RMSE mean ± SD：

| Dataset | MLP | ResMLP | RV-PURe | CV-PURe | 结论 |
|---|---:|---:|---:|---:|---|
| Friedman | 0.1956 ± 0.0137 | 0.1704 ± 0.0171 | 0.1625 ± 0.0193 | **0.0829 ± 0.0092** | CV-PURe 明显最佳 |
| Concrete | 5.7824 ± 0.0833 | 5.7020 ± 0.1478 | **5.6500 ± 0.1301** | 5.6590 ± 0.2066 | RV-PURe 最佳 |
| California | 0.5072 ± 0.0068 | 0.5053 ± 0.0090 | **0.4996 ± 0.0051** | 0.5055 ± 0.0079 | RV-PURe 小幅最佳 |
| Energy | 0.5792 ± 0.0606 | 0.5615 ± 0.0277 | 0.6082 ± 0.0248 | **0.5407 ± 0.0186** | CV-PURe 平均最佳，配对区间跨零 |
| Yacht | 0.9214 ± 0.0960 | 1.1027 ± 0.1568 | 0.7516 ± 0.1291 | **0.5762 ± 0.1199** | CV-PURe 明显最佳 |
| Auto MPG | 3.3808 ± 0.1562 | 3.3479 ± 0.0972 | 3.2756 ± 0.0402 | **3.2390 ± 0.0555** | CV-PURe 平均最佳，RV-PURe 改善更稳定 |

按 training seed 配对的 bootstrap 支持以下表述：

- Friedman：RV-PURe 相对 MLP 的 RMSE 改善为 0.0331，95% CI [0.0138, 0.0507]；CV-PURe 改善为 0.1126，95% CI [0.1014, 0.1239]。
- Concrete：RV-PURe 改善为 0.1324，95% CI [0.0110, 0.2538]。
- California：RV-PURe 改善为 0.0075，95% CI [0.0007, 0.0174]。
- Energy：CV-PURe 的平均改善为 0.0385，但 95% CI [-0.0048, 0.0925] 跨零，因此只表述为平均最佳。
- Yacht：RV-PURe 改善为 0.1698，95% CI [0.0549, 0.2848]；CV-PURe 改善为 0.3453，95% CI [0.2090, 0.5156]。
- Auto MPG：RV-PURe 改善为 0.1052，95% CI [0.0046, 0.1933]；CV-PURe 的平均改善为 0.1418，但 95% CI [-0.0070, 0.2709] 略跨零，因此只表述为平均最佳。

## 3. Ground-truth interaction recovery

| Task | Model | RMSE | Precision@K | Recall@K | True-pair mean rank |
|---|---|---:|---:|---:|---:|
| Pairwise | MLP | 0.0442 ± 0.0005 | 1.000 | 1.000 | 1.5 |
| Pairwise | ResMLP | 0.0449 ± 0.0017 | 1.000 | 1.000 | 1.5 |
| Pairwise | RV-PURe | **0.0390 ± 0.0010** | 1.000 | 1.000 | 1.5 |
| Pairwise | CV-PURe | 0.0415 ± 0.0013 | 1.000 | 1.000 | 1.5 |
| Higher-order | MLP | 0.1346 ± 0.0025 | 0.333 | 0.333 | 86.4 ± 39.7 |
| Higher-order | ResMLP | 0.1344 ± 0.0044 | 0.333 | 0.333 | 82.0 ± 21.1 |
| Higher-order | RV-PURe | 0.1368 ± 0.0025 | 0.333 | 0.333 | 78.3 ± 11.3 |
| Higher-order | CV-MLP | 0.1377 ± 0.0023 | 0.333 | 0.333 | 117.6 ± 39.2 |
| Higher-order | CV-PURe | **0.0735 ± 0.0071** | **1.000** | **1.000** | **5.0 ± 0.0** |

这是目前最强的新证据：CV-PURe 在高阶任务上同时改善预测，并恢复高阶乘法项诱导的全部真实 pairwise projections。相对 MLP 的 RMSE 配对改善为 0.0611，95% CI [0.0573, 0.0681]；相对 CV-MLP 的改善为 0.0643，[0.0578, 0.0745]。CV-MLP 与 CV-PURe 都使用 hidden dimension 64，真实自由度分别为 53,637 和 53,381，因此结果说明普通复值参数化本身不足以解释该优势，复值 Product Unit 是关键组成。

### 3.1 Interaction attribution sampling robustness

为排除 finite-difference reference sampling 偶然性，对每个已保存模型使用 sampling seeds 0–4 重复计算，每次仍使用 256 个 evaluation/reference pairs。每个模型因此有 3 个训练 seeds × 5 个 attribution resamples = 15 次恢复评估；这些重采样不计为新的训练运行。

| Task | Model | Attribution resamples | Precision@K | Recall@K | Mean rank | Exact recovery |
|---|---|---:|---:|---:|---:|---:|
| Pairwise | 全部四模型 | 各 15 | 1.000 | 1.000 | 1.5 | 15/15 |
| Higher-order | MLP | 15 | 0.333 | 0.333 | 91.3 ± 34.4 | 0/15 |
| Higher-order | ResMLP | 15 | 0.333 | 0.333 | 78.1 ± 13.8 | 0/15 |
| Higher-order | RV-PURe | 15 | 0.333 | 0.333 | 82.6 ± 13.9 | 0/15 |
| Higher-order | CV-MLP | 15 | 0.333 | 0.333 | 115.3 ± 22.7 | 0/15 |
| Higher-order | CV-PURe | 15 | **1.000** | **1.000** | **5.0 ± 0.0** | **15/15** |

mean rank 5.0 是九个真实 pairwise projections 恰好占据排名 1–9 时的理论最优均值。该稳健性检查说明 CV-PURe 的高阶结果不依赖某一次 reference sampling。严谨表述应是“恢复高阶乘法项诱导的 pairwise interaction projections”，而不是声称当前二阶 finite difference 直接识别了三阶或四阶交互阶数。

## 4. Context-aware 完整结果

### Bike Sharing

| Model | MAE | RMSE | \(R^2\) |
|---|---:|---:|---:|
| MLP-ordinary | 145.71 ± 1.09 | 207.42 ± 1.98 | 0.1150 ± 0.0168 |
| MLP-context | 70.64 ± 6.34 | 101.03 ± 9.24 | 0.7886 ± 0.0391 |
| MLP-explicit-cross | 77.99 ± 4.05 | 116.00 ± 6.56 | 0.7225 ± 0.0319 |
| DCNv2-context | 84.07 ± 22.83 | 111.16 ± 27.27 | 0.7336 ± 0.1314 |
| RV-PURe-context | 65.60 ± 8.47 | 93.61 ± 14.02 | 0.8165 ± 0.0553 |
| CV-MLP-context | 65.92 ± 8.11 | 95.70 ± 11.32 | 0.8095 ± 0.0453 |
| **CV-PURe-context** | **61.39 ± 5.41** | **88.45 ± 6.96** | **0.8383 ± 0.0243** |

CV-PURe 相对 MLP-context 的配对 MAE 改善为 **9.24**，95% bootstrap CI **[3.30, 14.46]**；RV-PURe 的改善为 **5.03**，95% CI **[0.35, 11.83]**。统一为 hidden 64、100 epochs、patience 15、batch size 256 后，CV-PURe 相对自由度近似相同的 CV-MLP 的 MAE/RMSE 平均改善分别为 4.52 和 7.25，但 95% CI 分别为 [-1.81, 10.46] 和 [-1.65, 16.15]。因此完整五 seed 结果支持 PURe 相对 MLP-context 的 Bike 预测优势；CV-PURe 的均值最好，但 Bike 单独不足以确认 Product Unit 相对普通复值网络的增量。

Bike 消融：

- no-time：MAE 141.66 ± 1.85；与同 seed 完整 RV-PURe 配对后恶化 72.35，95% CI [67.09, 75.40]，说明时间情境最关键；
- no-calendar：MAE 96.32 ± 13.40；配对恶化 27.01，95% CI [10.16, 45.23]，说明日历情境也有明确贡献；
- no-weather：MAE 62.94 ± 3.67；配对差值为 -6.37，95% CI [-17.63, 0.32]，不足以确认删除天气会改善预测。

### Parkinsons Telemonitoring

| Model | MAE | RMSE | \(R^2\) |
|---|---:|---:|---:|
| MLP-ordinary | 7.6034 ± 0.1325 | 9.7073 ± 0.0964 | 0.2227 ± 0.0154 |
| MLP-context | 3.4560 ± 0.1384 | 4.5896 ± 0.1489 | 0.8261 ± 0.0112 |
| MLP-explicit-cross | 3.4705 ± 0.0864 | 4.4101 ± 0.1337 | 0.8395 ± 0.0098 |
| DCNv2-context | 3.7143 ± 0.2688 | 5.0823 ± 0.3917 | 0.7859 ± 0.0319 |
| RV-PURe-context | 3.4633 ± 0.1172 | 4.5058 ± 0.0716 | 0.8325 ± 0.0054 |
| CV-MLP-context | 3.2353 ± 0.0783 | 4.3832 ± 0.1532 | 0.8414 ± 0.0109 |
| **CV-PURe-context** | **3.2082 ± 0.0557** | **4.1932 ± 0.0803** | **0.8549 ± 0.0055** |

CV-PURe 在三个指标上均最佳。相对 MLP-context 的配对 MAE 改善为 **0.2478（7.17%）**，95% bootstrap CI **[0.1479, 0.3451]**；RMSE 改善为 **0.3964**，95% CI **[0.2299, 0.5461]**。相对 CV-MLP，CV-PURe 的 RMSE 改善为 **0.1900**，95% CI **[0.0744, 0.2877]**；MAE 平均改善为 0.0271，但区间 [-0.0375, 0.0880] 跨零。MLP-ordinary 与所有 context-aware 模型之间的巨大差距也表明 subject、age、sex 和 test time 对纵向预测不可缺少。该数据集使用每位受试者内部的时间顺序 60%/20%/20% 划分，并排除另一个临床目标 `motor_UPDRS`。

35 份有限差分交互分析已经完成。`subject–test_time` 在六个 context-aware 模型的全部 30 个 training-seed 运行中均排名第 1，说明最稳定的结构是个体病程随随访时间变化。CV-PURe 的跨 seed 平均 ordinary–context 排名前五项依次为 `DFA–test_time`、`Shimmer:APQ3–test_time`、`Shimmer:DDA–test_time`、`Jitter(Abs)–test_time` 和 `Shimmer–test_time`，形成一致的语音测量–随访时间结构。该结果是模型层面的关联与敏感性证据，不作因果解释。

## 5. Gaussian feature-noise 鲁棒性

协议固定为标准化特征空间中的 \(\sigma\in\{0,0.02,0.05,0.10\}\)。每个噪声强度使用 corruption seeds 0–4；先在每个 training seed 内平均五次污染，再跨 training seeds 0–4 计算 mean、SD 和配对 bootstrap CI，避免把相关污染重复当成独立样本。只污染连续 ordinary features，context 和 one-hot 列保持合法不变；\(\sigma=0\) 的 190 个结果与主实验对齐，最大 RMSE 差约为 \(10^{-7}\)。

主比较使用 **带噪性能 AUC**：对每个 training seed 的 test RMSE 先除以同一数据集固定测试目标的标准差，再对 \(\sigma\in[0,0.1]\) 做梯形积分并除以区间长度；随后先在同一 seed 内等权平均数据集，再跨五个 training seeds 汇总。该指标不会用模型自身的 clean RMSE 作分母，也不会因几个 \(\sigma\) 间隔不相等而错误加权，数值越小越好。

| Suite | Model | 带噪性能 AUC | 95% bootstrap CI |
|---|---|---:|---:|
| Scientific | MLP | 0.2720 ± 0.0054 | [0.2673, 0.2758] |
| Scientific | ResMLP | 0.2711 ± 0.0039 | [0.2679, 0.2739] |
| Scientific | RV-PURe | 0.2699 ± 0.0021 | [0.2684, 0.2716] |
| Scientific | **CV-PURe** | **0.2649 ± 0.0040** | **[0.2617, 0.2680]** |
| Context | MLP-context | 0.4376 ± 0.0252 | [0.4195, 0.4579] |
| Context | MLP-explicit-cross | 0.4641 ± 0.0114 | [0.4546, 0.4735] |
| Context | DCNv2-context | 0.4832 ± 0.0625 | [0.4399, 0.5324] |
| Context | RV-PURe | 0.4170 ± 0.0327 | [0.3906, 0.4440] |
| Context | CV-MLP | 0.4163 ± 0.0294 | [0.3946, 0.4382] |
| Context | **CV-PURe** | **0.3912 ± 0.0185** | **[0.3747, 0.4025]** |

按 training seed 配对且先等权平均数据集后，Scientific 中 CV-PURe 相对 MLP、ResMLP、RV-PURe 的带噪性能 AUC 改善分别为 0.0071（95% CI [0.0005, 0.0119]）、0.0063（[0.0007, 0.0107]）和 0.0051（[0.0006, 0.0093]），三个区间均为正。Context 中相对 MLP-context 的改善为 0.0464（[0.0223, 0.0705]），相对 CV-MLP 为 0.0250（[0.0030, 0.0471]）；相对 RV-PURe 的平均改善为 0.0258，但区间 [-0.0046, 0.0562] 跨零。

结论必须区分综合带噪性能和纯衰减：CV-PURe 的两组带噪性能 AUC 均最低，且在 \(\sigma=0.05\) 时于 8 个数据集中的 6 个保持最低绝对 RMSE；但 Scientific 的纯衰减 AUC 最低模型是 ResMLP。因此论文主张是 CV-PURe 在测试噪声区间内具有最佳综合预测性能，而不是普遍具有最小噪声衰减。Friedman 和 Yacht 的 clean error 很低，相对百分比增幅会被小分母放大；Bike 与 Parkinsons 的 ordinary-only 噪声结果也不能外推为对 context corruption 鲁棒。

## 6. 样本效率

在 Friedman、Concrete 和 Parkinsons 上使用训练比例 10%/25%/50%/100%，training seeds 0–4。三个小比例均采用固定 `subset_seed=0` 的嵌套子集；Parkinsons 在每位受试者内部抽取，10% 时仍覆盖全部 42 位受试者。验证集和测试集固定不变，每个比例的标准化器只在该训练子集上拟合。新增 210/210 个训练均完成且 checkpoint 可加载；100% 直接复用 70 个主实验运行。

| Dataset | 模型 | 10% RMSE | 25% RMSE | 50% RMSE | 100% RMSE | RMSE AULC |
|---|---|---:|---:|---:|---:|---:|
| Friedman | MLP | 0.8350 | 0.4966 | 0.3048 | 0.1956 | 0.3252 |
| Friedman | RV-PURe | 0.7839 | 0.4680 | 0.2756 | 0.1625 | 0.2964 |
| Friedman | **CV-PURe** | **0.5250** | **0.2404** | **0.1255** | **0.0829** | **0.1553** |
| Concrete | MLP | 9.6000 | 7.1475 | 6.5289 | 5.7824 | 6.0434 |
| Concrete | RV-PURe | 8.5531 | 6.9582 | **6.3205** | **5.6500** | **5.8158** |
| Concrete | CV-PURe | **8.3095** | **6.6784** | 6.5521 | 5.6590 | 5.8307 |
| Parkinsons | MLP-context | 5.1644 | 5.0418 | 4.8752 | 4.5896 | 4.3713 |
| Parkinsons | RV-PURe | 5.0180 | 4.9608 | 4.7337 | 4.5058 | 4.2701 |
| Parkinsons | **CV-PURe** | **4.8233** | **4.4861** | **4.3996** | **4.1932** | **3.9571** |

按 training seed 配对的 RMSE AULC 改善均以“基线减候选”定义：Friedman 上 RV-PURe 为 0.0288，95% CI [0.0205, 0.0391]，CV-PURe 为 0.1699 [0.1658, 0.1751]；Concrete 上 RV-PURe 为 0.2276 [0.1868, 0.2684]，CV-PURe 为 0.2127 [0.0794, 0.3359]；Parkinsons 上 CV-PURe 相对 MLP-context 为 0.4142 [0.1802, 0.5981]，RV-PURe 为 0.1012 [-0.0440, 0.2380]。因此样本效率结果强力支持 CV-PURe 在 Friedman 与 Parkinsons 的优势，也支持两种 PURe 在 Concrete 整体学习曲线上的改善；RV-PURe 在 Parkinsons 的整体改善仍不确定。

## 7. 计算效率

下表为一次完整 test split 的端到端推理时间（ms）：Friedman、Yacht 和 Auto MPG 的测试样本数分别为 1,501、47 和 60。每个 checkpoint 至少预热 0.5 秒，再执行 100 次带 CUDA 同步的测量并取中位数，最后报告五个 training seeds 的 mean ± SD。所有测量均固定在单张 NVIDIA Tesla V100-SXM2 32GB GPU 上，batch size 为 128。

| Dataset | MLP | ResMLP | RV-PURe | CV-PURe |
|---|---:|---:|---:|---:|
| Friedman | 4.016 ± 0.005 | 4.269 ± 0.015 | 4.860 ± 0.004 | 6.291 ± 0.008 |
| Yacht | 0.398 ± 0.001 | 0.410 ± 0.002 | 0.465 ± 0.005 | 0.602 ± 0.001 |
| Auto MPG | 0.399 ± 0.001 | 0.410 ± 0.001 | 0.463 ± 0.002 | 0.601 ± 0.001 |

Scientific benchmark 不是严格等参数比较；CV-PURe 的实际容量约为其他三个模型的一半，因此其预测优势并非来自更大的参数容量。

用于区分普通复值表示与复值 Product Unit 的三组 CV 直接对照如下。它们均使用 hidden dimension 64、100 epochs、patience 15、batch size 256 和相同的 learnable real projection。

| Dataset | CV-MLP 自由度 | CV-PURe 自由度 | 差值 | 相对差异 |
|---|---:|---:|---:|---:|
| Synthetic higher-order | 53,637 | 53,381 | 256 | 0.48% |
| Bike Sharing | 53,509 | 53,253 | 256 | 0.48% |
| Parkinsons Telemonitoring | 57,989 | 57,733 | 256 | 0.44% |

因此，计算效率结论限定为：RV-PURe 与 MLP/ResMLP 的推理耗时处于同一数量级；CV-PURe 的复值运算带来一定推理开销。三组 CV 直接对照的容量差异小于 0.5%，可用于分析 Product Unit 相对普通复值网络的增量。

## 8. 建议论文主张

可以支持：

1. PURe 的优势具有任务依赖性，而不是所有 tabular regression 上普遍领先；
2. CV-PURe 在明确的高阶乘法结构上同时提供显著预测优势，并准确恢复高阶项诱导的 pairwise interaction projections；
3. Higher-order 的 CV-PURe 明显优于同规格 CV-MLP，说明高阶优势不是普通复值参数化本身造成；
4. RV-PURe 和 CV-PURe 均改善 Bike 相对 MLP-context 的情境依赖预测，CV-PURe 均值最佳并恢复合理的 ordinary–context 结构；其相对 CV-MLP 的预测优势区间跨零，Bike 不单独承担机制归因；
5. CV-PURe 在 Parkinsons 纵向预测中优于 MLP-context、人工交叉、DCNv2 和 CV-MLP；相对 CV-MLP 的 RMSE 优势明确，而 MAE 只小幅改善；
6. CV-PURe 在 Friedman 与 Parkinsons 的整个固定子集学习曲线上保持优势；Concrete 上 RV/CV-PURe 的 AULC 也优于 MLP。

不能支持：

- PURe 在所有科学回归任务上都优于强基线；
- 所有 context 数据都能从 Product Units 获益；
- finite-difference interaction strength 是因果效应或直接参数耦合。
- 当前二阶 finite difference 能直接判定一个交互项属于三阶还是四阶。
- PURe 在所有任务上都具有最小的相对噪声退化。

### 8.1 实验逻辑闭环判断

在限定主张下，五个研究问题已经形成闭环：RQ1 由六数据集 benchmark 回答跨任务表现；RQ2 由 ground truth、五次 attribution resampling 以及 CV-MLP/CV-PURe 对照回答“是否恢复正确交互以及普通复值表示能否解释优势”；RQ3 由 Bike/Parkinsons 的 ordinary、implicit、explicit-cross、DCNv2、RV/CV 模型比较回答真实 context 中的表现；RQ4 明确把 synthetic correctness 与真实数据 association 分开；RQ5 由带噪性能 AUC 和固定嵌套子集学习曲线回答限定范围内的可靠性。后续不再需要新增实验，剩余工作是把模型规格、参数自由度和结论边界准确写入正文。

## 9. 结果文件

完整自动汇总位于 `artifacts/reports/jocs_v1/`：

- `summary.md`：自动主表；
- `table1_benchmark.csv`；
- `table2_context_main.csv`；
- `table2_context_ablations.csv`；
- `table2_context_ablation_paired_effects.csv`；
- `table3_synthetic.csv`；
- `table3_synthetic_interaction_robustness.csv`；
- `inference_efficiency/inference_timings_{raw,summary}.csv` 与 `protocol.json`；
- `paired_effects_bootstrap.csv`；
- `cv_architecture_effects.csv`；
- `all_runs.csv`；
- `figure1_synthetic_recovery.{png,pdf,tiff}`；
- `figure2_context_interactions.{png,pdf,tiff}`；
- `robustness/noise_robustness_{raw,training_seed,summary}.csv`；
- `robustness/noise_robustness_auc.csv`、`noise_robustness_overall_auc.csv`、`noise_robustness_auc_effects.csv` 与 `figure3_noise_robustness.{png,pdf,tiff}`；
- `sample_efficiency/sample_efficiency_{raw,summary,paired_effects}.csv`；
- `sample_efficiency/sample_efficiency_{aulc,aulc_effects}.csv` 与 `figure4_sample_efficiency.{png,pdf,tiff}`。
