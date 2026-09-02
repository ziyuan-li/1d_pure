# PURe JoCS 期刊扩展执行计划

## 1. 计划定位

本计划面向 *Journal of Computational Science*（JoCS）的期刊扩展，目标不是堆叠大量数据集或重新设计一套复杂网络，而是在可控工作量内回答一个比会议版更完整的问题：

> **显式乘法交互何时有用，PURe 是否能够恢复正确的交互，并在特征作用依赖情境时提供更好的预测和可解释证据？**

期刊版采用三条相互支撑的证据链：

1. **广度**：在更多科学与工程回归任务上验证稳定性；
2. **正确性**：在具有真实交互标签的合成任务上验证 interaction recovery；
3. **可解释情境案例**：Bike Sharing 与 Parkinsons Telemonitoring 作为两个正式 context-aware 案例，分别覆盖群体出行需求和个体纵向健康监测。

这三部分足以形成明确的期刊增量。默认不开发 PURe-v2，不把工作范围扩展到图像、医学影像、MovieLens 或大规模鲁棒性网格。

---

## 2. 与 JoCS 的契合点

JoCS 强调用于科学和工程问题的计算建模、算法、数据驱动方法及计算分析。本项目的投稿叙事应始终围绕以下内容，而不是把文章写成普通的 tabular benchmark：

- PURe 是一种针对非线性、高阶交互的计算模型；
- 科学与工程回归任务是主要验证场景；
- interaction recovery 用来解释模型为何有效；
- Bike Sharing 提供容易理解且五 seed 结果支持的社会计算情境案例；
- Parkinsons Telemonitoring 提供语音生物标志物随个体与随访时间变化的健康监测场景，并由五 seed 结果支持 CV-PURe 的明确优势；
- 计算时间、参数量和统计稳定性作为计算方法证据报告。

官方期刊定位参考：[Journal of Computational Science](https://shop.elsevier.com/journals/journal-of-computational-science/1877-7503)。

---

## 3. 核心研究问题

### RQ1：跨任务有效性

在不同交互结构与科学回归任务中，RV-PURe 和 CV-PURe 何时能够相对普通 MLP 和 ResMLP 获得稳定改善，其收益是否具有任务依赖性？

### RQ2：交互恢复正确性

PURe 是否能够把较高的交互强度分配给真实二阶交互及高阶乘法项诱导的 pairwise projections，而不仅仅产生更集中的解释图？

### RQ3：情境依赖预测

当输入—输出关系随已知情境变化时，学习型乘法建模相对普通 context-aware MLP、人工二阶交叉和 cross network 表现如何？深入考察 Bike Sharing 中天气/日历/时间情境，以及 Parkinsons 中语音测量随受试者与随访时间变化的关系，并比较 RV、普通复值网络与复值 Product Unit 的差异。

### RQ4：可解释性

模型恢复出的 feature-context 交互是否具有领域合理性，例如 Bike Sharing 中 hour–working-day、temperature–hour，以及 Parkinsons 中 voice-measure–subject、voice-measure–test-time 等交互？

### RQ5：噪声与低数据条件

在连续 ordinary features 受到小幅 Gaussian 扰动时，模型的绝对预测性能是否保持竞争力；在训练数据减少时，PURe 的乘法表示是否带来更低的学习曲线误差？

---

## 4. 期刊版贡献声明

建议把新增贡献压缩为四项：

1. 在六个科学与工程回归数据集上进行统一、可复现的 PURe 扩展评估；
2. 使用带真实交互标签的合成任务，定量评估 Product Unit 的 interaction recovery；
3. 加入 CV-MLP 以区分普通复值表示与复值 Product Unit 的作用，并提出一套情境交互评估协议，在 Bike Sharing 与 Parkinsons Telemonitoring 两个互补案例中展示预测表现和可解释 feature-context 结构；
4. 使用紧凑的 Gaussian feature-noise 与三数据集 low-data 协议，检验绝对性能稳定性和样本效率，而不扩展为高成本的全组合压力测试。

不把“全新网络结构”作为贡献。复值模型统一使用一个初始化为实部映射的 learnable real projection，以支持标准化后的有符号目标；只有基础 PURe 无法支持情境实验时，才考虑加入受约束的 context-cross PU 模块。

---

## 5. 必须完成的实验

### 5.1 Scientific regression benchmark

控制在六个数据集，不再扩展到八个或更多。

| 数据集 | 领域 | 用途 |
|---|---|---|
| Friedman 1 | 可控非线性回归 | 保留会议版任务 |
| Concrete | 材料/土木工程 | 保留会议版任务 |
| California Housing | 真实世界回归 | 保留会议版任务 |
| Energy Efficiency | 建筑能源 | 新增 |
| Yacht Hydrodynamics | 船舶水动力学 | 新增 |
| Auto MPG | 汽车能效/工程回归 | 新增 |

六数据集作为固定的案例研究集合，覆盖可控非线性、材料、住房、建筑能源、船舶水动力与汽车能效。其结果用于比较本文四种神经网络，不用于声称对所有 tabular 方法或所有随机数据划分普遍占优。

主表模型限制为：

1. MLP；
2. ResMLP；
3. RV-PURe；
4. CV-PURe。

额外的大规模基线调参不列为必做，因为这会明显增加工作量，但不会直接增强本文的核心论点。

统一要求：

- 使用固定、保存的数据划分；
- 神经网络采用相同的调参预算和 early stopping 规则；
- 数据划分固定为 `split_seed=0`，每个主结果运行 5 个 training seeds；
- 报告 RMSE、MAE 和 \(R^2\)，scientific benchmark 统一以 RMSE 为主指标；
- 报告 mean ± std；
- 保存预测、训练日志、配置和最佳 checkpoint。

### 5.2 Ground-truth interaction recovery

仅保留两个合成任务，避免完整参数网格。

#### Synthetic A：二阶交互

\[
y=x_1x_2+x_3x_4+\epsilon.
\]

#### Synthetic B：高阶交互

\[
y=x_1x_2x_3+x_4x_5x_6x_7+\epsilon.
\]

固定设置：

- 加入 20 个 irrelevant features；
- 使用一个中等噪声水平，例如目标标准差的 5%；
- 运行 3 个随机种子；
- 比较 MLP、ResMLP、RV-PURe 和 CV-PURe；
- 报告 RMSE、Precision@K、Recall@K 和 true-interaction mean rank。
- 每个已保存模型使用五个 reference-sampling seeds 重复 finite-difference attribution，每次 256 个 evaluation/reference pairs；这些 resamples 只评估解释稳定性，不计为额外训练运行。

不做 interaction order、噪声、无关特征数量的全组合扫描。若主结果不稳定，只增加一个补充噪声水平进行诊断。

为区分普通复值网络与复值 Product Unit 的作用，在 higher-order、Bike 和 Parkinsons 中加入 hidden dimension 64 的 CV-MLP，共 13 个运行：higher-order 3 个、Bike 5 个、Parkinsons 5 个。全部运行完成主 interaction attribution，higher-order 的 CV-MLP 另外完成五个 reference-sampling seeds 的恢复稳定性检查。CV-MLP/CV-PURe 的真实自由度分别为 53,637/53,381（higher-order）、53,509/53,253（Bike）和 57,989/57,733（Parkinsons）。三组 CV-MLP/CV-PURe 直接对照均由正式汇总器生成。

### 5.3 Parkinsons Telemonitoring 正式情境案例

- **数据集**：UCI 189，Parkinsons Telemonitoring；
- **规模**：5,875 条记录、42 位受试者；
- **目标**：`total_UPDRS`；排除高度相关的另一个临床目标 `motor_UPDRS`；
- **ordinary features**：16 个语音生物标志物；
- **context features**：subject、age、sex、test time；
- **核心问题**：语音测量与总体症状严重度之间的关系是否随受试者身份和随访时间变化。

划分在每位受试者内部按 `test_time` 排序后执行 60%/20%/20%，确保全部 42 位受试者均出现在 train、validation 和 test；标准化只使用训练集统计量。正式比较包含 MLP-ordinary、MLP-context、MLP-explicit-cross、DCNv2-context、RV-PURe-context、CV-MLP-context 和 CV-PURe-context，并固定 training seeds 0–4。

完整 35/35 个预测运行已经完成。CV-PURe 的 MAE、RMSE、\(R^2\) 分别为 3.2082 ± 0.0557、4.1932 ± 0.0803、0.8549 ± 0.0055，三个指标均为最佳。相对 MLP-context 的配对 MAE 改善为 0.2478（7.17%），95% bootstrap CI [0.1479, 0.3451]；RMSE 改善为 0.3964，95% CI [0.2299, 0.5461]。相对 CV-MLP 的 RMSE 改善为 0.1900，95% CI [0.0744, 0.2877]。35/35 份有限差分交互也已完成，`subject–test_time` 在全部 30 个 context-aware runs 中均排名第 1；为控制工作量，不新增调参网格或细粒度消融。

数据来源：[UCI Parkinsons Telemonitoring](https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring)。

### 5.4 Bike Sharing 辅助情境实验

Bike Sharing 作为低成本、易解释的辅助案例，已完成七模型 × 五 seeds 和三组 context 消融 × 三 seeds；所有模型均按预先统一的划分与训练协议汇总。

#### 预测目标

预测总租赁量 `cnt`。

必须移除：

- `casual`；
- `registered`；
- `instant`。

其中 `casual + registered = cnt`，保留它们会造成直接目标泄漏。`dteday` 只用于时间排序和划分，不直接作为数值特征输入。

#### 特征分组

Ordinary features：

- temperature；
- apparent temperature；
- humidity；
- windspeed。

Context features：

- hour；
- season；
- weather condition；
- working day；
- holiday；
- weekday；
- month；
- year。

#### 编码与划分

- 按时间顺序划分 train/validation/test，建议 60%/20%/20%；
- 不使用随机数据划分；
- 连续变量只使用训练集统计量标准化；
- hour 和 month 使用周期编码；
- season、weather condition 和 weekday 使用 one-hot；
- working day、holiday 和 year 使用二元编码；
- 所有预处理器只在训练集拟合。

#### 必要模型

| 模型 | Ordinary | Context | 交互机制 |
|---|---:|---:|---|
| MLP-ordinary | 是 | 否 | 无 |
| MLP-context | 是 | 是 | 隐式 |
| MLP-explicit-cross | 是 | 是 | 人工 ordinary-context 二阶交叉 |
| DCNv2-context | 是 | 是 | 显式 cross network |
| RV-PURe-context | 是 | 是 | 学习型乘法交互 |
| CV-MLP-context | 是 | 是 | 普通复值隐式交互 |
| CV-PURe-context | 是 | 是 | 复值学习型乘法交互 |

Bike Sharing 与 Parkinsons Telemonitoring 的正式主实验均包含 hidden 64 的 CV-MLP 与 CV-PURe，并统一使用 100 epochs、patience 15、batch size 256 和 learnable real projection。Bike 上两者真实自由度相差不足 0.5%，因此可直接检验 Product Unit 在普通复值表示之外的增量。

每个模型运行 5 个随机种子，报告 MAE、RMSE 和 \(R^2\)，其中 MAE 为主指标。

### 5.5 情境消融

Bike Sharing 不逐个删除所有 context feature，只保留三组已完成消融：

1. 去掉时间情境：hour、month；
2. 去掉日历情境：working day、holiday、weekday；
3. 去掉天气情境：season、weather condition。

Bike Sharing 的 RV-PURe-context 每组运行 3 个种子。Parkinsons 不再增加逐组 context 消融：MLP-ordinary 与 MLP-context 已覆盖“无 context/完整 context”的核心对照，七模型主比较覆盖人工交叉、DCNv2、RV-PURe、CV-MLP 和 CV-PURe 的机制差异。这样可以避免在选定数据集后继续扩大工作量。

### 5.6 紧凑鲁棒性与样本效率

Gaussian feature-noise 已按以下冻结协议完成：

- 复用 scientific benchmark、Bike 和 Parkinsons 的 190 个主模型，不重新训练；
- 在标准化连续 ordinary features 上施加 \(\sigma\in\{0,0.02,0.05,0.10\}\) 的 Gaussian noise，context 和 one-hot 列不改变；
- 每个非零噪声水平使用 corruption seeds 0–4；先在每个 training seed 内平均，再跨五个 training seeds 报告 mean、SD 和 bootstrap CI；
- 共完成 3,040 次推理，\(\sigma=0\) 与主实验结果逐项对齐；正式主指标为等权数据集、target-SD-normalized 带噪性能 AUC，纯衰减 AUC 仅作辅助。

样本效率已按以下冻结协议完成：

- 数据集限制为 Friedman、Concrete 和 Parkinsons；训练比例为 10%/25%/50%/100%；
- 小比例使用固定 `subset_seed=0` 的嵌套训练子集，validation/test 不变，标准化器仅在当次子集拟合；
- Parkinsons 在每位受试者内部构建嵌套子集，所有比例均覆盖 42 位受试者；
- 三个小比例新增 210/210 个训练，100% 复用 70 个主实验运行；
- 报告 MAE、RMSE、\(R^2\)、按 training seed 配对的 bootstrap CI 和 RMSE AULC。

该设计比会议版单一 \(\sigma=0.05\) 和单一 Concrete 学习曲线更完整，同时把新增训练控制在 210 个，不做噪声、缺失、划分和数据比例的笛卡尔积。

---

## 6. 可解释交互的统一定义

当前 PURe 并不直接产生一个严格的 feature-feature 参数矩阵，因此论文中不应把热力图简单称为“PU 权重耦合矩阵”。统一使用：

> **model-based finite-difference interaction strength**

对样本 \(x\) 和从训练集抽取的参考样本 \(x'\)，分别替换第 \(i\)、第 \(j\) 个特征，定义：

\[
\Delta_{ij}(x,x')=
f(x)-f(x^{(i)})-f(x^{(j)})+f(x^{(ij)}).
\]

总体交互强度为：

\[
I_{ij}=\frac{1}{N}\sum_{n=1}^{N}\left|\Delta_{ij}(x_n,x'_n)\right|.
\]

该指标具有以下优点：

- 可用于 MLP、DCNv2 和 PURe，比较公平；
- 不需要修改 PURe 架构；
- 可用于 synthetic interaction recovery；
- 可生成 Bike Sharing 或 Parkinsons Telemonitoring 的 ordinary-context 和 context-context 热力图。

对于 one-hot 或周期编码变量，替换和汇总必须按原始变量组进行，不能单独扰动一个 one-hot 列并产生非法类别组合。

论文中应使用“estimated interaction strength”或“interaction attribution”，避免声称它是直接从 Product Unit 参数中读取的因果耦合。

---

## 7. 计算效率与统计报告

为了符合计算方法论文的基本要求，保留一个紧凑的效率表：

- trainable parameters；
- 单次训练 wall-clock time；
- 单样本或固定 batch inference time；
- 最佳 epoch；
- 是否出现 NaN/Inf 或训练失败。

只在 Friedman 1、Yacht 和 Auto MPG 三个代表任务上报告。使用相同 GPU、batch size 和计时方式，不要求额外的 FLOPs、能耗或完整显存剖析。

统计分析采用：

- 主表 mean ± std，覆盖固定 split 0 下的 5 个 training seeds；
- PURe 与主要神经网络基线按 training seed 配对比较；
- 报告效应大小和 95% bootstrap confidence interval；
- 不同时堆叠多种显著性检验。
- 鲁棒性先平均同一模型/训练种子下的 corruption seeds，再跨 training seeds 统计，避免伪重复；
- 样本效率同时报告各比例的配对效果和整条学习曲线的 AULC 配对效果。

---

## 8. 最小论文图表集合

### Tables

1. **Table 1**：6 datasets × 4 neural models 的主结果；
2. **Table 2**：Bike Sharing 与 Parkinsons Telemonitoring 的完整主比较，以及 Bike 的紧凑 context ablation；
3. **Table 3**：Synthetic recovery、attribution robustness 与 CV-MLP/CV-PURe 直接对照；
4. **Table 4**：Gaussian noise 的分层结果与带噪性能 AUC；
5. **Table 5**：三数据集 sample-efficiency、AULC 与计算效率。

### Figures

1. **Figure 1**：Synthetic true interactions 与模型恢复结果；
2. **Figure 2**：Bike Sharing 与 Parkinsons Telemonitoring 的 ordinary-context 交互热力图；
3. **Figure 3**：8 个主数据集的 normalized-RMSE Gaussian noise curves；
4. **Figure 4**：Friedman、Concrete 与 Parkinsons 的 sample-efficiency curves。

Figure 3–4 保留在正文；不设置附录或补充材料。深度曲线、缺失值曲线和更大的 robustness/low-data 网格不列为必做。

---

## 9. 分阶段实施与停止条件

### Phase 0：协议冻结

- 固定数据划分、预处理和主指标；
- 建立统一配置、日志和 checkpoint 命名；
- 先完成一组 smoke test，确认所有模型能在 GPU 上训练和恢复。

### Phase 1：低成本可行性验证

- Bike Sharing：MLP-context、MLP-explicit-cross、DCNv2-context、RV-PURe-context，各 1 seed（已完成）；
- Synthetic A：MLP 与 RV-PURe，各 1 seed（已完成）；
- Synthetic B：MLP 与 RV-PURe，各 1 seed（已完成；预测相近，RV-PURe 的 true-pair mean rank 更好）；
- Parkinsons Telemonitoring：六个基础模型 × 五 seeds 已完成，随后加入 CV-MLP × 五 seeds；
- 检查性能、数值稳定性、跨批次误差和交互热力图。

### Phase 2：必须实验

仅当 Phase 1 没有实现错误或明显数值问题时，运行：

- 6-dataset benchmark；
- 2 个 synthetic recovery 实验；
- Bike Sharing 与 Parkinsons Telemonitoring 均完成 7 模型主比较；
- Bike Sharing 完成三组精简 context ablation；Parkinsons 使用 ordinary/context 主对照控制工作量；
- 紧凑效率统计。

### Phase 3：按结果补充

针对会议版已有但新主线缺失的两项证据，已完成：

- 四噪声水平、五 corruption seeds 的 inference-only Gaussian robustness；
- 10%/25%/50%/100% 的三数据集 sample efficiency。
- higher-order、Bike 与 Parkinsons 上 13 个 CV-MLP 对照。

除上述已完成内容外，只有出现明确审稿风险时才增加：

- 一个备用数据集；
- 1/2/4/8 blocks 的小型深度实验；
- context corruption 或 missingness 的单独诊断。

### 情境案例确认

Bike Sharing 的完整五-seed结果中，RV-PURe 相对 MLP-context 的配对 MAE 改善区间为正；CV-PURe 的均值最佳，但相对同规格 CV-MLP 的预测区间跨零，因此 Bike 支撑真实情境预测与交互解释，不单独承担 Product Unit 的机制归因。Parkinsons 中 CV-PURe 在三个平均指标上均最佳，相对 MLP-context 的 MAE/RMSE 配对区间均为正，且相对同规格 CV-MLP 的 RMSE 改善区间为正。两项案例的角色互补，结论边界清楚。

Context-aware 主结果仅保留 Bike 与 Parkinsons。不继续开发 MovieLens、Hillstrom 或复杂 context-PU 新结构。

---

## 10. 明确排除的工作

为了控制工作量，当前投稿版本不做：

- MovieLens；
- Hillstrom、Criteo、MIND、Avazu 等会改变任务范式或显著增加工程量的数据集；
- 图像分类或医学影像；
- 超过六个主 benchmark 数据集；
- 额外模型族的完整调参；
- 噪声 × 缺失 × covariate shift × low-data 的全组合实验；
- 16-block 深度实验；
- 新的 PURe-v2 主网络；
- 对每一个 context variable 分别进行消融；
- 多套解释方法同时比较。

这些项目可以在审稿人明确要求时作为补充，而不是首轮投稿前的默认工作。

---

## 11. 推荐论文结构

1. Introduction；
2. Related Work；
3. PURe Method Recap；
4. Interaction Recovery and Context-Aware Evaluation；
5. Experimental Protocol；
6. Scientific Regression Results；
7. Synthetic Interaction Recovery；
8. Context-Dependent Case Studies（Bike Sharing 与 Parkinsons Telemonitoring）；
9. Robustness, Sample Efficiency, Computational Efficiency and Limitations；
10. Conclusion。

主叙事保持为：

> **PURe 的优势不仅体现在更多数据集上的预测结果，还体现在恢复真实非线性交互，以及处理随时间、日历、个体和随访过程变化的特征关系。**

---

## 12. 完成标准

首轮投稿前只要求满足以下条件：

- 六个 benchmark 数据集结果完整；
- 四个主模型使用统一数据划分和调参规则；
- 两个 synthetic recovery 任务完成；
- synthetic interaction attribution 在五个 reference-sampling seeds 下完成稳健性检查；
- Bike Sharing 与 Parkinsons Telemonitoring 的七模型主比较均完整运行；Bike 的三组消融完整；
- Bike Sharing 无目标泄漏并使用时间划分；Parkinsons 排除 `motor_UPDRS`，并在每位受试者内部使用时间划分；
- Bike 的 44 个、Parkinsons 的 35 个已保存结果均完成相应有限差分交互分析；
- 五张主表和五张主图完整；
- Gaussian noise 的 3,040 次分层推理以及三数据集 210 个新增 low-data 训练完整，checkpoint 审计通过；
- robustness 与 sample-efficiency 的统计表和 Figure 3–4 完整并进入正文；
- 固定 split 0 的 5-training-seed 主结果、置信区间和效率数据可复现；
- higher-order、Bike 与 Parkinsons 的 13 个 CV-MLP 运行及交互分析完整，checkpoint 已加载审计；
- 配置、日志、预测和 checkpoint 均保存；
- 论文中的“interaction strength”定义与实现一致；
- 不把后验 interaction attribution 表述为因果关系或直接参数耦合。

满足以上内容即可进入写作和投稿阶段，不再主动扩大实验范围。
