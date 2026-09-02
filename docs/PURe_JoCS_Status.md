# PURe JoCS 扩展项目状态、待办与研发顺序

更新时间：2026-09-02

## 1. 当前冻结范围

- 论文主线：scientific regression benchmark、ground-truth interaction recovery、Bike Sharing 与 Parkinsons Telemonitoring 两个 context-aware 案例。
- Scientific benchmark：Friedman、Concrete、California、Energy、Yacht、Auto MPG；固定 `split_seed=0`，training seeds 0–4。
- Context-aware：Bike 与 Parkinsons 均使用七模型 × 五 seeds；Bike 另有三组消融 × 三 seeds。
- Synthetic：pairwise 与 higher-order 使用 training seeds 0–2；每个模型另做五个 attribution sampling seeds。
- 鲁棒性：标准化 ordinary features 上的 Gaussian noise，`sigma = 0, 0.02, 0.05, 0.10`，主指标为带噪性能 AUC。
- 样本效率：Friedman、Concrete、Parkinsons 的 10%/25%/50%/100% 嵌套训练子集。
- 不继续新增数据集、网络结构或大规模超参数网格；英文稿暂不恢复，先保持实验版本稳定。

## 2. 总体完成情况

| 模块 | 状态 | 已完成内容 |
|---|---|---|
| 项目结构与统一记录 | 已完成 | 数据、模型、训练、评估、实验与测试分层；每个正式运行保存配置、环境、split、history、prediction、metrics、checkpoint 和 manifest；不做哈希 |
| Scientific benchmark | 已完成 | 六数据集 × 四模型 × 五 seeds，共 120/120 个正式运行 |
| Ground-truth recovery | 已完成 | 27/27 个正式运行、27 份主交互结果和 135 次 attribution resamples |
| Bike Sharing | 已完成 | 35 个主比较、9 个消融和 44 份交互结果 |
| Parkinsons Telemonitoring | 已完成 | 七模型 × 五 seeds，共 35 个预测运行与 35 份交互结果 |
| CV-MLP 对照 | 已完成 | higher-order 3 个、Bike 5 个、Parkinsons 5 个，共 13/13 个运行及交互分析 |
| Gaussian feature-noise | 已完成 | 复用 190 个 checkpoint，完成 3,040 次推理与 AUC 汇总 |
| 样本效率 | 已完成 | 210 个新增训练，合并 70 个 100% 主运行形成完整学习曲线 |
| 统计、效率与图表 | 已完成 | 226-run 主汇总、配对 bootstrap CI、效率表和正式图表 |
| 自动测试 | 已完成 | 服务器完整测试 43/43 通过 |
| 期刊正文 | 已暂停 | 旧英文初稿已删除；实验完成后再迁移到 JoCS 模板 |

## 3. 正式数据集状态

| 数据集 | 划分与种子 | 正式运行 | 论文角色 |
|---|---|---:|---|
| Friedman 1 | split 0；training seeds 0–4 | 20/20 | 主 benchmark；CV-PURe 最强 |
| Concrete | split 0；training seeds 0–4 | 20/20 | 主 benchmark；RV-PURe 最佳 |
| California Housing | split 0；training seeds 0–4 | 20/20 | 主 benchmark；RV-PURe 小幅改善 |
| Energy Efficiency | split 0；training seeds 0–4 | 20/20 | 主 benchmark；CV-PURe 平均最佳，配对区间跨零 |
| Yacht Hydrodynamics | split 0；training seeds 0–4 | 20/20 | 主 benchmark；CV-PURe 明显最佳 |
| Auto MPG | split 0；training seeds 0–4 | 20/20 | 主 benchmark；CV-PURe 平均最佳 |
| Synthetic pairwise | data/split 0；training seeds 0–2 | 12/12 | 二阶 ground-truth recovery |
| Synthetic higher-order | data/split 0；training seeds 0–2 | 15/15 | 高阶投影恢复与 CV-MLP 对照 |
| Bike Sharing | 固定时间划分 | 44/44 | 第一项 context-aware 案例 |
| Parkinsons Telemonitoring | 受试者内时间划分 | 35/35 | 第二项 context-aware 案例 |

Parkinsons 数据来源：[UCI 189 — Parkinsons Telemonitoring](https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring)。

## 4. 模型与分析组件

| 模型/组件 | 状态 | 用途 |
|---|---|---|
| MLP / MLP-context | 已完成 | 全部核心实验 |
| ResMLP | 已完成 | scientific benchmark 与 synthetic |
| RV-PURe / RV-PURe-context | 已完成 | 全部核心实验 |
| CV-MLP | 已完成 | higher-order、Bike 与 Parkinsons 复值基线 |
| CV-PURe / CV-PURe-context | 已完成 | 全部核心实验 |
| MLP-explicit-cross | 已完成 | Bike 与 Parkinsons |
| DCNv2-context | 已完成 | Bike 与 Parkinsons |
| Synthetic generators | 已完成 | recovery 实验 |
| Finite-difference interactions | 已完成 | synthetic correctness 与真实数据关联分析 |
| 统计汇总、鲁棒性、样本效率和作图 | 已完成 | 正式表格、置信区间与图 |

## 5. 剩余工作与执行顺序

| 顺序 | 任务 | 状态 | 完成标准 |
|---:|---|---|---|
| 1 | 完整回归测试 | 已完成 | 服务器 43/43 项测试通过 |
| 2 | 重建正式汇总与图表 | 已完成 | 226 个主运行、106 份主交互，正式图可重复生成 |
| 3 | 最终实验一致性审计 | 已完成 | 226 个主运行及 210 个 low-data 运行的必要工件 0 缺失，全部 436 个 checkpoint 可加载 |
| 4 | 迁移至 JoCS LaTeX 模板 | 待开始 | 作者信息确认后制作投稿稿件 |

## 6. 当前结论边界

可以支持 PURe 在这组六个固定 scientific case studies 的若干任务上改善预测、CV-PURe 在 higher-order synthetic 中稳定恢复真实乘法项诱导的 pairwise projections，以及在 Bike/Parkinsons 中学习到稳定的 ordinary–context 结构。Bike 上 CV-PURe 均值最佳，但相对同规格 CV-MLP 的预测区间跨零；其机制增量主要由 higher-order 与 Parkinsons 的 CV-MLP/CV-PURe 直接对照支撑。不能声称 PURe 在所有数据集、所有划分、所有噪声定义或所有评价指标上均占优。

实验逻辑已经闭环：RQ1 由六数据集 benchmark 回答；RQ2 由 ground truth、五次 attribution resampling 和 CV-MLP 对照回答；RQ3 由 Bike/Parkinsons 的 ordinary、implicit、explicit-cross、DCNv2、RV/CV 模型比较回答；RQ4 区分 synthetic correctness 与真实数据 association；RQ5 由带噪性能 AUC 和固定嵌套子集学习曲线回答。
