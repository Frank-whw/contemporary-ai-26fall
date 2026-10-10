# 实验一：10 类新闻文本分类

## 环境

- Python 3.12（见仓库根目录 `.python-version`），用 [uv](https://docs.astral.sh/uv/) 管理依赖，版本锁定在根目录 `uv.lock`。
- 本次运行版本：scikit-learn 1.9.1、numpy 2.5.3、scipy 1.18.1、pandas 3.0.5、matplotlib 3.11.2。
- 运行机器：macOS 26.5（arm64，Apple M5，10 核），只用 CPU，完整跑一次约 16–20 分钟（线性核 SVC 和 9 次 MLP 训练占大头）。

## 从干净状态复现

在仓库根目录执行：

```bash
uv sync && cd data-Project1 && rm -rf outputs predictions.csv && uv run python main.py
```

脚本用相对路径读取 `train_data.csv`、`test_data_unlabeled.csv`，所以必须在 `data-Project1/` 目录下运行。

## 随机种子

- `SEED = 42`：用于训练/验证分层划分（`train_test_split(stratify=target, test_size=0.2, random_state=42)`）以及 MLP 的初始化和每轮打乱。
- `REPEAT_SEEDS = (1, 2)`：只有当两个配置在 seed 42 上的验证 macro-F1 差距小于 `SMALL_GAP = 0.01` 时才追加。每个种子会重新划分数据、重新在训练折上拟合向量器，MLP 也会换初始化种子。
- MLP 的 `random_state` 传入的是 `np.random.RandomState(seed)` 实例而不是整数。原因是 sklearn 每次调用 `partial_fit` 都会重新执行 `check_random_state`：传整数时，从第 2 轮起每轮的打乱顺序都一样。这一点的对照结果见 `outputs/checkpoint_check.csv`。
- MultinomialNB、LogisticRegression（lbfgs）和线性核 SVC 在给定数据上是确定性的。仍然可能存在的非确定性来源是多线程 BLAS 的浮点求和顺序，换机器或换 BLAS 后末位可能不同。在本机上多次从干净状态运行，`results.csv`（耗时列 `fit_s` 除外）、`mlp_curves.csv` 和 `predictions.csv` 逐位一致。

## 流程概要

1. 先分层切出 20% 验证集，再做任何统计。`CountVectorizer(min_df=2)` 的词表和 `TfidfTransformer` 的 IDF 只在训练折上拟合，然后用来变换验证折和测试集。
2. Q1：词表固定，分类器固定为 LR(C=1) 和 MNB(alpha=1)，比较 原始词频 / 仅 L2 归一化 / TF-IDF 三种表示。
3. Q2：在同一套 TF-IDF 上扫描 MNB 的 alpha（0.01、0.1、1），LR 和线性核 SVC 的 C（0.1、1、10）。另外用更严格的收敛容差各重训一次，检查优化是否充分。
4. Q3：MLP 隐藏层宽度取 100、200、300，其余超参不变。用 `partial_fit` 逐轮训练 50 轮，每轮记录训练损失和验证指标，取验证 macro-F1 最高的那一轮权重。
5. 最终选择：各模型族的最优配置比较 3 个种子的验证 macro-F1 均值，按一倍标准误规则在相近者中取参数最少的。随后用完整的 `train_data.csv` 重新拟合向量器和模型（MLP 训练轮数取各种子最佳轮数的中位数），预测测试集。
6. 主实验结束后做一次单因素消融：去掉邮件头（第一个空行之前的内容）。消融结果不影响最终模型。

## 输出文件

| 文件                                                          | 内容                                                                                                                                                                                    |
| ------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `predictions.csv`                                             | 测试集预测：2457 行，无表头、无索引，每行一个 0–9 的整数，行序与 `test_data_unlabeled.csv` 一致                                                                                         |
| `outputs/run.log`                                             | 完整运行日志                                                                                                                                                                            |
| `outputs/results.csv`                                         | 每次拟合一行：模型、超参、种子、维度、训练/验证 macro-F1 与 accuracy、参数量、耗时、迭代数、是否收敛                                                                                    |
| `outputs/comparisons.csv`                                     | 每组对照的汇总（seed 42 得分、多种子均值/标准差、是否被选中）                                                                                                                           |
| `outputs/mlp_curves.csv`                                      | MLP 逐轮原始记录：`train_loss`（即该次训练的 `loss_curve_`，含 L2 项）、`train_logloss`、训练与验证 macro-F1、`val_logloss`、`val_acc`；`run=final_full_train` 的行是最终全量重训的损失 |
| `outputs/mlp_summary.csv`                                     | 每个宽度、每个种子的最佳轮与最后一轮对比                                                                                                                                                |
| `outputs/loss_curve.png`                                      | 由 `mlp_curves.csv` 画出的损失与验证曲线（未平滑）                                                                                                                                      |
| `outputs/checkpoint_check.csv`                                | 逐轮训练方式的对照：fit / partial_fit / warm_start                                                                                                                                      |
| `outputs/per_class.csv`、`outputs/confusion_final_seed42.csv` | seed 42 验证集上的每类 F1、召回率，以及最终模型的混淆矩阵                                                                                                                               |
| `outputs/split_counts.csv`                                    | 训练折与验证折的类别计数                                                                                                                                                                |
| `outputs/summary.json`                                        | 配置、版本、数据统计、每组对照的决策与规则、最终模型信息                                                                                                                                |
