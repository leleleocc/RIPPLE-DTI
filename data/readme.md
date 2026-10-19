# 数据 Demo

本目录只保留第二阶段的关键样例和必要输入文件，供理解数据格式。原料、普查结果、中间数据、缓存与过程脚本已清理。

| 文件 | 内容 |
|---|---|
| [demo.json](examples/stage2/demo.json) | 两条真实药理记录、一个真实结构几何样例、一个明确标注的虚构回放格式样例 |
| [O35433.fasta](examples/stage2/assets/O35433.fasta) | 大鼠 TRPV1 的完整蛋白序列，共 838 个残基 |
| [3J5P-assembly1.cif](examples/stage2/assets/3J5P-assembly1.cif) | demo 使用的四聚体参考结构，四条链共 2,368 个有 C-alpha 坐标的残基 |

## 如何看 Demo

| 字段 | 含义 |
|---|---|
| `sources` | 原始来源 URL、版本及文件哈希；完整药理快照已删除 |
| `shared_protein_input` | 药理与结构样例共用的蛋白序列、参考结构及编号说明 |
| `pharmacology_samples` | 辣椒素对大鼠 TRPV1 的增强记录，以及 capsazepine 的抑制记录 |
| `geometry_sample` | 指定参考残基对，学习当前结构中的距离 |
| `replay_format_example` | 展示如何回放基础数值任务；10 nM 是演示值，禁止用于训练 |
| `input` / `training_target` | 分别是模型输入和训练答案；答案不能混入输入 |
| `loss_enabled` | 当前样本参与哪些训练任务 |

JSON 内的 `assets/` 路径相对于 `demo.json`；`#/shared_protein_input` 指向同一 JSON 中的共用蛋白输入。

两条药理记录来自 GtoPdb 2026.3.1，保留来源 ID、SMILES、作用标注及 PMID。三维结构是独立参考，不是两次药理实验的配套结构。天然序列、实验构建体和结构残基编号仍需完整映射，不能按行号直接对应。

这不是可直接训练的缓存：候选口袋、特征和正式划分尚未生成，相关字段为 `null`。`model_prediction=null` 表示没有正式模型预测结果；结构距离只表示当前几何，不表示药物诱导变化。

两阶段数据规模与详细示例见 [数据说明](../docs/DATA.md)，输入输出见 [模型说明](../docs/MODEL.md)，执行计划见 [训练策略与脚本](../docs/TRAINING.md)。
