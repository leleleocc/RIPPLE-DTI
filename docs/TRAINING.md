# 训练策略与脚本

更新：2026-10-17。设备：**4 张 RTX 4090，每张 24 GB**。数据量见 [DATA.md](DATA.md)，输入输出见 [MODEL.md](MODEL.md)。

**先基础预训练，再通道专项微调，最后冻结模型用于 PIEZO1/2。** 方案已确定，正式训练尚未开始；目前只能运行数据盘点、基础处理和历史小规模几何检查。

## 1. 两阶段怎么训练

| 项目 | 阶段一：基础预训练 | 阶段二：通道专项微调 |
|---|---|---|
| 目标 | 学习结合数值与候选口袋选择 | 学习增强 / 抑制方向 |
| 数据批次 | 测量 60%、位点 25%、复合物 15% | 药理 60%、当前几何 20%、基础回放 20% |
| 起始权重 | 各启用任务为 1 | 药理 1、当前几何 0.2、基础回放组 0.2 |
| 学习率 | RIPPLE 轻量模块 1e-4 | 新任务头 1e-4，共享轻量模块 2e-5 |
| 训练预算 | 主测量 train 集最多遍历 2 次 | 仅 effect 头热身 5 次，再联合微调最多 30 次药理遍历 |
| 验证与早停 | 每 1,000 优化步及结束时验证，连续 3 次无改善停止 | 每次药理遍历验证，连续 5 次无改善停止 |
| checkpoint 选择 | 各终点标准化 RMSE 的平均值最低 | 增强 / 抑制 macro-F1 最高 |

两阶段始终冻结大型 ProtBert/ChemBERTa 特征模型，离线缓存逐残基和药物特征；长序列用重叠窗口并保留位置映射。阶段二加载阶段一最优权重，热身时固定共享模块；联合微调解冻 RIPPLE 残基编码/融合、交互、来源融合、传播和当前几何读出。热身不计入 30 次联合遍历。

阶段一的相互作用二分类仅在具备可靠实验正负标签时启用。数值预测按 Kd/Ki/IC50/EC50 区分，未知终点及不支持的删失记录屏蔽。候选口袋未覆盖真实位置时单独记录，不能伪造正确候选。

阶段二每 10 个任务批次先按 6 药理、2 几何、2 回放调度。只在 train 内按类别和家族平衡药理抽样；一次药理遍历对应 train 药理样本数的抽样量。回放只取阶段一 train 清单。

每项损失按有效标签数归一化，回放各子任务归一化后取平均。表中权重是任务批次上的乘数，不能再重复乘批次比例；记录实际批次数、原始/加权损失和有效标签数。缺失任务不以全 mask 批次凑比例。

**首版关闭** `function`、`mechanism`、`allosteric_pocket`、`pose_rank`、`geometry_response`、`gating`、`state_preference`、`nonbinder_delta`。当前几何只重建参考结构距离；若装配审核未完成，先用合格小子集，必要时先启动药理并记录实际比例。

## 2. 训练前必须固定什么

按 [数据划分规则](DATA.md#4-清洗划分与交付) 一次性生成两阶段清单：完整序列 30% identity、双向至少 80% 覆盖聚类，并合并同一基因的物种同源物；按蛋白组约 80/10/10 划分。第二阶段验证/测试蛋白不得进入第一阶段训练；PIEZO 全家族及同源物单列 external，两个阶段及几何辅助均排除。

这是 cold-target 协议，另报化学骨架重叠。验证/测试保留真实分布；类别缺失时报告指标不可估计，不改为随机行划分。测试配体坐标和真实位点只作答案，不参与候选口袋生成。

## 3. 四卡执行与评测

| 项目 | 起始设置 |
|---|---|
| 特征准备 | 按蛋白/药物分片分配四卡，避免重复提取 |
| RIPPLE | 隐维度 128、稀疏残基图、按长度分桶 |
| 优化 | 4 进程 DDP、混合精度、梯度累积 |
| 一般蛋白 | 每卡微批 2，累积 8，有效批量 64 |
| 长蛋白 / 大装配 | 每卡微批 1，累积 16，有效批量 64，需实测显存 |
| 首轮 | 先 1 seed 打通，再补 3 seeds 和必要对照 |

四卡显存不能当一张 96 GB 卡使用；不能随意截取 PIEZO 前 1,024 个残基或用单链替代三聚体。各卡同一步使用相同任务，处理未使用参数，并按全局有效标签数归一化。先测试 100-200 个真实批次的显存、吞吐和 I/O，再估计耗时。

阶段一报告各终点原始 RMSE 与标准化 RMSE，标准化参数只由 train 计算；有可靠二分类标签时报告 AUPRC/AUROC。口袋报告 Top-1/Top-3 命中、覆盖率和有效样本数。

阶段二报告 macro-F1、balanced accuracy、每类 precision/recall、混淆矩阵和家族支持数；几何报告独立蛋白组上的距离 MAE。必要对照为类别先验、只训练 effect 头、去掉几何辅助、完整微调，使用相同划分与预算。最终给出三 seed 均值与波动。

## 4. 当前可运行的脚本

以下命令在 `RIPPLE-DTI/` 根目录执行，`python` 为已激活环境的解释器。基础原料在服务器，本机不具备完整训练数据。

| 入口 | 状态和用途 |
|---|---|
| [`prepare_sources.py`](../scripts/ripple/prepare_sources.py) | 可运行，需远端原料；基础中间表及残基映射核验 |
| [`census_channel_structures.py`](../scripts/ripple/census_channel_structures.py) | 检索 PDB 身份；需另行提供通道目录 CSV，未审核全量坐标 |
| [`prepare_channels.py`](../scripts/ripple/prepare_channels.py) | 历史八结构准备，不是全量装配处理器 |
| [`smoke_channel_geometry.py`](../scripts/ripple/smoke_channel_geometry.py) | 历史当前几何优化及一次完整前向/反向检查 |

基础/八结构处理依赖 `requirements-data.txt`，几何检查还需 PyTorch。结构普查依赖 Python 标准库与系统 `curl`。正式模型依赖见 `requirements.txt`，检查依赖见 `requirements-dev.txt`。

### 基础数据

```console
python -m pip install -r requirements-data.txt
python scripts/ripple/prepare_sources.py --raw data/raw --out data/processed --mapping-limit 300 --workers 8 --contact-cutoff 4.0
```

`--mapping-limit 300` 只限制逐残基核验的复合物数，不限制全来源扫描；接触阈值为 4.0 Angstrom。可追加 `--resume --offline` 复用已有阶段和映射缓存，缺失缓存不会补下载。来源或参数变化时另用输出目录。该脚本不生成正式特征、候选口袋或划分。

### 通道药理与结构普查

历史药理审计过程脚本、原料和普查报告已清理，正式通道数据准备入口尚待补齐；历史计数见 [DATA.md](DATA.md)。

结构普查脚本仍保留，但默认通道目录已删除。运行前须从服务器取得或从 [GtoPdb](https://www.guidetopharmacology.org/DATA/targets_and_families.csv) 重新下载目录，并用 `--targets` 显式指定现存 CSV。以下是参数模板，需替换路径后执行：

```console
python scripts/ripple/census_channel_structures.py --targets <现存通道目录CSV路径> --output <结构普查输出目录> --cutoff 2026-10-17
```

可加 `--refresh` 重新查询。脚本只盘点条目；截止日期约束首次发布，不保证未来检索结果字节不变。重新取得的来源应固定版本和哈希，并重新统计。

### 历史八结构检查

```console
python scripts/ripple/prepare_channels.py --config configs/channel_pilot.json --raw-dir data/raw/channels --output-dir data/processed/channels
python scripts/ripple/smoke_channel_geometry.py --data-dir data/processed/channels --output-dir outputs/channel_geometry_smoke --steps 64 --seed 83
```

准备器可加 `--offline`，前提是已有下载缓存；这条检查仅在小模型上优化当前几何，不能产生正式训练结果。

## 5. 正式训练还缺什么

| 顺序 | 待完成工作 | 当前入口 |
|---|---|---|
| 1 | 化学标准化、实验/装配审核、统一样本表与标签 mask | 盘点结果只是候选输入 |
| 2 | 聚类、同源排除、跨阶段划分及版本冻结 | [`make_splits.py`](../scripts/ripple/make_splits.py) 抛占位异常 |
| 3 | 独立口袋、逐残基特征、稀疏图及缓存 | [`build_cache.py`](../scripts/ripple/build_cache.py) 抛占位异常 |
| 4 | 通用批处理与当前几何查询 | [`dataset.py`](../src/ripple/data/dataset.py) 尚未实现；每装配至多 64 对查询 |
| 5 | 两阶段配置、四卡优化、日志、保存与恢复 | [`train.py`](../src/ripple/train.py) 无正式训练循环 |
| 6 | 独立评测与冻结推理、PIEZO 结果导出 | [`evaluate.py`](../src/ripple/evaluate.py) 为占位且无 CLI；PIEZO 入口尚不存在 |

先用真实小子集打通上述链路，再扩大到预算；因此现在不提供无法运行的正式 `torchrun` 命令。`configs/ds_zero2.json` 存在不代表已接入分布式训练。

实现时一起修正四个接口差异：effect 从三类改为两类并屏蔽未知方向；旧 `CHANNEL`/`JOINT` 策略不能代替本方案；未知终点不能与 EC50 共用嵌入且必须屏蔽回归；collate 图索引保持实际 `[B,2,E]` 样本内编号，不能错误累加跨样本偏移。详情见 [模型代码边界](MODEL.md)。

修改实现后执行对应检查：

```console
python -m pytest tests/test_source_alignment.py tests/ripple
```

历史 42 项通过并非本次重跑结果。每次正式运行归档数据哈希、实际样本量、划分、配置、环境/CUDA、seed、显存/吞吐、checkpoint 和独立指标。

## 6. PIEZO 实践与最终交付

冻结模型后，先审核人 PIEZO1 参考结构候选 `8YEZ`，再开展 PIEZO2；`6KG7` 是小鼠 PIEZO2，不能混写为人源。完整装配、物种和残基对应都需核对。

将 823 个来源药物 ID 标准化后作为约 800 个候选的起点，实际独立分子数另报。已知 PIEZO 实验对照分开保存，不参与训练或调参。输出 Top 20 候选、受监督支持的结合排序、条件性增强/抑制分数、口袋及链/残基/参考坐标、跨 seed 一致性。

若相互作用头未获得可靠训练，采用指定终点的数值排序并明确终点。未知组合不造阴性，不能据此拼出准确率；当前几何重建与残基显著性也不能当药物诱导结构变化证据。模型未显式建模膜张力或机械刺激。

最终交付：冻结数据与无泄漏报告、可运行四卡链路、两个 checkpoint、独立评测与对照、PIEZO 候选表及证据来源。
