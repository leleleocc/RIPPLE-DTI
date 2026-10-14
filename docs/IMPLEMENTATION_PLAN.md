# RIPPLE-DTI 终版本实施清单

> **RIPPLE-DTI：Pocket-aware DTI with ligand-conditioned pocket-to-function propagation**
> 配体条件的 pocket-to-function 传播作为主创新，pose/conformer 模块作为结构辅助创新。

本文是可执行的实施清单，不是论文大纲。每一项给出目标产物、依赖、完成判据。
顺序是硬约束：数据契约先于下载，缓存先于训练，泄漏审计先于任何报数字。

## 0. 终版本相对现有代码的偏差（必须先改）

现有实现（`src/models/ripple.py`、`src/models/ripple_training.py`、`docs/NETWORK_MODULES.md`）
按旧方案写，和终版本有三处实质性冲突。不改这些，后续数据管线和训练脚本都会接错。

| # | 现有行为 | 终版本要求 | 改哪里 |
|---|---|---|---|
| D1 | `PocketConformerMatcher` 是硬门控：无有效 pair 时交互信号和残基差分为零，且声称没有绕过匹配的完整预测路径 | pose confidence 是**软条件**，不是门控。DTI 主路径在没有 pose 标签、甚至没有 3D 构象时必须仍能前向。pose 模块只调制 pocket-conditioned interaction state | `ripple.py` forward；`shared_fusion.py` 的 `pair_valid` 逻辑 |
| D2 | 监督是 `native_conformer_index`（"实验监督的 pocket-conformer pair matching"） | 删除该目标。改为 `L_pose_rank`：同一 ligand × 同一 candidate pocket 下的 pose quality 分类 / pairwise ranking / near-native enrichment。标签只来自 CrossDocked2020、CASF、PDBbind 的 RMSD-to-crystal | `ripple_training.py` 的 `RIPPLELabels`、`RIPPLELossWeights`；`configs/ripple_network.yaml` |
| D3 | 没有 function 终点监督。`function_mask` 只是传播模块的可选输入，`propagation` 阶段的损失是 effect + nonbinder_delta | 增加 `L_function`：pocket-conditioned residue state → functional-site 预测（残基级 BCE）。`function_mask` 同时是输入 sink 先验**和**监督标签，两者必须分开字段，避免标签泄漏进输入 | `RIPPLEBatch` 增加 `function_label`；loss 增加 `function` |
| D4 | 课程是 structure → dti → propagation → pharmacology → joint，structure 阶段训 joint matcher | 课程改为 pocket → dti → propagation → pharmacology → joint。structure 阶段改名为 `pocket`，监督 `L_pocket`（候选 pocket 选择）+ 可选 allosteric，不再监督 conformer index | `TrainingStage`、`_STAGE_MODULES`、`stage_loss_weights` |
| D5 | README 把 "joint pocket-conformer matching" 写成核心，并说 matcher 必须先产出有效 pair | 改写为两主一辅：pocket-aware DTI、ligand-conditioned pocket-to-function propagation、pose confidence 作为 structure-aware auxiliary module | `README.md`、`docs/NETWORK_MODULES.md` |

完成判据：`python -m pytest -q` 全绿，且新增测试覆盖

- 无 conformer（`conformer_mask` 全 False）时 DTI logit 仍有限、梯度仍到达 drug/protein 编码器；
- `function_label` 缺失的样本 `label_count["function"] == 0`，且不产生 NaN；
- pose ranking loss 在无 pose 标签批次上贡献精确为 0。

## 1. 目标与非目标

### 要交付的

1. 一套按任务分层、带独立 label mask 的数据管线（下载 → 清洗 → 映射 → 划分 → 缓存）。
2. 与终版本一致的模型：pocket 选择进入 DTI 表征，pose confidence 软调制，残基图上的 pocket-to-function 传播。
3. 可复现的训练入口：单卡调试 + DeepSpeed 多卡，课程训练，masked multi-task loss。
4. 泄漏受控的评测：cold-target / cold-pocket / cold-both，外加 pocket localization 与 pose ranking 的独立指标。
5. PIEZO1/2 冷靶点外部 case study 的数据与分析脚本（不进训练集）。

### 明确不做

- 不声称恢复真实物理能量传播路径。表述固定为 *ligand-conditioned predictive propagation from a binding pocket to annotated functional regions*。
- 不做 "实验监督的 pocket × conformer pair matching"。PDBbind 没有完整的 `pocket × conformer` 实验标签。
- 不把 Kd/Ki/IC50/EC50 混成一个连续标签。
- 不把缺失亲和力、缺失功能注释、缺失机制注释当作负样本。
- 不把 PIEZO 或其同源蛋白放进训练集。
- 不把 DiffDock / EquiBind 的预测结果当训练标签，只作实现参考。

## 2. 数据契约（一切的前提）

所有下游脚本只读这一个 schema。改 schema 必须先改本节再改代码。

### 2.1 样本主键

```text
sample_id = sha1(canon_smiles + "|" + uniprot_acc + "|" + source + "|" + source_id)
```

一条样本是一个 (化合物, 蛋白) 对加来源。同一对可以有多条记录（不同 assay / endpoint），
不在清洗阶段合并，合并规则放在划分与采样阶段。

### 2.2 主表 `samples.parquet`

| 列 | 类型 | 说明 |
|---|---|---|
| `sample_id` | str | 主键 |
| `canon_smiles` | str | RDKit canonical, isomeric |
| `inchikey` | str | 立体敏感 InChIKey，划分去重用 |
| `uniprot_acc` | str | 主 UniProt accession |
| `organism` | str | 物种，人类优先标记 |
| `sequence` | str | UniProt 全长序列 |
| `source` | str | bindingdb / chembl / biosnap / pdbbind / gtopharm / asd |
| `source_id` | str | 来源内部 ID，可回溯 |
| `y_interaction` | int8 或 null | 1 正 / 0 明确负 / null 未知。**null 不是 0** |
| `affinity_value` | float32 或 null | pActivity，`-log10(M)` |
| `endpoint_type` | int8 | 0=Kd 1=Ki 2=IC50 3=EC50 4=未知 |
| `assay_id` | str | 保留，不参与主键 |
| `standard_units` | str | 统一前的原始单位，审计用 |
| `effect_label` | int8 或 null | 见 2.5 |
| `mechanism_label` | int8 或 null | 见 2.5 |
| `split_group` | str | 划分时写入 |

### 2.3 结构表（与主表分开，不拼成一张宽表）

- `pockets.parquet`：`uniprot_acc, pocket_id, source(native/predicted), residue_indices(list[int], UniProt 编号), center(3), score, pdb_id, chain, method`。native pocket 只作标签，预测 pocket 才是模型输入。
- `poses.parquet`：`sample_id 或 ligand_id, pocket_id, pose_id, rmsd_to_crystal, vina_score, pose_label(good/poor/null), ligand_sdf_path, receptor_pdb_path, source(pdbbind/crossdocked/casf)`。
- `function_sites.parquet`：`uniprot_acc, residue_index, site_type(binding/active/functional_region/transmembrane/catalytic), source(uniprot/biolip2)`。
- `variants.parquet`：`uniprot_acc, residue_index, source(clinvar/cosmic), significance`。疾病相关残基，只作 function 终点的补充，不单独成任务。
- `piezo_exclusion.parquet`：PIEZO1/2 及同源蛋白的 UniProt accession 列表，划分时强制进 external。

### 2.4 Label mask（与 `RIPPLELabels` 一一对应）

```text
interaction_mask   samples.y_interaction is not null
affinity_mask      affinity_value is not null AND endpoint_type != 4
pocket_mask        该蛋白有 native pocket 标签
pose_mask          该 (ligand, pocket) 有 RMSD-to-crystal
function_mask      该蛋白有 functional residue 注释
effect_mask        effect_label is not null
mechanism_mask     mechanism_label is not null
allosteric_mask    该蛋白在 ASD 有变构 pocket 注释
```

没有某项标签的样本只参与已有标签对应的 loss。mask 由数据决定，不由模型决定。

### 2.5 标签枚举（写死，改动要重新生成缓存）

- effect：`0=agonist/activator, 1=antagonist/inhibitor, 2=other`。GtoPdb 与 ChEMBL mechanism 的映射表放 `configs/label_maps/effect.yaml`，不写进代码。
- mechanism：`0=inhibitor, 1=blocker, 2=opener, 3=modulator, 4=agonist, 5=other`，映射表 `configs/label_maps/mechanism.yaml`。覆盖率低于 1% 的类别并入 other，并在数据报告里记录。

## 3. 阶段 A — 代码与契约对齐（第 1 周）

在下载任何大数据之前完成。这一阶段只改代码和文档。

- [ ] **A1** 改 `RIPPLEBatch`：新增 `function_label: Optional[Tensor]`（残基级多热标签），与输入用的 `function_mask` 分离。新增 `pose_rmsd`、`pose_label`、`pose_mask` 字段，形状 `[B, Kp, Kc]`，只在有 pose 监督的批次非空。
- [ ] **A2** 改 forward：pose/conformer 路径改为软调制。定义一个 `pose_confidence` 标量（来自 matcher 的 marginal 或单独的 pose head），用它缩放 pair context，而不是用 `pair_valid` 把整条交互路径置零。无构象时退化为 pocket-only 条件化，输出仍有限。
- [ ] **A3** 新增 pose head：对每个 (pocket, pose) 输出 quality logit。监督来自 RMSD 阈值（near-native: RMSD < 2Å 为 good，与 CrossDocked2020 惯例一致，阈值写进配置不写死）。
- [ ] **A4** 改 `RIPPLEMultiTaskLoss`：删除 `native_conformer`；新增 `function`（残基级 masked BCE）、`pose_rank`（pairwise logistic ranking，以 `-rmsd` 或 good/poor 为序）。两者在标签缺失时贡献精确为 0。
- [ ] **A5** 改课程：`TrainingStage.STRUCTURE` 重命名为 `POCKET`；`stage_loss_weights` 中 pocket 阶段启用 `native_pocket` 与 `allosteric_pocket`；dti 阶段启用 `interaction`、`affinity`、`native_pocket`；propagation 阶段启用 `function` 与 `effect`；pharmacology 阶段启用 `effect`、`mechanism`。
- [ ] **A6** 损失对齐终版本公式：

```text
L = L_DTI
  + λ_affinity  L_affinity
  + λ_pocket    L_pocket
  + λ_pose      L_pose_rank
  + λ_function  L_function
  + λ_effect    L_effect
  + λ_mechanism L_mechanism
```

  `nonbinder_delta` 保留为 propagation 的正则，不写入论文公式，权重默认降到 0.01 并在配置中可关。
- [ ] **A7** 更新 `configs/ripple_network.yaml` 的 `LOSS_WEIGHTS` 与 `DATA` 段（`NUM_POSES` 取代把 conformer 当监督对象的语义）。
- [ ] **A8** 重写 `README.md` 与 `docs/NETWORK_MODULES.md` 的架构图和训练阶段表，使其与终版本一致。旧的 "matcher 必须先产出有效 pair" 表述全部删除。
- [ ] **A9** 补测试：D1、D2、D3 的完成判据各一条，加进 `tests/test_ripple_training.py`。

完成判据：测试全绿；`NETWORK_MODULES.md` 的数据流图不再出现 "无有效 pair 则预测路径为零"。

## 4. 阶段 B — 数据获取与清洗（第 2–4 周）

所有原始文件只进 `data/raw/`（已在 `.gitignore` 语义内，确认不被跟踪），清洗产物进 `data/processed/`。
每个来源一个脚本，统一放 `scripts/data/`，统一输出到 2.2–2.3 的 schema。

### B1. 主任务数据

- [ ] **B1.1 BindingDB**。下载 TSV。清洗规则：只保留有 UniProt accession 的记录；SMILES 经 RDKit 解析失败的丢弃并计数；亲和力从 nM 统一到 M 再取 `-log10`，原始单位非浓度的（如 % inhibition）不进 affinity，只在有明确活性阈值时进 interaction；同一 (化合物, 蛋白, endpoint) 的多条测量取中位数，保留测量数。
- [ ] **B1.2 ChEMBL**。用 ChEMBL API 或 dump，取 `standard_type ∈ {Kd, Ki, IC50, EC50}`、`standard_relation = '='`、有 `pchembl_value` 的记录。`pchembl_value` 直接作 affinity。assay 类型、`bao_format`、靶点物种保留。
- [ ] **B1.3 BioSNAP ChG-Miner**。只作与 TAMR-DTI 对齐的 benchmark，不进主训练集。单独目录 `data/benchmarks/biosnap/`，沿用其原始划分，不重新切。
- [ ] **B1.4 负样本**。只使用来源明确的负样本：BindingDB/ChEMBL 中同一 assay 下明确无活性的记录，或既定 decoy 协议。**随机未观测对一律不标 0**，标 null，不参与 `L_DTI`。负样本构造策略写成 `docs/NEGATIVE_SAMPLING.md`，因为审稿人会问。

### B2. Pocket 与结构

- [ ] **B2.1 PDBbind**。解析复合物，配体接触残基（重原子距离 < 4Å，阈值进配置）定义为 native pocket，残基编号经 SIFTS 映射到 UniProt。映射失败的复合物剔除并计数，不猜测。实验亲和力进 affinity，endpoint 单独标记。
- [ ] **B2.2 BioLiP2**。ligand-binding residues 与 catalytic residues 进 `function_sites.parquet`，`site_type` 区分。与 PDBbind 重叠的条目以 PDB ID 去重，保留两者来源标记。
- [ ] **B2.3 SIFTS**。PDB chain → UniProt 残基映射表落地为 `data/processed/sifts.parquet`。这是 B2.1、B2.2、B4 的共同依赖，先做。
- [ ] **B2.4 候选 pocket 生成（推理可用的那种）**。对每条有结构的蛋白，用 **apo 结构或 AlphaFold 结构**跑 pocket detector 生成 top-K 候选，K 默认 5。禁止用测试配体坐标定义候选 pocket。native pocket 只写进标签列。detector 的选择与参数写进 `configs/pocket_detection.yaml`，脚本 `scripts/data/build_candidate_pockets.py`。
- [ ] **B2.5 无结构蛋白**。AlphaFold DB 结构按 UniProt 拉取，pLDDT < 70 的残基在 `residue_mask` 中标无效，不直接丢弃整条蛋白。拉取清单与失败清单都落盘。

### B3. Pose 监督（辅助模块专用，覆盖率低是预期）

- [ ] **B3.1 CrossDocked2020**。解析 RMSD-to-crystal、good/poor pose 标签、Vina score、受体与配体文件路径。只保留能经 SIFTS 映射到 UniProt 的受体。
- [ ] **B3.2 CASF**。只作 pose ranking 评测集，不进训练。
- [ ] **B3.3 PDBbind crystal pose**。作为 RMSD=0 的参考 pose 进入 `poses.parquet`，`source=pdbbind`。
- [ ] **B3.4 配对**。pose 监督样本必须同时有候选 pocket 与 RMSD 标签。对不上的 pose 丢弃并在数据报告中给出丢弃率。预期 pose_mask 覆盖率远低于 interaction_mask，这不是问题，是设计。

### B4. Function、mechanism、allostery

- [ ] **B4.1 UniProt**。对主表中所有 accession 批量取 feature：binding site、active site、functional region、transmembrane。进 `function_sites.parquet`。
- [ ] **B4.2 ClinVar / COSMIC**。疾病相关变异映射到残基，作为 function 终点的附加正例，`site_type=disease_associated`。与功能位点分开统计，训练时可通过配置开关决定是否并入 `function_label`。默认**不并入**，先只用 UniProt + BioLiP2，疾病残基留作 case study 分析。
- [ ] **B4.3 Guide to Pharmacology 与 ChEMBL mechanism**。按 `configs/label_maps/` 映射到 effect 与 mechanism 枚举。映射不上的记 null，不记 other（other 只接收明确归类为其他的条目）。
- [ ] **B4.4 ASD**。变构位点经 SIFTS 或序列比对映射到 UniProt 残基，进 `pockets.parquet`（`source=allosteric`）与 `allosteric_mask`。只作辅助损失，不进主任务采样权重。

### B5. PIEZO 排除与外部集

- [ ] **B5.1** 建 `piezo_exclusion.parquet`：PIEZO1 (Q92508)、PIEZO2 (Q9H5I1)，加序列同源搜索（identity ≥ 30%）命中的蛋白。
- [ ] **B5.2** 任何划分脚本读取该表，命中的蛋白全部从 train/val/test 移除，只进 `data/external/piezo/`。
- [ ] **B5.3** 收集 PIEZO case study 材料：RCSB 与 AlphaFold 结构、UniProt 功能与疾病注释、GtoPdb/ChEMBL 中的已知调节剂与其结合证据。整理成 `data/external/piezo/README.md`，逐条注明证据来源与证据等级（结构复合物 / 突变 / 药理推断）。

### B6. 数据报告

- [ ] **B6.1** `scripts/data/report.py` 生成 `data/processed/DATA_REPORT.md`：每来源记录数、清洗丢弃数与原因、每个 mask 的覆盖率、endpoint 分布、物种分布、序列长度分布、pocket 与 pose 的覆盖率。没有这份报告不进入阶段 C。

完成判据：`DATA_REPORT.md` 存在；抽查 20 条样本，`sample_id` 可回溯到原始记录；PIEZO accession 在主表中出现次数为 0。

## 5. 阶段 C — 划分与泄漏控制（第 4 周）

这是论文能成立的前提，单独成阶段。

- [ ] **C1 冷启动划分**。`scripts/data/make_splits.py` 产出三套划分，互不混用：
  - `cold_target`：按蛋白聚类（序列 identity < 30%，用 MMseqs2）切分，测试蛋白与训练蛋白无同簇。
  - `cold_pocket`：按 pocket 结构相似（pocket 残基的局部结构比对或 pocket 级聚类）切分，测试 pocket 与训练 pocket 无同簇。pocket 聚类方法与阈值写进 `docs/SPLITS.md`，因为 cold-pocket 没有现成标准，必须自定义并论证。
  - `cold_both`：同时满足以上两条。
- [ ] **C2 化合物泄漏**。同一 InChIKey 不跨 train/test。scaffold 泄漏单独统计但不作为主划分（避免划分过碎导致对比不公平），数字进 `DATA_REPORT.md`。
- [ ] **C3 基准对齐划分**。BioSNAP 与 TAMR-DTI 用过的 Human / C.elegans / BindingDB 划分原样保留在 `data/benchmarks/`，用于和 baseline 对齐，不与冷启动划分混合报告。
- [ ] **C4 泄漏审计**。`scripts/data/audit_leakage.py` 对每个划分输出：跨 split 的同簇蛋白数（必须为 0）、同 InChIKey 数（必须为 0）、PIEZO 命中数（必须为 0）。审计不过，训练入口直接拒绝加载该划分。

完成判据：三套划分的审计报告全部为 0 泄漏；`docs/SPLITS.md` 写明 cold-pocket 的定义与阈值选择理由。

## 6. 阶段 D — 特征缓存（第 5 周）

现有 `scripts/pre_extract.py` 产出的是 TAMR 的池化缓存（蛋白 128 token、药物 3D 固定 64 原子）。RIPPLE 需要**未池化的残基级**特征与坐标，二者不能混用。

- [ ] **D1** 新脚本 `scripts/build_ripple_cache.py`，不改 `pre_extract.py`（baseline 还要用）。产物按 UniProt / SMILES 分键：
  - 残基级 PLM 特征 `[L, 1024]` 与有效 mask，L 为真实序列长度，不池化到 128；
  - 残基坐标 `[L, 3]`，来自 PDB 或 AlphaFold，无坐标的残基 mask 掉；
  - 残基图 `edge_index`：Cα 距离 < 8Å 的稀疏边（阈值进配置）；
  - 候选 pocket 的残基索引 `[Kp, Np]` 与 pocket score；
  - 药物 1D token、2D 图、多构象 3D（复用现有 drug3d 缓存，不重建）；
  - pose 级坐标与 RMSD（仅 pose_mask 为真的样本）。
- [ ] **D2** 长序列处理。PIEZO1 超过 2500 残基，主表里也有长蛋白。缓存按真实长度存，batch 时再截断或分块。截断策略（保留有 pocket 与 function 注释覆盖的区段，而不是简单取前 N）写进 `docs/NETWORK_MODULES.md`。
- [ ] **D3** `scripts/check_ripple_cache.py`：校验形状、mask 与索引不越界、pocket 索引都落在 `residue_mask` 内、pose RMSD 非负。
- [ ] **D4** 缓存版本号。schema 或特征定义变化时递增，训练入口校验版本，不匹配即报错退出。

完成判据：`check_ripple_cache.py` 对 train/val/test 三个 split 全部通过；抽查一条 PDBbind 样本，pocket 索引与 SIFTS 映射手工核对一致。

## 7. 阶段 E — 数据加载与 batch 适配（第 6 周）

README 已指出这是当前缺口：把缓存加 pocket/残基图元数据转成 `RIPPLEBatch`。

- [ ] **E1** `src/data/ripple_dataset.py`：`RIPPLEDataset` 读 `samples.parquet` + 缓存，返回 `RIPPLEBatch` 与 `RIPPLELabels`。缺失标签返回 None 或 NaN，由 loss 侧的 mask 处理，不在 dataset 里填 0。
- [ ] **E2** `src/data/ripple_collate.py`：变长残基、变长 pocket、变数目 pose 的 padding 与 mask。edge_index 加 batch 偏移。
- [ ] **E3** 多来源采样。`MultiSourceSampler`：每个 batch 以主任务样本为底，按配置概率混入有 pocket / pose / function 标签的样本，保证稀疏任务在每个 epoch 见到足够正例。采样权重进配置，不写死。
- [ ] **E4** 与现有 `DTIDataset` 隔离。baseline 路径不动，RIPPLE 走新 dataset。两者不要共用 collate。

完成判据：用真实缓存跑一个 batch，打印每个字段形状与每个 mask 的有效数，与 schema 一致；一个只有 interaction 标签的 batch 能完整前向并反向。

## 8. 阶段 F — 训练入口（第 6–7 周）

现有 `src/core/main_ds.py` 只训练 TAMR baseline。RIPPLE 需要自己的入口，不往 baseline 入口里加分支。

- [ ] **F1** `src/core/train_ripple.py`：单卡入口。读 `configs/ripple_network.yaml`，构建模型、loss、优化器（复用 `build_optimizer_param_groups`），按课程阶段循环。支持 `--stage` 单独跑某一阶段，便于调试。
- [ ] **F2** DeepSpeed 入口。复用 `scripts/run_deepspeed.sh` 与 `configs/ds_zero2.json`，新增 `scripts/run_ripple.sh`，指向 `train_ripple.py`。不与 baseline 共用 save_dir。
- [ ] **F3** 课程调度。每个阶段记录自己的 best checkpoint，下一阶段从上一阶段 best 初始化。joint 阶段用 `backbone_lr_multiplier` 降低几何编码器学习率（已实现，确认接线即可）。
- [ ] **F4** 日志。每个 loss 项单独记录 raw 值与 `label_count`，而不只记录 total。label_count 为 0 的项不参与该 step 的梯度，日志里标 `skipped`。用现有 swanlab 集成。
- [ ] **F5** 梯度与数值护栏。检测非有限 loss 直接中止并保存现场；梯度裁剪阈值进配置。
- [ ] **F6** 断点续训。沿用 baseline 的 epoch tag 与 prune 逻辑，但 client state 增加 `stage` 字段。
- [ ] **F7** Smoke test。`scripts/smoke_train.py`：用 64 条样本跑 2 个 step，断言 loss 有限、checkpoint 可重新加载、续训后 loss 连续。进 CI 前的本地必跑项。

完成判据：smoke test 通过；在一个小子集上完整跑完五个阶段不报错；日志里能看到每项 loss 的 label_count。

## 9. 阶段 G — 评测（第 7–8 周）

指标按创新点分开报，不混成一个总分。

- [ ] **G1 DTI 主指标**。cold-target / cold-pocket / cold-both 三套划分上的 AUROC、AUPRC、F1。affinity 按 endpoint 分开报 RMSE 与 Pearson，不报混合 endpoint 的单一数字。
- [ ] **G2 Pocket localization**。候选 pocket 排序的 Top-1 命中率与原生 pocket 的重叠（IoU 或覆盖率）。这是主创新一的直接证据，必须和 DTI 数字一起报。
- [ ] **G3 Pose ranking**。只在 CASF 与 CrossDocked 测试子集上报：good pose 的 enrichment、ranking AUC。明确标注这是辅助模块的指标，样本数与主任务不同。
- [ ] **G4 Function propagation**。残基级 functional-site 预测的 AUPRC（正例稀疏，不用 accuracy）。按 site_type 分开报。
- [ ] **G5 消融**。最少四组：去掉 pocket 条件化、去掉 pose 调制、去掉 propagation、完整模型。消融配置用 yaml 覆盖，不改代码。
- [ ] **G6 Baseline 对齐**。在 BioSNAP / Human / C.elegans / BindingDB 原始划分上跑 TAMR baseline（现有入口）与 RIPPLE，保证对比使用同一划分与同一指标实现。指标实现抽到 `src/eval/metrics.py`，两边共用。
- [ ] **G7** `scripts/evaluate.py` 一次运行产出 `outputs/<exp>/metrics.json` 与一份 markdown 表。

完成判据：`metrics.json` 包含 G1–G4 全部字段；消融四组都能从同一 checkpoint 族加载；baseline 数字可与已发表 TAMR-DTI 结果对上（允许小幅浮动，浮动需解释）。

## 10. 阶段 H — PIEZO 外部 case study（第 8–9 周）

- [ ] **H1** 确认 PIEZO1/2 及同源蛋白不在任何训练 checkpoint 的数据里（复跑 C4 审计）。
- [ ] **H2** 对已知调节剂，在 PIEZO 结构上生成候选 pocket（同样不使用配体坐标），跑推理，导出：pocket 排序、残基传播权重、到达 pore / gating / 功能区域的 sink 质量。
- [ ] **H3** 与文献证据对照：哪些预测落在已有结构或突变证据支持的区域，哪些没有。输出 `outputs/piezo/case_study.md`，每条结论标注证据等级。
- [ ] **H4** 传播可视化脚本 `scripts/visualize_propagation.py`：残基图上的 edge flux 与 sink weight。图中注明 "predictive information flow, not a physical energy pathway"。

完成判据：case study 文档存在，且没有任何一句把预测描述成物理能量路径或真实传播路径。

## 11. 阶段 I — 复现与交付（第 9 周）

- [ ] **I1** `docs/REPRODUCE.md`：从原始数据到 metrics.json 的完整命令序列，含每个阶段的预期耗时与硬件。
- [ ] **I2** 配置锁定。最终实验用的 yaml 进 `configs/experiments/`，文件名带划分名与阶段名。
- [ ] **I3** 随机种子。所有划分、采样、训练入口固定种子，三次重复的方差进报告。
- [ ] **I4** 依赖锁定。`requirements.txt` 已有，补一个实际跑通的版本记录 `requirements.lock.txt`（pip freeze），不手写。

## 12. 依赖关系

```text
A 代码对齐
  └─> B 数据清洗 ──> B6 数据报告
        └─> C 划分与泄漏审计
              └─> D 特征缓存
                    └─> E batch 适配
                          └─> F 训练入口
                                └─> G 评测
B5 PIEZO 排除 ──> C4 审计 ──> H case study
G6 baseline 对齐依赖现有 main_ds.py，可与 F 并行
```

可并行的只有：G6 的 baseline 复现（不依赖 RIPPLE 改动），以及 B1 各数据源脚本之间。
其余按上面的链走，不允许跳。

## 13. 风险与对应动作

| 风险 | 表现 | 动作 |
|---|---|---|
| pose 标签覆盖率太低，辅助模块学不动 | `DATA_REPORT.md` 中 pose_mask 覆盖率 < 5% 主任务样本 | 可接受。pose loss 只在有标签样本上启用，不因此扩大标签定义。若 enrichment 不优于随机，论文中如实报告并降低该模块的表述权重 |
| cold-pocket 定义站不住 | 审稿人质疑 pocket 聚类 | `docs/SPLITS.md` 先写清定义、阈值、替代方案与选择理由，划分脚本把阈值留成参数 |
| 长蛋白与 PIEZO 截断丢失功能区 | 功能注释落在截断范围外 | D2 的截断以注释覆盖为准，不用固定前 N |
| 负样本策略被质疑 | 随机负样本引入假负 | B1.4 只用明确无活性记录，随机未观测对标 null |
| function 终点过于稀疏 | function AUPRC 方差大 | 按 site_type 分开报，并报支持数。支持数低于阈值的 site_type 不进主表 |
| 现有 joint matcher 与终版本冲突 | 旧测试假设 "无 pair 则路径为零" | 阶段 A 先改测试再改代码，以终版本为准 |

## 14. 完成定义

终版本实施完成，当且仅当：

1. 阶段 A 的偏差 D1–D5 全部落地，测试全绿。
2. `DATA_REPORT.md` 与三套划分的泄漏审计通过。
3. 训练入口能在五个阶段上跑通，并产出带 label_count 的日志。
4. `metrics.json` 同时包含 DTI、pocket localization、pose ranking、function prediction 四组指标。
5. PIEZO case study 文档存在，且训练数据审计确认 PIEZO 及其同源蛋白被排除。
