# 统一数据格式：第一步

入口：`scripts/ripple/prepare_unified.py`。这一层是经过标准化的来源记录，
后续结构审核、数据划分和特征提取完成后，再生成 [训练缓存契约](CACHE.md) 中的样本。
当前所有记录的 `trainable=False`，`split=null`。

## 服务器结果

2026-10-21 已在 `/home/rifo/RIPPLE-DTI/data/processed/unified_v1/` 完成全量处理，
生成 13 张 Parquet 表。完整统计、版本、代码与输出文件哈希见该目录 `report.json`。

| 内容 | 数量 |
|---|---:|
| 标准化分子 | 1,461,283 |
| 不同输入 SMILES | 1,509,854 |
| 来源化合物交叉引用 | 1,525,168 |
| 有完整序列的蛋白身份 | 47,245 |
| BindingDB 原始记录 / 分终点测量 | 3,243,660 / 3,249,488 |
| BindingDB 实验描述 / 测量与实验关联 | 224,448 / 3,185,528 |
| BioLiP 位点 | 989,058 |
| PDBbind 复合物 | 19,037 |
| GtoPdb 药理记录 | 24,968 |
| RCSB 结构 / 聚合物实体 | 3,225 / 5,278 |
| SIFTS 区间 | 1,047,374 |

10,146 种输入未通过化学标准化，保留原值和失败原因；113 个来源化合物 ID
对应多个标准化身份，已在交叉引用表标记冲突。3,189 个 InChIKey 有多个保留的
SMILES 表示，后续构象生成应沿用被冻结的代表与版本。

标签及基本身份初筛得到 BindingDB 2,411,796 条、BioLiP 302,605 条、
PDBbind 14,970 条、GtoPdb 通道药理 1,619 条候选记录。
这些是证据记录数，尚未完成结构审核、实验去重或划分，不能当成训练样本数。

`audit/raw_integrity.json` 确认 raw 内 7,472 个文件（含下载元数据）及目录清单、
内容哈希、大小、修改时间、权限均未变化；本次中止尝试前的快照也与最终快照一致。

## 运行

在项目根目录使用独立 CPU 环境：

```bash
python -m venv .venv-data
.venv-data/bin/pip install -r requirements-staging.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.venv-data/bin/python scripts/ripple/prepare_unified.py \
  --raw data/raw --out data/processed/unified_v1 --workers 24
```

中断后在原命令末尾加 `--resume`。已完成的来源解析和分子标准化会复用。
代码、依赖或原料变化时拒绝复用旧结果，应选择新的输出目录。
`report.json` 只在全流程及 raw 完整性验证成功后生成。

服务器当前的基础 Python 存在 `readline` 导入崩溃，pytest 可关闭捕获插件运行；
处理入口不依赖 readline。当前本地和服务器均通过 26 项相关测试：

```bash
.venv-data/bin/python -m pytest -p no:capture \
  tests/test_unified_staging.py tests/test_source_alignment.py -q
```

脚本不下载、不解压到 raw、不更新 raw 元数据。运行前后对 raw 的文件清单、
SHA-256、大小、修改时间和权限进行比较，结果写到输出目录的 `audit/`。
读取可能触发文件系统的访问时间更新；访问时间不作为完整性判据。
输出目录不能位于 raw 内，也不能包含指向其他位置的符号链接。

## 表及连接键

文件均为 Parquet；嵌套的原始证据保留为 JSON 字符串，不作为模型输入。

| 文件（相对输出目录） | 一行表示 | 连接键 |
|---|---|---|
| `identities/molecules.parquet` | 一个标准化分子 | `molecule_id = inchikey` |
| `identities/chemical_inputs.parquet` | 一种输入 SMILES 及处理结果 | `chemical_input_id → molecule_id` |
| `identities/molecule_xrefs.parquet` | 一个来源化合物 ID 的一种表示 | `source + source_id → chemical_input_id` |
| `identities/proteins.parquet` | 一条有序列的 UniProt 记录 | `protein_id = accession:taxon_id` |
| `sources/bindingdb/measurements.parquet` | 一条来源记录中的一个终点 | `record_id`，保留 Reactant set ID |
| `sources/bindingdb/assays.parquet` | 一条实验描述 | `ENTRYID + ASSAYID` |
| `sources/bindingdb/assay_links.parquet` | 测量与实验的关联 | `REACTANT_SET_ID → ENTRYID_ASSAYID` |
| `sources/biolip2/sites.parquet` | 一个结构中的配体接触位点 | `record_id`，保留 PDB、链、位点和 CCD |
| `sources/pdbbind/complexes.parquet` | 一个来源复合物 | `pdb_id`，保留压缩包内文件路径 |
| `sources/gtopharm/interactions.parquet` | 一条药理证据 | `record_id`，保留 ligand/target ID |
| `references/sifts_segments.parquet` | 一个 SIFTS 映射区间 | `pdb_id + auth_chain + accession` |
| `references/structures.parquet` | 一个已下载的 RCSB 非对称单元 | `pdb_id`，内容哈希与装配定义 |
| `references/structure_entities.parquet` | 一个结构中的聚合物实体 | `pdb_id + entity_id` |

同一分子、蛋白可以对应许多实验。身份连接不合并实验、不平均测量值。
原始证据和 `evidence_hash` 保留；重复证据仅计数，跨来源实验去重仍需审核。
来源版本由 raw 快照及下载元数据固定，处理版本和依赖版本写入 `report.json`。

## 关键规则

- 化学处理 `ripple-parent-v1`：RDKit 校验，移除白名单内的常见无机反离子及结晶水，中和可中和电荷，保留立体化学；多个有机片段、未知无机伴随片段或立体标注冲突先隔离。未通过的输入仍在表中，`molecule_id=null` 并记录原因。
- 不做互变异构体变换。同一标准 InChIKey 对应的多个 SMILES 全部保存；`canonical_smiles` 为排序后的稳定代表。来源 ID 对应不同分子时在交叉表标记 `source_id_conflict`。
- 蛋白保留 accession、物种、isoform、全长序列和 SHA-256。缺失序列不造蛋白身份；构建体差异保留到结构审核。多蛋白靶点不把证据强行赋给第一条链。
- Kd/Ki/IC50/EC50 分开，终点编码为 0/1/2/3。精确正浓度转换为 `-log10(mol/L)`。不等号、近似、范围、未知终点保留原值，不生成精确回归标签。
- 效应为增强 0、抑制 1。Action 必须有明确方向且不与 Type 冲突；仅 Binding、未知、混合、部分/偏向激动等记录进入待审。标准化后同一药物和蛋白方向相反时，阻止直接生成效应候选。
- GtoPdb 保留全部药理记录，`channel_catalogue` 标记通道目录；首版仅将物种核对通过的通道记录列入方向候选。其他物种仍保留，但审核状态为 pending。
- `label_candidate` 只表示标签及基本身份满足初筛，不代表可训练。`blocking_reasons` 明确列出身份、化学、标签和后续处理缺口。

## 结构边界

RCSB 下载文件是非对称单元。表中保存装配 ID、生成表达式、变换矩阵、
实体构建体序列和来源物种信息，`audit_status=pending`。
PDBbind 保留配体 SDF 解析结果及原压缩包成员指针；未将其自动视为已审核装配。

SIFTS 当前原料只有区间表，不能按起止位置线性补出逐残基映射。
作者编号、插入码和缺失坐标还需结合逐残基 SIFTS 与 mmCIF 审核。
这一阶段不生成 canonical 残基索引、跨亚基查询、坐标张量或训练 manifest。

## 下一步

先审核装配、构建体与逐残基映射，并清点可用的药物、蛋白与结构组合。
补查结果写到 `data/processed/` 下的独立参考缓存，继续保持 raw 不变。
之后做跨来源证据去重、冻结序列/同源分组与 PIEZO 家族隔离，再构建模型输入和 GPU 特征。

旧的 `prepare_sources.py` 包含向 raw 写入补查数据的路径；后续复用该入口前，
需要先将补查缓存迁移到 processed。
