# 原始数据收集

服务器项目：`/home/rifo/RIPPLE-DTI`。原料写入 `data/raw/`，不提交 Git。
此步骤使用 CPU、网络和磁盘，不使用 GPU。下载完成不代表数据已经可以训练。

## 2026-10-21 收集结果

| 来源 | 本次核验结果 |
|---|---|
| BindingDB 202610 | 3,243,660 条原始测量记录；三个 ZIP 均通过 CRC 和官方 MD5 |
| BioLiP | 989,058 条原始位点记录；位点、配体、受体序列文件均通过 gzip 校验 |
| GtoPdb 2026.3.1 | 24,968 条相互作用、14,029 条配体记录；五张 CSV 发布版本一致 |
| SIFTS | 1,047,374 条链区间映射，快照文件已校验 |
| PDBbind 2020.R1 | 19,037 个复合物，76,148 个归档文件；四种文件各 19,037 份，无缺失 |
| UniProt | 请求 48,231 个 canonical accession，返回 47,245 条有效完整序列，未返回 986 个 |
| RCSB 通道结构 | 3,225 个候选 PDB 的原始 mmCIF，覆盖 297 个目录 accession，下载失败数为 0 |

986 个 UniProt ID 见服务器 `data/raw/uniprot/summary.json` 的
`not_returned_under_requested_id`。这些 ID 尚需排查删除、合并或历史版本，不能用空序列代替。
目录中的 `MICU1_RAT`、`MICU2_RAT` 已分别解析为 `Q6P6Q9`、`Q99P63`，并用于重新查询结构。
这些数量是原始来源规模，不是清洗、划分后可训练的样本数。

## 重跑命令

需要 Python 3.10+ 和 `curl`。安装了 `aria2c` 时，BindingDB 使用四连接下载和断点续传。
服务器可以使用 `/home/rifo/miniconda3/bin/python`，无需安装模型依赖。

```bash
cd /home/rifo/RIPPLE-DTI
PY=/home/rifo/miniconda3/bin/python

$PY scripts/ripple/download_sources.py --workers 3
$PY scripts/ripple/download_references.py uniprot --workers 4

$PY scripts/ripple/census_channel_structures.py \
  --targets data/raw/gtopdb/targets_and_families.csv \
  --uniprot-aliases data/raw/uniprot/features_entry_names.tsv \
  --output data/raw/channel_census --cutoff 2026-10-21

$PY scripts/ripple/download_references.py structures --workers 6
```

先完成基础下载并导入 PDBbind，再执行 UniProt 收集，确保清单包含 PDBbind 索引对应的蛋白。
同一命令不应并发运行。已完成文件仅在 URL、SHA256 与元数据一致时复用；失败文件可重跑。
BindingDB 还核对官方 MD5 和 ZIP CRC，其余 gzip 检查完整压缩流。
下载过程中保留至少 15 GiB 可用磁盘空间，不展开大型压缩包。

同一目录代表一个固定快照。需要更新版本时使用新的 `--raw` 目录和新的 census 输出目录，
不要将不同发布版本混在已有文件中。

## 来源与文件

| 来源 | 目录 | 内容 |
|---|---|---|
| BindingDB 202610 | `bindingdb/` | 全量测量、assay 描述、Reactant set 到 assay 的映射及官方 MD5 |
| GtoPdb | `gtopdb/` | interactions、ligands、targets、targets_and_families、ligand_id_mapping、字段定义 |
| BioLiP | `biolip/` | 冗余位点表、配体身份、受体序列及字段定义 |
| SIFTS | `sifts/` | PDB 链到 UniProt 的区间映射 |
| PDBbind 2020.R1 | `pdbbind/` | `index.tar.gz`、`P-L.tar.gz`，复用用户已下载的本地文件 |
| UniProt | `uniprot/` | 来源 accession 清单、分批完整序列及功能注释、未返回 ID 清单 |
| RCSB | `channel_census/`、`rcsb/cif/` | 通道目录查询及命中的原始 mmCIF 文件 |

PDBbind 官方页面需要登录。本次无需再次登录，两个文件从本地通过 SSH 复制后，
用本地 SHA256 验证传输结果并登记来源：

```bash
$PY scripts/ripple/download_sources.py \
  --import-pdbbind <index.tar.gz的SHA256> <P-L.tar.gz的SHA256>
```

每个原始文件旁有 `.metadata.json`，记录 URL、获取或校验时间、大小、SHA256、状态。
汇总见 `data/raw/download_manifest.json`；来源统计见各目录的 `summary.json`。
UniProt 的旧 ID、替代 ID 和 isoform 会单独列出，不自动当作同一条 canonical 序列。

## 后续边界

- RCSB 文件是原始非对称单元，包含生物学装配定义；还需要选取并生成适合模型的装配。
- SIFTS 区间表不是逐残基映射；筛选实际使用的结构后再获取对应 XML，并核验构建体和残基编号。
- 通道 census 排除了目录直接标注的 PIEZO；全家族同源隔离仍需序列聚类后检查。
- 分子标准化、身份连接、单位与终点统一及方向标签初筛已完成，结果见 [统一格式](UNIFIED_DATA.md)。证据审核、跨来源实验去重、结构与残基映射、数据划分、特征提取和训练缓存仍按 [TODO](TODO.md) 推进。
