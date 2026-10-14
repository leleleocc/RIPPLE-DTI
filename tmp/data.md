可以。RIPPLE-DTI 不适合把所有数据库直接拼成一个大表，因为它们的标签语义不同。更合理的是按任务分层使用。

## 1. 主任务：DTI 与亲和力

| 数据集 | 主要内容 | RIPPLE 用途 | 链接 |
|---|---|---|---|
| **BindingDB** | SMILES、UniProt 靶点、Ki/Kd/IC50 等实验结合数据 | DTI 二分类、亲和力回归主数据源 | [BindingDB](https://www.bindingdb.org/) |
| **ChEMBL** | 标准化 bioactivity、assay、target、compound、activity type | 大规模 DTI/DTA 预训练 | [ChEMBL](https://www.ebi.ac.uk/chembl/) |
| **Papyrus** | 整合并标准化 ChEMBL、BindingDB 等来源的生物活性数据 | 作为统一 bioactivity 扩展集 | [Papyrus 官网](https://papyrus.nibiohn.go.jp/) |
| **BioSNAP ChG-Miner** | 药物-基因/靶点相互作用网络，约 1.5 万条边 | 传统 DTI benchmark，不建议作为唯一训练源 | [BioSNAP ChG-Miner](https://snap.stanford.edu/biodata/datasets/10002/10002-ChG-Miner.html) |
| **Drug Target Commons** | 多来源药物-靶点和亲和力记录 | 可作为外部补充，但需要做来源和 assay 清洗 | [DTC](https://drugtargetcommons.fimm.fi/) |

### 推荐用法

第一阶段建议：

```text
BindingDB + ChEMBL/Papyrus
```

用于训练 DTI 和 endpoint-specific affinity。

其中：

- `Kd/Ki/IC50/EC50` 不要直接混为一个连续标签；
- 使用 `endpoint_type` 区分不同实验终点；
- assay、物种、细胞体系和测量单位需要保留；
- 缺失亲和力不能当作负样本；
- DTI 阴性样本应使用明确的 decoy 或严格的负样本构造策略。

BioSNAP 更适合作为和 TAMR-DTI 对齐的 benchmark，而不是 RIPPLE 的唯一训练集。

---

## 2. Pocket 与构象监督

| 数据集 | 主要内容 | RIPPLE 用途 | 链接 |
|---|---|---|---|
| **PDBbind** | 蛋白-配体复合物结构、结合口袋、实验亲和力 | native pocket、bound conformer、结构亲和力监督 | [PDBbind](https://www.pdbbind.org.cn/) |
| **BioLiP2** | 蛋白-配体结合位点和残基接触注释 | pocket residue 标签、接触残基监督 | [BioLiP/BioLiP2](https://zhanggroup.org/BioLiP/) |
| **RCSB PDB** | 蛋白、配体和复合物三维结构 | 结构坐标、配体 pose、候选 pocket 构建 | [RCSB PDB](https://www.rcsb.org/) |
| **PDB-REDO/SIFTS** | 结构质量和 PDB-UniProt 残基映射 | PDB chain 到 UniProt sequence 的映射 | [SIFTS](https://www.ebi.ac.uk/pdbe/docs/sifts/) |
| **AlphaFold DB** | 蛋白预测结构 | 无实验结构目标的 apo structure 和候选 pocket 生成 | [AlphaFold DB](https://alphafold.ebi.ac.uk/) |

### 这部分对应 RIPPLE 的标签

- `native_pocket_index`
- `native_conformer_index`
- `pocket_mask`
- `pocket_valid_mask`
- `pocket_score`
- residue-level contact 或 binding-site mask

需要注意：

- PDBbind/BioLiP2 中的 pocket 是实验复合物中的 native pocket；
- 推理时不能使用测试配体坐标直接定义 pocket；
- 应使用 apo 结构、预测结构或 pocket predictor 生成候选 pocket；
- native pocket 只作为训练和评估标签。

推荐的候选 pocket 生成方式是：

```text
PDB/AlphaFold structure
    -> pocket predictor 或几何 pocket detection
    -> top-K candidate pockets
    -> RIPPLE joint pocket-conformer matcher
```

---

## 3. Pharmacology、effect 与 mechanism

| 数据源 | 主要内容 | RIPPLE 用途 | 链接 |
|---|---|---|---|
| **IUPHAR/BPS Guide to Pharmacology** | ligand、target、agonist/antagonist/inhibitor 等药理作用 | effect direction、mechanism 标签 | [Guide to Pharmacology](https://www.guidetopharmacology.org/) |
| **ChEMBL mechanism annotations** | compound-target mechanism、作用类型和 assay | mechanism 辅助监督 | [ChEMBL](https://www.ebi.ac.uk/chembl/) |
| **DrugCentral** | 药物、靶点、适应症和药理信息 | mechanism/effect 的外部补充 | [DrugCentral](https://drugcentral.org/) |
| **UniProt** | 蛋白序列、active site、binding site、功能区间 | function mask、functional residue 标签 | [UniProt](https://www.uniprot.org/) |

这些数据通常远小于 BindingDB/ChEMBL，因此不建议让 effect/mechanism 任务主导训练。

更合理的训练方式是：

```text
DTI/DTA：大规模主任务
effect/mechanism：稀疏辅助任务
function-site：残基传播监督
```

没有功能或机制注释的样本应使用 mask 忽略，而不是标记为 negative。

---

## 4. Allosteric 与长程传播监督

| 数据源 | 主要内容 | RIPPLE 用途 | 链接 |
|---|---|---|---|
| **ASD Allosteric Database** | 变构位点、变构配体和相关结构 | allosteric pocket auxiliary task | [ASD](http://mdl.shsmu.edu.cn/ASD/) |
| **BioLiP2** | 结合位点、残基接触和配体信息 | pocket/function residue 监督 | [BioLiP2](https://zhanggroup.org/BioLiP/) |
| **UniProt feature annotations** | active site、binding site、functional region | propagation sink/function mask | [UniProt](https://www.uniprot.org/) |
| **PDB/RCSB** | 多状态结构、配体复合物、突变结构 | 长程结构变化和位点验证 | [RCSB PDB](https://www.rcsb.org/) |

ASD 的数据量和标签覆盖不如 BindingDB/ChEMBL，建议只作为：

```text
allosteric_pocket BCE auxiliary loss
```

不建议把它作为 RIPPLE 的主体训练数据。

---

## 5. PIEZO 外部验证数据

PIEZO1/2 更适合做冷靶点外部 case study，而不是主训练集。

推荐来源：

- [PIEZO1 UniProt Q92508](https://www.uniprot.org/uniprotkb/Q92508/entry)
- [PIEZO2 UniProt Q9H5I1](https://www.uniprot.org/uniprotkb/Q9H5I1/entry)
- [PIEZO 结构检索：RCSB PDB](https://www.rcsb.org/)
- [PIEZO 预测结构：AlphaFold DB](https://alphafold.ebi.ac.uk/)
- [药理作用和配体：Guide to Pharmacology](https://www.guidetopharmacology.org/)
- [药物活性数据：ChEMBL](https://www.ebi.ac.uk/chembl/)

PIEZO case study 可以分析：

- Yoda/Jedi 等已知调控 pocket；
- blade-beam-anchor-cap 区域；
- pore 与 ion-conduction 相关区域；
- pocket-to-function propagation 是否到达已知功能区域；
- closed/intermediate/open 不同结构状态下的预测变化。

但是这些结果应称为外部验证或案例研究，不能当作大规模监督真值。

---

## 最推荐的数据组合

### 可行的 MVP

```text
BindingDB
+ PDBbind
+ BioLiP2
+ UniProt
+ RCSB PDB
```

对应：

```text
DTI/affinity
+ native pocket/conformer
+ residue contact
+ functional mask
```

### 完整论文版本

```text
BindingDB + ChEMBL/Papyrus
+ PDBbind + BioLiP2
+ UniProt + SIFTS
+ GtoPdb
+ ASD
+ PIEZO external case study
```

对应 RIPPLE 的训练结构：

| 训练阶段 | 数据 |
|---|---|
| `structure` | PDBbind、BioLiP2、PDB |
| `dti` | BindingDB、ChEMBL、Papyrus |
| `propagation` | UniProt functional sites、BioLiP2 |
| `pharmacology` | GtoPdb、ChEMBL mechanism |
| `joint` | 所有有对应 mask 的样本 |
| external evaluation | PIEZO1/2 及其结构和药理资料 |

核心原则是：不同数据源不直接拼接成一个统一标签，而是通过独立 label mask 进入多任务损失。这样才符合当前 [RIPPLEMultiTaskLoss](/Users/rifo/Code/Drug-Screening/RIPPLE-DTI/src/models/ripple_training.py:75) 的设计，也能避免把“没有注释”误当成“负样本”。