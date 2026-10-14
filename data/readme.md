# 数据集

RIPPLE 的数据按标签来源分开存放，不合并成一张宽表。每个来源只提供它真实
拥有的标签，缺失的标签保持缺失，不补成 0，也不当成负样本。某个来源能监督
哪些损失，由 `src/ripple/data/schema.py` 里的 `SourceTable` 决定。

Kd、Ki、IC50、EC50 是四种不同的实验终点，分别回归，绝不混成一个连续标签。

PIEZO1（UniProt Q92508）和 PIEZO2（Q9H5I1）及其近同源蛋白从所有训练集剔除，
单独留作外部冷靶点 case study。

## 目录约定

```text
data/raw/<source>/        下载的原始文件，不改动
data/processed/<source>/  prepare_sources.py 清洗后的样本表
data/benchmarks/          只用于和既有工作对齐的划分，不进主训练集
data/external/            PIEZO 排除名单与 case study 材料
```

## 现在就用：最小可训练集

这四个来源覆盖 DTI、亲和力、pocket、功能残基四条监督，是第一版训练的全部数据。
初版就用 BindingDB、BioLiP2、PDBbind 三个库，加上 SIFTS 做残基编号映射，UniProt 按用到的蛋白查询位点注释。BindingDB 负责亲和力，BioLiP2 负责口袋残基和催化残基，PDBbind 
两者都有但口袋残基要自己算。两个主创新点都能做：候选口袋从 AlphaFold 预测结构和 PDBbind 的蛋白结构上算，口袋对不对由 BioLiP2 和 PDBbind 监督，功能残基由 UniProt 和 BioLiP2 监督。pose 
置信度这一项初版先不做损失，因为 PDBbind 只有晶体 pose，没有好坏对比，等以后补 CrossDocked2020。效应和机制也不在初版范围里，它们的标签在另外的库。

### BindingDB

蛋白质-配体结合亲和力的公开数据库，每条记录是一次实验测量：配体的 SMILES、
靶点的 UniProt accession，以及 Ki、Kd、IC50、EC50 等测得的亲和力。数据来自
文献和专利，按月更新，2026-09 版约 320 万条测量、145 万个化合物、1.1 万个靶点。

主数据源。SMILES 和靶点给 DTI 二分类（`L_DTI`），浓度型亲和力取负对数后给
亲和力回归（`L_affinity`）。不是浓度单位的测量（如抑制率百分比）不进亲和力，
只在有明确活性判定时进 DTI。同一化合物、蛋白、终点的多次测量取中位数。

一条记录是一次实验测量。

```text
Protein
 ├── UniProt accession
 └── 靶点名、物种

        +

Ligand
 └── SMILES

        ↓

Affinity label
 ├── Kd
 ├── Ki
 ├── IC50
 └── EC50

没有序列，没有结构，没有 pocket。
```

下载：[BindingDB_All_202610_tsv.zip](https://bindingdb.org/rwd/bind/chemsearch/marvin/SDFdownload.jsp?download_file=/rwd/bind/downloads/BindingDB_All_202610_tsv.zip)（567 MB，[下载页](https://bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp)）

### BioLiP2

从 PDB 中整理出的蛋白质-配体相互作用数据库，每条记录是一个结合位点，给出
结合残基、催化残基、配体、受体序列和对应的 UniProt accession。2026 年更新为
BioLiP3，文件仍放在原地址，当前约 99 万条记录。

提供两类残基级标签：结合残基是 native pocket 的监督（`L_pocket`），催化残基
是功能残基的监督（`L_function`）。残基编号是 PDB 作者编号，必须经 SIFTS 映射
到 UniProt 编号后才能用，映射失败的记录剔除，不猜测。记录里大量是金属离子、
辅因子、核酸和多肽，做 DTI 时只保留类药小分子。催化残基列只有约 17% 的行有值，
空值是"未注释"，不是"无功能"。

一条记录是一个结合位点。

```text
Protein
 ├── PDB ID + chain
 ├── sequence
 ├── UniProt accession
 └── structure（实验复合物，必有）

        +

Ligand
 ├── CCD 三字母码（如 HEM，不是 SMILES）
 └── 3D 坐标（复合物里的坐标）

        ↓

Site label
 ├── binding residues      → L_pocket
 └── catalytic residues    → L_function（多数为空）

Affinity
 └── 基本没有，不作为这个库的标签
```

下载：[BioLiP.txt.gz](https://zhanggroup.org/BioLiP/download/BioLiP.txt.gz)（[字段说明](https://zhanggroup.org/BioLiP/download/readme.txt)）

### PDBbind

蛋白质-配体复合物的结构数据库，每个条目是一个实验解析的复合物，配有实测亲和力。
v2020 的 refined 集约 5300 个高质量复合物，蛋白给 PDB 格式，配体给 Mol2 和 SDF。
v2020 及更早版本免费，但需要注册登录后下载。

三个用途：复合物里配体接触的残基（重原子距离阈值见配置）定义 native pocket，
监督 `L_pocket`；实测亲和力进 `L_affinity`，终点单独标记；晶体 pose 作为
RMSD 为 0 的参考 pose，监督 `L_pose_rank`。PDBbind 没有完整的 pocket × conformer
实验配对标签，所以不做 pair matching，只做同一 pocket 内的 pose 排序。

一条记录是一个实验解析的复合物。

```text
Protein
 ├── PDB code
 └── structure（蛋白 PDB）

        +

Ligand
 └── 3D 坐标（Mol2 / SDF，晶体 pose）

        ↓

Pocket
 └── 配体接触残基（距离阈值现算）   → L_pocket

Pose
 └── 晶体 pose，RMSD = 0            → L_pose_rank

Affinity
 └── 负对数亲和力                   → L_affinity
```

索引文件的具体列需要登录后核对。文献里常见的是 PDB code、分辨率、发布年份、
`-logKd/Ki`、参考配体，亲和力是否带终点类型（Kd 还是 Ki）不确定，下载后确认。

下载：登录后从 [pdbbind.org.cn/download.php](http://www.pdbbind.org.cn/download.php)
取 "Index files of PDBbind" 和 "Protein-ligand complexes: The refined set"。
备用入口 [PDBbind+](https://www.pdbbind-plus.org.cn/)。

### UniProt

蛋白质序列与功能注释的权威数据库。这里只用它的位点注释：binding site、
active site、functional region、跨膜区，作为功能残基预测的监督（`L_function`）。

不下载全库。等 BindingDB 和 BioLiP2 清洗出实际用到的蛋白 accession 列表后，
按 accession 批量查询即可。这些注释只作为预测目标，绝不作为模型输入。

一条记录是一个蛋白。

```text
Protein
 ├── UniProt accession
 ├── sequence
 └── 位点注释
      ├── binding site
      ├── active site
      ├── functional region
      └── 跨膜区                → L_function

没有配体，没有亲和力，没有结构。
```

查询入口：[uniprot.org](https://www.uniprot.org/)

### SIFTS

PDB 与 UniProt 之间的残基级映射，由 PDBe 维护，每周更新。它不是训练数据，
而是清洗 BioLiP2 和 PDBbind 的共同依赖：把结构里的 PDB 残基编号翻译成
UniProt 序列编号，否则三个来源的残基对不齐。

一条记录是一条 PDB 链到 UniProt 的一段对应。

```text
PDB
 ├── PDB ID
 ├── chain
 └── 残基编号区间（作者编号、SEQRES 编号）

        ↓

UniProt
 ├── accession
 └── 残基编号区间

没有配体，没有亲和力，没有标签。
```

下载：[pdb_chain_uniprot.csv.gz](https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/csv/pdb_chain_uniprot.csv.gz)（6 MB，[说明](https://www.ebi.ac.uk/pdbe/docs/sifts/quick.html)）




## 后补：完整版本才需要

这些来源只喂辅助损失或只做评测，数据量大或获取麻烦，第一版训练不依赖它们。

### ChEMBL

欧洲生物信息研究所维护的药物生物活性数据库，记录标准化过的化合物、靶点、
assay 和活性值，规模比 BindingDB 更大。

用途与 BindingDB 相同，扩充 DTI 和亲和力的样本量，另外它的 mechanism 注释
可以监督 `L_mechanism`。取 `standard_type` 为 Kd、Ki、IC50、EC50 且
`standard_relation` 为等号的记录，`pchembl_value` 直接作为亲和力。

一条活性记录是一次实验测量，和 BindingDB 同形。

```text
Protein
 └── UniProt accession（经 target 表关联）

        +

Ligand
 └── SMILES（经 compound_structures 表关联）

        ↓

Affinity label
 ├── Kd
 ├── Ki
 ├── IC50
 └── EC50                → L_DTI、L_affinity

Mechanism（另一张表，drug_mechanism）
 └── 作用类型              → L_mechanism
```

`activities` 表的字段已核对：`standard_type`、`standard_relation`、
`pchembl_value`。`drug_mechanism` 表的字段这次没核到原文，下载后确认。

下载：[chembl_36_sqlite.tar.gz](https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_36_sqlite.tar.gz)（4.7 GB）

### Guide to Pharmacology

IUPHAR/BPS 维护的药理学数据库，记录配体与靶点的作用类型，如 agonist、
antagonist、inhibitor、blocker、opener。数据量远小于 BindingDB。

监督作用方向（`L_effect`）和作用机制（`L_mechanism`）。原始分类到模型枚举的
映射放 `configs/label_maps/`，映射不上的记为缺失，不记为 other。

一条记录是一个配体对一个靶点的作用。

```text
Protein
 ├── target id
 └── UniProt accession

        +

Ligand
 ├── ligand id
 └── PubChem SID（没有 SMILES）

        ↓

Pharmacology
 ├── type                作用类别
 ├── action              agonist、antagonist、inhibitor 等
 └── affinity            多数为空

没有结构，没有 pocket。
```

`type` 和 `action` 监督 `L_effect`、`L_mechanism`。亲和力只是附带，
主数据仍以 BindingDB 和 ChEMBL 为准。

下载：[interactions.csv](https://www.guidetopharmacology.org/DATA/interactions.csv)，字段说明见 [file_descriptions.txt](https://www.guidetopharmacology.org/DATA/file_descriptions.txt)，[下载页](https://www.guidetopharmacology.org/download.jsp)

### ASD

上海交通大学维护的变构数据库，收录变构位点、变构配体和相关结构。

只监督变构 pocket 的辅助损失（`L_allosteric`），不进入主任务的采样权重。
变构位点经 SIFTS 或序列比对映射到 UniProt 残基。

一条记录是一个蛋白的一个变构位点。

```text
Protein
 ├── UniProt ID
 └── PDB ID

        +

Modulator
 └── 名字（不是 SMILES）

        ↓

Allosteric site
 └── 残基列表（Chain A: ALA123, GLY124, ...）  → L_allosteric
```

残基列表只在 XML 版里完整。2023 版的表格文件里这一列有约一半是空的，
所以要下 `ASD_Release_*_XF.tar.gz`，不要下表格版。

下载：[download.jsp](https://mdl.shsmu.edu.cn/ASD/module/download/download.jsp)，可能需要先申请访问。

### CrossDocked2020

把配体对接到非共晶受体上生成的 pose 数据集，每条 pose 带有相对晶体结构的
RMSD。规模约 50 GB。

只监督 pose 排序（`L_pose_rank`），near-native 的 RMSD 阈值见配置。只保留
能经 SIFTS 映射到 UniProt 的受体。第一版可只用 PDBbind 的晶体 pose 代替。

一条记录是一个对接出来的 pose。

```text
Protein
 └── receptor（PDB code + chain，PDB 格式）

        +

Ligand
 └── pose（SDF，对接或最小化得到的坐标）

        ↓

Pose label
 ├── RMSD to crystal
 └── label（RMSD ≤ 2Å 为 1，否则 0）   → L_pose_rank

Affinity
 └── 只有一个 -log 值，不区分 Kd/Ki/IC50，不使用
```

文件按 pocket 分目录。训练用的是 types 文件，每行格式为
`label  pK  RMSD  receptor  ligand  # Vina score`。

下载：[CrossDocked2020_v1.3_types.tgz](http://bits.csb.pitt.edu/files/crossdock2020/CrossDocked2020_v1.3_types.tgz)（3.5 GB，只有标签和文件名），原始结构 [CrossDocked2020_v1.3.tgz](http://bits.csb.pitt.edu/files/crossdock2020/CrossDocked2020_v1.3.tgz)（50 GB）。[说明](https://github.com/gnina/models/blob/master/data/CrossDocked2020/README.md)

### CASF

PDBbind 核心集上的比较评估基准，是对接和打分函数的标准测试集。

只做 pose ranking 的评测，不进训练。

```text
Protein
 └── PDB code（core set 里的复合物）

        +

Ligand
 └── 晶体 pose

        ↓

评测
 └── 给对接程序打分用，不产生训练标签
```

随 PDBbind 的 core set 一起获得，没有单独的下载。

### BioSNAP ChG-Miner

斯坦福 SNAP 整理的药物-基因相互作用网络，约 1.5 万条边，是 DTI 论文常用的
基准数据集。

不进主训练集。原样保留其原始划分，放在 `data/benchmarks/biosnap/`，只用于
和既有方法在同一划分上对齐比较。

一条记录是一条相互作用边，没有任何属性。

```text
Drug
 └── DrugBank ID（不是 SMILES）

        +

Gene
 └── UniProt accession

        ↓

Label
 └── 有边即为相互作用，无亲和力、无作用类型
```

只有两列，`Drug` 和 `Gene`，约 1.5 万条边。要用它做对照，还得另从
DrugBank 补 SMILES。

下载：[ChG-Miner_miner-chem-gene.tsv.gz](https://snap.stanford.edu/biodata/datasets/10002/files/ChG-Miner_miner-chem-gene.tsv.gz)（[说明页](https://snap.stanford.edu/biodata/datasets/10002/10002-ChG-Miner.html)）

## 不下载、训练时现算的部分

配体的三维构象由 RDKit 从 SMILES 生成，不依赖任何外部结构库。

没有实验结构的蛋白，其三维结构在清洗阶段按 UniProt accession 从
[AlphaFold DB](https://alphafold.ebi.ac.uk/) 拉取，pLDDT 低于阈值的残基在
`residue_mask` 里标为无效，不整条丢弃。

候选 pocket 由 apo 结构或预测结构上的 pocket detector 生成，作为模型输入。
实验复合物里的 native pocket 只作训练和评估的标签，推理时绝不使用测试配体的
坐标来定义 pocket。
