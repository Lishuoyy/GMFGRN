![Python Versions](https://img.shields.io/badge/python-3.8+-brightgreen.svg)

# GMFGRN: a matrix factorization and graph neural network approach for gene regulatory network inference

# Abstract

The recent advances of single-cell RNA sequencing (scRNA-seq) have enabled reliable profiling of gene expression at the
single-cell level, providing opportunities for accurate inference of gene regulatory networks (GRNs) on scRNA-seq data.
Most methods for inferring GRNs suffer from the inability to eliminate transitive interactions or necessitate expensive
computational resources. To address these, we present a novel method, termed GMFGRN, for accurate graph neural network (GNN)-based GRN inference from scRNA-seq data. GMFGRN employs GNN for matrix factorization and learns representative
embeddings for genes. For transcription factor–gene pairs, it utilizes the learned embeddings to determine whether they
interact with each other. The extensive suite of benchmarking experiments encompassing eight static scRNA-seq datasets
alongside several state-of-the-art methods demonstrated mean improvements of 1.9 and 2.5% over the runner-up in area
under the receiver operating characteristic curve (AUROC) and area under the precision–recall curve (AUPRC). In
addition, across four time-series datasets, maximum enhancements of 2.4 and 1.3% in AUROC and AUPRC were observed in
comparison to the runner-up. Moreover, GMFGRN requires significantly less training time and memory consumption, with
time and memory consumed <10% compared to the second-best method. These findings underscore the substantial potential of
GMFGRN in the inference of GRNs.

![model.png](https://github.com/Lishuoyy/GMFGRN/blob/main/model.png)

# Requirement:

```console
pip install dgl=1.0.1
pip install torch=1.12.1
```

# Datasets

The data for evaluating GMFGRN and data related to the experiment in the manuscript is
in https://zenodo.org/record/8418696.

# How to run

## Self-supervised train

### Parameters

- data_name: You can customize the name of the dataset. Embedding after supervised learning will be saved in the
  Embedding/data_name/ folder.
- data_path: The file of the gene expression profile. Can be h5 or csv file, the format please refer the example data.
- class_rating: The maximum expression level of gene expression matrix. Non-values in the matrix will be mapped to 1 to
  class_rating.
- gcn_out_units: The embedding size of genes and cells.
- gcn_agg_units: GCN aggregation unit size, whose value is equal to class_rating * gcn_out_units.
- train_max_iter: Number of iterations of training.
- train_lr: Learning rate.
- is_time: Whether it is a time series datasets.
- is_h5: If data_path specifies a h5 format file, then it should be true.

```bash
# static datasets
python3 self_supervised.py --data_name=mHSC_E --data_path ../data_evaluation/single_cell_type/mHSC-E/ExpressionData.csv  --class_rating 15 --gcn_out_units 256 --gcn_agg_units 3840 --train_max_iter 20000 --train_lr 0.01

# time series datasets
python3 self_supervised.py --data_name=hESC --data_path ../data_evaluation/Time_data/scRNA_expression_data/hesc1_expression_data/  --class_rating 25 --gcn_out_units 256 --gcn_agg_units 6400 --train_max_iter 20000 --train_lr 0.01 --is_time

```

**For boneMarrow, dendritic, mESC(1), mHSC(E), mHSC(GM) of static datasets, and mESC1 and mESC2 datasets of time series
datasets, class_rating is set to 15. Other datasets are set to 25.**

## Supervised train

### Parameters

- data_name: To be consistent with the data_name parameters of self-supervised learning, it is convenient to find the
  embedding of gene and cell learned by self-supervised learning.
- rpkm_path: The file of the gene expression profile. Can be h5 or csv file, the format please refer the example data.
- label_path: The file of the training gene pairs and their labels.
- divide_path: File that indicate the position in label_path to divide pairs into different TFs.
- gene_list_path: The file to map the name of gene in rpkm_path to the label_path.
- TF_num: To generate representation for this number of TFs. Should be a integer that equal or samller than the number
  of TFs in the label_path.
- TF_random: If the TF_num samller than the number of TFs in the label_path, we need to indicate TF_random, if
  TF_random=True, then the code will generate representation for randomly selected TF_num TFs.
- is_time: Whether it is a time series datasets.
- is_h5: If data_path specifies a h5 format file, then it should be true.

```bash
# static datasets
python3 supervised.py --data_name=bonemarrow --rpkm_path ../data_evaluation/bonemarrow/bone_marrow_cell.h5  --label_path ../data_evaluation/bonemarrow/gold_standard_for_TFdivide --divide_path ../data_evaluation/bonemarrow/whole_gold_split_pos --TF_num 13 --gene_list_path ../data_evaluation/bonemarrow/sc_gene_list.txt --is_h5

python3 supervised.py --data_name=dendritic --rpkm_path ../data_evaluation/dendritic/dendritic_cell.h5  --label_path ../data_evaluation/dendritic/gold_standard_dendritic_whole.txt --divide_path ../data_evaluation/dendritic/dendritic_divideTF_pos.txt --TF_num 16 --gene_list_path ../data_evaluation/dendritic/sc_gene_list.txt --is_h5

python3 supervised.py --data_name=mESC_1 --rpkm_path ../data_evaluation/mesc/mesc_cell.h5  --label_path ../data_evaluation/mesc/gold_standard_mesc_whole.txt --divide_path ../data_evaluation/mesc/mesc_divideTF_pos.txt --TF_num 38 --gene_list_path ../data_evaluation/mesc/mesc_sc_gene_list.txt --is_h5


python3 supervised.py --data_name=mHSC_E --rpkm_path ../data_evaluation/single_cell_type/mHSC-E/ExpressionData.csv  --label_path ../data_evaluation/single_cell_type/training_pairsmHSC_E.txt --divide_path ../data_evaluation/single_cell_type/training_pairsmHSC_E.txtTF_divide_pos.txt --TF_num 18 --gene_list_path ../data_evaluation/single_cell_type/mHSC_E_geneName_map.txt --TF_random

python3 supervised.py --data_name=mHSC_GM --rpkm_path ../data_evaluation/single_cell_type/mHSC-GM/ExpressionData.csv  --label_path ../data_evaluation/single_cell_type/training_pairsmHSC_GM.txt --divide_path ../data_evaluation/single_cell_type/training_pairsmHSC_GM.txtTF_divide_pos.txt --TF_num 18 --gene_list_path ../data_evaluation/single_cell_type/mHSC_GM_geneName_map.txt --TF_random

python3 supervised.py --data_name=mHSC_L --rpkm_path ../data_evaluation/single_cell_type/mHSC-L/ExpressionData.csv  --label_path ../data_evaluation/single_cell_type/training_pairsmHSC_L.txt --divide_path ../data_evaluation/single_cell_type/training_pairsmHSC_L.txtTF_divide_pos.txt --TF_num 18 --gene_list_path ../data_evaluation/single_cell_type/mHSC_L_geneName_map.txt --TF_random

python3 supervised.py --data_name=hESC --rpkm_path ../data_evaluation/single_cell_type/hESC/ExpressionData.csv  --label_path ../data_evaluation/single_cell_type/training_pairshESC.txt --divide_path ../data_evaluation/single_cell_type/training_pairshESC.txtTF_divide_pos.txt --TF_num 18 --gene_list_path ../data_evaluation/single_cell_type/hESC_geneName_map.txt --TF_random

python3 supervised.py --data_name=mESC_2 --rpkm_path ../data_evaluation/single_cell_type/mESC/ExpressionData.csv  --label_path ../data_evaluation/single_cell_type/training_pairsmESC.txt --divide_path ../data_evaluation/single_cell_type/training_pairsmESC.txtTF_divide_pos.txt --TF_num 18 --gene_list_path ../data_evaluation/single_cell_type/mESC_geneName_map.txt --TF_random

# time series datasets
python3 supervised.py --data_name=hESC1 --rpkm_path ../data_evaluation/Time_data/scRNA_expression_data/hesc1_expression_data/  --label_path ../data_evaluation/Time_data/DB_pairs_TF_gene/hesc1_gene_pairs_400.txt --divide_path ../data_evaluation/Time_data/DB_pairs_TF_gene/hesc1_gene_pairs_400_num.txt --TF_num 36 --gene_list_path ../data_evaluation/Time_data/DB_pairs_TF_gene/hesc1_gene_list_ref.txt --is_time

python3 supervised.py --data_name=hESC2 --rpkm_path ../data_evaluation/Time_data/scRNA_expression_data/hesc2_expression_data/  --label_path ../data_evaluation/Time_data/DB_pairs_TF_gene/hesc2_gene_pairs_400.txt --divide_path ../data_evaluation/Time_data/DB_pairs_TF_gene/hesc2_gene_pairs_400_num.txt --TF_num 36 --gene_list_path ../data_evaluation/Time_data/DB_pairs_TF_gene/hesc2_gene_list_ref.txt --is_time

python3 supervised.py --data_name=mESC1 --rpkm_path ../data_evaluation/Time_data/scRNA_expression_data/mesc1_expression_data/  --label_path ../data_evaluation/Time_data/DB_pairs_TF_gene/mesc1_gene_pairs_400.txt --divide_path ../data_evaluation/Time_data/DB_pairs_TF_gene/mesc1_gene_pairs_400_num.txt --TF_num 36 --gene_list_path ../data_evaluation/Time_data/DB_pairs_TF_gene/mesc1_gene_list_ref.txt --is_time

python3 supervised.py --data_name=mESC2 --rpkm_path ../data_evaluation/Time_data/scRNA_expression_data/mesc2_expression_data/  --label_path ../data_evaluation/Time_data/DB_pairs_TF_gene/mesc2_gene_pairs_400.txt --divide_path ../data_evaluation/Time_data/DB_pairs_TF_gene/mesc2_gene_pairs_400_num.txt --TF_num 38 --gene_list_path ../data_evaluation/Time_data/DB_pairs_TF_gene/mesc2_gene_list_ref.txt --is_time

```

# Contact

Please contact us if you have any questions: lishuocyy@njust.edu.cn
