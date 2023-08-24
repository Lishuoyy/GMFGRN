# from copy import deepcopy, copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
# from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, precision_recall_curve, auc
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, linear_kernel

import random
import time
from torch.autograd import Variable
import collections
from down_task_model import *
# import dgl
import os
# from train_on_GNN import *
from losses import SupConLoss, sup_constrive
# from scipy.linalg import svd

# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import networkx as nx
from utils import drnl_node_labeling
from torch.utils.data import DataLoader, Dataset
from dgl.dataloading import GraphDataLoader
from test import Transformer
# from utils2 import *
# from torchvision.transforms import transforms
# import lightgbm as lgb


def set_seed(seed_num):
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)
    # np.random.seed(seed_num)
    random.seed(seed_num)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed_num)


# seed_num = 3407
set_seed(114514)  # 27


class GraphDataSet(Dataset):
    """
    GraphDataset for torch DataLoader
    """

    def __init__(self, graph_list, tensor):
        self.graph_list = graph_list
        self.tensor = tensor

    def __len__(self):
        return len(self.graph_list)

    def __getitem__(self, index):
        return (self.graph_list[index], self.tensor[index])


class GeneData:
    def __init__(self, rpkm_path, label_path, divide_path, TF_num, gene_emb_path, cell_emb_path, istime, ish5=False,
                 gene_list_path=None, data_name=None, TF_random=False, save=False):
        self.gene_cell_src = None
        self.gene_cell_dst = None
        self.istime = istime
        self.data_name = data_name
        self.TF_random = TF_random
        self.save = save

        if not istime:
            if not ish5:
                self.df = pd.read_csv(rpkm_path, header='infer', index_col=0)
            else:
                self.df = pd.read_hdf(rpkm_path, key='/RPKMs').T

        else:
            time_h5 = []
            files = os.listdir(rpkm_path)
            for i in range(len(files)-(len(files)-4)):
                if self.data_name == 'mesc1':
                    time_pd = pd.read_hdf(rpkm_path + 'RPM_' + str(i) + '.h5', key='/RPKM')
                else:
                    time_pd = pd.read_hdf(rpkm_path + 'RPKM_' + str(i) + '.h5', key='/RPKMs')
                # print(time_pd)
                # exit()
                time_h5.append(time_pd)
            train_data = pd.concat(time_h5, axis=0, ignore_index=True)
            self.df = train_data.T
            # print(self.df)
            # exit()
            # self.df.columns = str(range(len(self.df.columns)))
        # print(self.df)
        # exit()
        self.origin_data = self.df.values
        # self.p, s, self.q = torch.svd_lowrank(torch.tensor(self.origin_data), q=256)
        # print(p.shape)
        # print(s.shape)
        # print(q.shape)
        # exit()
        # print(type(self.df.columns))
        # exit()
        # upper
        print(self.df)

        self.df.columns = self.df.columns.astype(str)
        self.df.index = self.df.index.astype(str)
        self.df.columns = self.df.columns.str.upper()
        self.df.index = self.df.index.str.upper()
        self.cell_to_idx = dict(zip(self.df.columns.astype(str), range(len(self.df.columns))))
        self.gene_to_idx = dict(zip(self.df.index.astype(str), range(len(self.df.index))))

        print(self.gene_to_idx)
        # exit()
        self.gene_to_name = {}
        if gene_list_path:
            gene_list = pd.read_csv(gene_list_path, header=None, sep='\s+')
            # upper
            print(gene_list)
            # exit()
            # gene_list.columns = gene_list.columns.astype(str)
            # gene_list.index = gene_list.index.astype(str)
            gene_list[0] = gene_list[0].astype(str)
            gene_list[1] = gene_list[1].astype(str)
            gene_list[0] = gene_list[0].str.upper()
            gene_list[1] = gene_list[1].str.upper()
            # print(gene_list)
            # exit()
            self.gene_to_name = dict(zip(gene_list[0].astype(str), gene_list[1].astype(str)))

            # print(self.gene_to_name)
            # exit()

        self.start_index = []
        self.end_index = []
        self.gene_emb = np.load(gene_emb_path)
        self.cell_emb = np.load(cell_emb_path)
        self.all_emb = np.concatenate((self.gene_emb, self.cell_emb), axis=0)

        # self.gene_gene_src, self.gene_gene_dst = self.getGeneOrCellGraph(self.gene_emb)
        # self.cell_cell_src, self.cell_cell_dst = self.getGeneOrCellGraph(self.cell_emb)
        # exit()

        self.key_list = []
        self.gold_standard = {}
        self.datas = []
        self.gene_key_datas = []
        self.h_datas = []
        self.cell_datas = []
        self.labels = []
        self.idx = []

        self.geneHaveCell = collections.defaultdict(list)
        self.node_src = []
        self.node_dst = []

        self.getStartEndIndex(divide_path)
        self.getLabel(label_path)
        self.getGeneCell(self.df)
        # x = self.origin_data
        # self.gene_emb_distance = np.corrcoef(self.gene_emb)  # euclidean_distances(self.gene_emb)
        # self.gene_emb_distance1 = np.cov(self.origin_data)
        # 对角线填充为0
        # np.fill_diagonal(self.gene_emb_distance, 0)
        # np.fill_diagonal(self.gene_emb_distance1, 0)
        # print(self.gene_emb_distance.shape)
        # exit()
        self.getTrainTest(TF_num)

        # nodes = self.df.shape[0] + self.df.shape[1]
        #
        # g = dgl.graph((self.gene_cell_src + self.gene_cell_dst, self.gene_cell_dst + self.gene_cell_src),
        #               num_nodes=nodes)
        # # g = dgl.add_self_loop(g)
        # embs = np.concatenate((self.gene_emb, self.cell_emb[:-1]), axis=0)
        # g.ndata['x'] = torch.FloatTensor(embs)
        # # print(g)
        # self.g = g
        # self.ndata = {k: v for k, v in self.g.ndata.items()}
        # self.edata = {k: v for k, v in self.g.edata.items()}
        # self.g.ndata.clear()
        # self.g.edata.clear()
        # print(self.g)
        # exit()

        print(len(self.key_list))
        # print(self.gold_standard.keys())
        print(len(self.gold_standard.keys()))
        # print('this', self.gold_standard['junb,stat6'])
        # exit()
        # self.getTrainTestSubGraph(TF_num)

    def getStartEndIndex(self, divide_path):
        tmp = []
        with open(divide_path, 'r') as f:
            for line in f:
                line = line.strip().split()
                tmp.append(int(line[0]))
        self.start_index = tmp[:-1]
        self.end_index = tmp[1:]
        # print(self.start_index)
        # print(self.end_index)

    def getLabel(self, label_path):
        s = open(label_path, 'r')
        for line in s:
            line1 = line
            line = line.split()
            # if int(line[2]) != 2:
            gene1 = line[0]
            gene2 = line[1]
            label = line[2]
            # print(gene1+','+gene2+',')
            key = str(gene1) + "," + str(gene2)
            # print(key)
            # if key == 'junb,elf4':
            #     print('this', line)
            #     print(line1)
            #     exit()
            if key not in self.gold_standard.keys():
                self.gold_standard[key] = int(label)
                self.key_list.append(key)
            else:
                # print(key, self.gold_standard[key])
                # print(key, label)
                # print('=' * 60)
                if label == 2:
                    # print(label)
                    pass
                else:
                    self.gold_standard[key] = int(label)
                self.key_list.append(key)
        s.close()
        print(len(self.key_list))
        print(len(self.gold_standard.keys()))
        # exit()

    def getTrainTest(self, TF_num):
        # random.seed(42)

        TF_order = list(range(0, len(self.start_index)))
        if self.TF_random:
            np.random.seed(42)
            np.random.shuffle(TF_order)
        # print(len(TF_order))
        # print("TF_order", TF_order)
        # random.shuffle(TF_order)
        # print("TF_order", TF_order)
        # exit()
        # TF_order = list(TF_order)
        # TF_order = [13, 45, 47, 44, 17, 27, 26, 25, 31, 19, 12, 4, 34,
        #             8, 3, 6, 40, 41, 46, 15, 9, 16, 24, 33, 30, 0, 43, 32, 5,
        #             29, 11, 36, 1, 21, 2, 37, 35, 23, 39, 10, 22, 18, 48, 20, 7, 42, 14, 28, 38]  # mHSC-E
        # TF_order = [11, 33, 10, 32, 34, 29, 12, 5, 25, 30, 27, 4, 9, 19, 3, 24, 20, 22, 16, 37, 31, 26, 0, 36, 13, 18,
        #             2, 35, 28, 23, 21, 6, 8, 14, 15, 17, 1, 7] # mesc2
        # TF_order = [55, 40, 19, 31, 115, 56, 69, 105, 81, 26, 95, 27, 64, 4, 97, 100, 36, 80, 93, 84, 18, 10, 120, 11,
        #             126, 45, 70, 78, 47, 0, 12, 42, 51, 24, 67, 113, 89, 15, 77, 22, 44, 65, 96, 30, 62, 124, 9, 33, 66,
        #             25, 117, 28, 98, 128, 53, 5, 68, 73, 39, 49, 35, 16, 125, 34, 85, 7, 43, 76, 104, 110, 8, 13, 94, 3,
        #             17, 38, 72, 101, 6, 83, 112, 114, 54, 50, 119, 46, 108, 61, 127, 79, 122, 91, 41, 58, 90, 48, 88,
        #             111, 57, 75, 32, 123, 59, 63, 109, 37, 29, 107, 116, 1, 52, 21, 2, 23, 103, 99, 87, 118, 74, 86, 82,
        #             121, 129, 20, 60, 71, 106, 14, 92, 102] # hesc
        # TF_order = [216, 66, 9, 192, 15, 198, 25, 194, 154, 126, 213, 19, 96, 177, 222, 137, 146, 30, 224, 184, 108,
        #             162, 203, 180, 125, 18, 147, 101, 104, 84, 196, 60, 118, 45, 16, 127, 159, 119, 82, 144, 93, 168,
        #             143, 73, 113, 112, 150, 55, 165, 69, 167, 109, 124, 79, 86, 95, 212, 117, 38, 24, 67, 185, 197, 10,
        #             120, 29, 153, 68, 75, 5, 56, 114, 139, 186, 65, 142, 193, 136, 31, 12, 35, 28, 42, 226, 115, 155,
        #             51, 132, 182, 76, 41, 97, 140, 78, 135, 26, 218, 171, 158, 0, 2, 77, 46, 100, 111, 138, 164, 90, 85,
        #             161, 152, 98, 36, 181, 61, 22, 208, 183, 33, 11, 223, 227, 6, 27, 141, 219, 220, 156, 4, 122, 32,
        #             163, 62, 128, 205, 172, 70, 175, 64, 44, 148, 40, 123, 23, 170, 178, 81, 39, 190, 47, 94, 173, 43,
        #             145, 204, 3, 105, 53, 133, 206, 176, 211, 49, 80, 34, 7, 110, 91, 83, 215, 207, 89, 8, 13, 59, 195,
        #             131, 17, 166, 72, 199, 134, 201, 209, 63, 54, 107, 50, 174, 189, 225, 200, 169, 58, 48, 88, 21, 57,
        #             160, 221, 187, 191, 129, 37, 157, 217, 1, 52, 149, 130, 151, 103, 99, 116, 87, 202, 74, 214, 210,
        #             121, 228, 20, 188, 71, 106, 14, 92, 179, 102] # mesc_2
        print("TF_order", TF_order)
        index_start_list = np.asarray(self.start_index)
        index_end_list = np.asarray(self.end_index)
        index_start_list = index_start_list[TF_order]
        index_end_list = index_end_list[TF_order]
        # print("index_start_list", index_start_list)
        # print("index_end_list", index_end_list)
        # exit()
        # f = open('test.txt', 'w')
        print(index_start_list)
        print(index_end_list)
        s = open(self.data_name + '_representation/gene_pairs.txt', 'w')
        ss = open(self.data_name + '_representation/divide_pos.txt', 'w')
        pos_len = 0
        ss.write(str(0) + '\n')
        for i in range(TF_num):
            name = self.data_name + '_representation/'
            # if os.path.exists(self.data_name + '_representation/' + str(i) + '_xdata.npy'):
            if self.save:
                x_data = np.load(name + str(i) + '_xdata.npy')
                h_data = np.load(name + str(i) + '_hdata.npy')
                y_data = np.load(name + str(i) + '_ydata.npy')
                gene_key_data = np.load(name + str(i) + '_gene_key_data.npy')
                # node_src = np.load('mesc2/' + str(i) + '_node_src.npy')
                # node_dst = np.load('mesc2/' + str(i) + '_node_dst.npy')
                self.datas.append(x_data)
                self.labels.append(y_data)
                self.gene_key_datas.append(gene_key_data)
                self.h_datas.append(h_data)
                # self.node_src.append(node_src)
                # self.node_dst.append(node_dst)
                continue

            start_idx = index_start_list[i]
            end_idx = index_end_list[i]
            # print(self.key_list[start_idx:end_idx])
            # exit()
            print(i)
            print(start_idx, end_idx)
            # continue

            this_datas = []
            this_key_datas = []
            this_h_datas = []
            this_cell_datas = []
            this_labels = []

            # print(len(self.key_list[start_idx:end_idx]))
            c = 0

            for line in self.key_list[start_idx:end_idx]:
                # print(self.key_lis[start_idx:end_idx])

                label = self.gold_standard[line]
                gene1, gene2 = line.split(',')
                gene1 = gene1.upper()
                gene2 = gene2.upper()
                if int(label) != 3:
                    this_key_datas.append([gene1.lower(), gene2.lower(), label])
                    s.write(gene1 + '\t' + gene2 + '\t' + str(label) + '\n')
                    if not self.gene_to_name:
                        gene1_idx = self.gene_to_idx[gene1]
                        gene2_idx = self.gene_to_idx[gene2]
                    else:
                        gene1_index = self.gene_to_name[gene1]
                        gene2_index = self.gene_to_name[gene2]
                        gene1_idx = self.gene_to_idx[gene1_index]
                        gene2_idx = self.gene_to_idx[gene2_index]

                    # graph
                    # this_node_src.append(gene1_idx)
                    # this_node_dst.append(gene2_idx)

                    gene1_emb = self.gene_emb[gene1_idx]  # + gene1_cells_emb
                    gene2_emb = self.gene_emb[gene2_idx]  # + gene2_cells_emb

                    gene1_emb = np.expand_dims(gene1_emb, axis=0)
                    gene2_emb = np.expand_dims(gene2_emb, axis=0)

                    # gene1_gene2_all_emb = []
                    # gene1_gene2_cat = np.concatenate((gene1_emb, gene2_emb), axis=1)
                    # gene1_gene2_all_emb.append(gene1_gene2_cat)
                    # gene1_gene1_cat = np.concatenate((gene1_emb, gene1_emb), axis=1)
                    # gene1_gene2_all_emb.append(gene1_gene1_cat)
                    # gene2_gene2_cat = np.concatenate((gene2_emb, gene2_emb), axis=1)
                    # gene1_gene2_all_emb.append(gene2_gene2_cat)

                    # gene1_neighbour_idx = self.gene_emb_distance[gene1_idx]
                    # gene2_neighbour_idx = self.gene_emb_distance[gene2_idx]
                    # 从大到小排序，取top 10
                    # top = 256
                    # gene1_neighbour_idx = np.argsort(-gene1_neighbour_idx)[:top]
                    # gene2_neighbour_idx = np.argsort(-gene2_neighbour_idx)[:top]
                    # gene1_neighbour_emb = self.gene_emb[gene1_neighbour_idx]
                    # gene2_neighbour_emb = self.gene_emb[gene2_neighbour_idx]
                    # gene1_neighbour_emb = np.mean(gene1_neighbour_emb, axis=0)
                    # gene2_neighbour_emb = np.mean(gene2_neighbour_emb, axis=0)
                    # gene1_neighbour_emb = np.expand_dims(gene1_neighbour_emb, axis=0)
                    # gene2_neighbour_emb = np.expand_dims(gene2_neighbour_emb, axis=0)
                    #
                    # gene1_emb_tile = np.tile(gene1_emb, (top, 1))
                    # gene2_emb_tile = np.tile(gene2_emb, (top, 1))
                    # gene1_neighbour_cat = np.concatenate((gene1_emb_tile, gene1_neighbour_emb), axis=1)
                    # gene2_neighbour_cat = np.concatenate((gene2_neighbour_emb, gene2_emb_tile), axis=1)
                    # gene1_gene2_all_emb.append(gene1_neighbour_cat)
                    # gene1_gene2_all_emb.append(gene2_neighbour_cat)

                    # 另一个箱关系
                    # top = 5
                    # gene1_neighbour_idx = self.gene_emb_distance1[gene1_idx]
                    # gene2_neighbour_idx = self.gene_emb_distance1[gene2_idx]
                    # # 从大到小排序，取top 10
                    # gene1_neighbour_idx = np.argsort(-gene1_neighbour_idx)[:top]
                    # gene2_neighbour_idx = np.argsort(-gene2_neighbour_idx)[:top]
                    # gene1_neighbour_emb = self.gene_emb[gene1_neighbour_idx]
                    # gene2_neighbour_emb = self.gene_emb[gene2_neighbour_idx]
                    # gene1_emb_tile = np.tile(gene1_emb, (top, 1))
                    # gene2_emb_tile = np.tile(gene2_emb, (top, 1))
                    # gene1_neighbour_cat = np.concatenate((gene1_emb_tile, gene1_neighbour_emb), axis=1)
                    # gene2_neighbour_cat = np.concatenate((gene2_neighbour_emb, gene2_emb_tile), axis=1)
                    # gene1_gene2_all_emb.append(gene1_neighbour_cat)
                    # gene1_gene2_all_emb.append(gene2_neighbour_cat)

                    # gene1_gene2_all_emb = np.concatenate(gene1_gene2_all_emb, axis=0)
                    # print(gene1_gene2_all_emb.shape)
                    # exit()
                    gene1_rpkm = self.origin_data[gene1_idx]
                    gene2_rpkm = self.origin_data[gene2_idx]

                    # gene1_emb = self.origin_data[gene1_idx]
                    # gene2_emb = self.origin_data[gene2_idx]

                    gene1_cells = self.geneHaveCell[gene1_idx]
                    if len(gene1_cells) == 0:
                        gene1_cells_emb = np.zeros(256)
                    else:
                        gene1_cells_emb = self.cell_emb[gene1_cells]
                        gene1_cells_emb = np.mean(gene1_cells_emb, axis=0)
                    # print(gene1_cells_emb.shape)
                    # exit()

                    gene2_cells = self.geneHaveCell[gene2_idx]
                    if len(gene2_cells) == 0:
                        gene2_cells_emb = np.zeros(256)
                    else:
                        gene2_cells_emb = self.cell_emb[gene2_cells]
                        gene2_cells_emb = np.mean(gene2_cells_emb, axis=0)

                    # gene1_emb = np.expand_dims(gene1_emb, axis=0)
                    # gene2_emb = np.expand_dims(gene2_emb, axis=0)

                    # 计算gene1_emb与gene1_cells_emb的欧式距离
                    # distances1 = np.sqrt(
                    #     np.sum((gene1_emb[:, np.newaxis, :] - gene1_cells_emb[np.newaxis, :, :]) ** 2, axis=2))
                    # distances1 = np.squeeze(distances1)
                    # sorted_indices1 = np.argsort(distances1)[:10]
                    # gene1_cells_emb = gene1_cells_emb[sorted_indices1]
                    # gene1_cells_emb = np.mean(gene1_cells_emb, axis=0)
                    gene1_cells_emb = np.expand_dims(gene1_cells_emb, axis=0)
                    # print(gene1_cells_emb.shape)
                    # exit()
                    # if gene1_cells_emb.shape[0] < 10:
                    #     gene1_cells_emb = np.concatenate((gene1_cells_emb, np.zeros((10 - gene1_cells_emb.shape[0], 256))), axis=0)

                    # 计算gene2_emb与gene2_cells_emb的欧式距离
                    # distances2 = np.sqrt(
                    #     np.sum((gene2_emb[:, np.newaxis, :] - gene2_cells_emb[np.newaxis, :, :]) ** 2, axis=2))
                    # distances2 = np.squeeze(distances2)
                    # sorted_indices2 = np.argsort(distances2)[:10]
                    # gene2_cells_emb = gene2_cells_emb[sorted_indices2]
                    # gene2_cells_emb = np.mean(gene2_cells_emb, axis=0)
                    gene2_cells_emb = np.expand_dims(gene2_cells_emb, axis=0)
                    # if gene2_cells_emb.shape[0] < 10:
                    #     gene2_cells_emb = np.concatenate((gene2_cells_emb, np.zeros((10- gene2_cells_emb.shape[0], 256))), axis=0)

                    # print(gene1_cells_emb.shape)
                    # print(gene2_cells_emb.shape)
                    # exit()

                    # 二维直方图
                    x_tf = np.log10(gene1_rpkm + 10 ** -2)
                    x_gene = np.log10(gene2_rpkm + 10 ** -2)
                    h = np.histogram2d(x_tf, x_gene, bins=16)
                    h = h[0].T
                    h = (np.log10(h / len(x_tf) + 10 ** -4) + 4) / 4
                    h = np.expand_dims(h, axis=0)
                    # print(h.shape)
                    # exit()
                    # gene_emb = np.concatenate((gene1_emb, gene2_emb, gene1_cells_emb, gene2_cells_emb), axis=0)
                    # gene_emb = np.concatenate((gene1_gene2_cat,
                    #                            gene1_neighbour_cat, gene2_neighbour_cat), axis=0)
                    # gene1_emb = np.mean(np.concatenate((gene1_emb, gene1_cells_emb, gene1_neighbour_emb), axis=0), axis=0)
                    # gene1_emb = np.expand_dims(gene1_emb, axis=0)
                    # gene2_emb = np.mean(np.concatenate((gene2_emb, gene2_cells_emb, gene2_neighbour_emb), axis=0), axis=0)
                    # gene2_emb = np.expand_dims(gene2_emb, axis=0)
                    gene1_emb_1 = (gene1_emb + gene1_cells_emb)
                    gene2_emb_1 = (gene2_emb + gene2_cells_emb)
                    # print(gene1_emb.shape)
                    # exit()
                    # gene1_emb = np.squeeze(gene1_emb_1, axis=0)
                    # gene2_emb = np.squeeze(gene2_emb_1, axis=0)
                    # print(gene1_emb.shape)
                    # exit()
                    # print(gene1_emb.shape)
                    # print(gene2_emb.shape)
                    # print(gene1_cells_emb.shape)
                    # print(gene2_cells_emb.shape)
                    # exit()
                    gene_emb = np.concatenate((gene1_emb, gene2_emb, gene1_cells_emb, gene2_cells_emb), axis=0)
                    # gene_emb = np.squeeze(gene_emb, axis=0)
                    # gene_emb = np.concatenate((gene1_emb, gene2_emb, gene1_cells_emb, gene2_cells_emb,
                    #                            gene1_neighbour_emb, gene2_neighbour_emb), axis=0)

                    # cell_emb = np.concatenate((gene1_cells_emb, gene2_cells_emb), axis=0)
                    # this_datas.append(gene1_gene2_all_emb)
                    this_datas.append(gene_emb)
                    this_h_datas.append(h)
                    # this_datas.append([gene1_idx, gene2_idx])
                    # this_cell_datas.append(cell_emb)
                    this_labels.append(label)
                    # this_idx.append([gene1_idx, gene2_idx])
            pos_len += len(this_datas)
            ss.write(str(pos_len) + '\n')

            this_datas = np.asarray(this_datas)
            # print(this_datas.shape)
            # this_h_datas = np.load(f'../../contrastive-predictive-coding-master/my/mHSC_L_representation/{i}_xdata.npy')
            this_h_datas = np.asarray(this_h_datas)
            this_cell_datas = np.asarray(this_cell_datas)
            print(this_datas.shape, this_cell_datas.shape, this_h_datas[:, 0, :, :].shape)

            # exit()
            # this_h_datas = np.asarray(this_h_datas)
            this_labels = np.asarray(this_labels)
            this_key_datas = np.asarray(this_key_datas)
            # this_idx = np.asarray(this_idx)
            # this_node_src = np.asarray(this_node_src)
            # this_node_dst = np.asarray(this_node_dst)
            if self.save:
                if not os.path.exists(name):
                    os.mkdir(name)
                # np.save(name + str(i) + '_xdata.npy', this_datas)
                # np.save(name + str(i) + '_ydata.npy', this_labels)
                # np.save(name + str(i) + '_hdata.npy', this_h_datas)
                np.save(name + str(i) + '_gene_key_data.npy', this_key_datas)
                print(this_datas.shape, this_labels.shape, this_h_datas.shape)
                # np.save('mesc2/' + str(i) + '_node_src.npy', this_node_src)
                # np.save('mesc2/' + str(i) + '_node_dst.npy', this_node_dst)

            self.datas.append(this_datas)
            self.h_datas.append(this_h_datas)
            # self.cell_datas.append(this_cell_datas)
            self.labels.append(this_labels)
            self.gene_key_datas.append(this_key_datas)
            # self.idx.append(this_idx)
            # self.node_src.append(this_node_src)
            # self.node_dst.append(this_node_dst)
        s.close()
        ss.close()
        # print(len(self.datas))
        # print(len(self.labels))
        # exit()

    def getTrainTestSubGraph(self, TF_num):
        # random.seed(42)
        g = self.gene_g
        TF_order = list(range(0, len(self.start_index)))
        # print(len(TF_order))
        # print("TF_order", TF_order)
        # random.shuffle(TF_order)
        # print("TF_order", TF_order)
        # exit()
        # TF_order = list(TF_order)
        TF_order = [13, 45, 47, 44, 17, 27, 26, 25, 31, 19, 12, 4, 34,
                    8, 3, 6, 40, 41, 46, 15, 9, 16, 24, 33, 30, 0, 43, 32, 5,
                    29, 11, 36, 1, 21, 2, 37, 35, 23, 39, 10, 22, 18, 48, 20, 7, 42, 14, 28, 38]  # mHSC-E
        # TF_order = [11, 33, 10, 32, 34, 29, 12, 5, 25, 30, 27, 4, 9, 19, 3, 24, 20, 22, 16, 37, 31, 26, 0, 36, 13, 18,
        #             2, 35, 28, 23, 21, 6, 8, 14, 15, 17, 1, 7] # mesc2
        # print("TF_order", TF_order)
        index_start_list = np.asarray(self.start_index)
        index_end_list = np.asarray(self.end_index)
        index_start_list = index_start_list[TF_order]
        index_end_list = index_end_list[TF_order]
        # print("index_start_list", index_start_list)
        # print("index_end_list", index_end_list)
        # exit()
        # f = open('test.txt', 'w')
        print(index_start_list)
        print(index_end_list)
        for i in range(TF_num):
            if self.istime and os.path.exists('mesc2/'):
                x_data = np.load('mesc2/' + str(i) + '_xdata.npy')
                h_data = np.load('mesc2/' + str(i) + '_hdata.npy')
                y_data = np.load('mesc2/' + str(i) + '_ydata.npy')
                node_src = np.load('mesc2/' + str(i) + '_node_src.npy')
                node_dst = np.load('mesc2/' + str(i) + '_node_dst.npy')
                self.datas.append(x_data)
                self.labels.append(y_data)
                self.h_datas.append(h_data)
                self.node_src.append(node_src)
                self.node_dst.append(node_dst)
                continue

            start_idx = index_start_list[i]
            end_idx = index_end_list[i]
            # print(self.key_list[start_idx:end_idx])
            # exit()
            print(i)
            print(start_idx, end_idx)
            # continue

            this_datas = []
            this_labels = []
            c = 0

            for line in self.key_list[start_idx:end_idx]:
                # print(self.key_lis[start_idx:end_idx])

                label = self.gold_standard[line]
                gene1, gene2 = line.split(',')
                if int(label) != 2:
                    gene1_idx = self.gene_to_idx[gene1]
                    gene2_idx = self.gene_to_idx[gene2]
                    target_nodes = torch.tensor([gene1_idx, gene2_idx])
                    sample_nodes = [target_nodes]
                    frontiers = target_nodes
                    # print(frontiers)
                    for _ in range(1):
                        frontiers = g.out_edges(frontiers)[1]
                        frontiers = torch.unique(frontiers)
                        sample_nodes.append(frontiers)
                    sample_nodes = torch.cat(sample_nodes)
                    sample_nodes = torch.unique(sample_nodes)
                    subgraph = dgl.node_subgraph(g, sample_nodes)
                    # print(subgraph)
                    # continue
                    # Each node should have unique node id in the new subgraph
                    u_id = int(
                        torch.nonzero(
                            subgraph.ndata[dgl.NID] == int(target_nodes[0]), as_tuple=False
                        )
                    )
                    v_id = int(
                        torch.nonzero(
                            subgraph.ndata[dgl.NID] == int(target_nodes[1]), as_tuple=False
                        )
                    )
                    z = drnl_node_labeling(subgraph, u_id, v_id)
                    subgraph.ndata["z"] = z
                    # subgraph to networkx
                    # nx_g = subgraph.to_networkx()
                    # nx.draw(nx_g, with_labels=True)
                    # plt.show()

                    # print(sample_nodes)
                    # print(subgraph)
                    # continue
                    # exit()
                    this_datas.append(subgraph)
                    this_labels.append(label)

            this_datas = np.asarray(this_datas)
            this_labels = np.asarray(this_labels)

            self.datas.append(this_datas)
            self.labels.append(this_labels)

        # print(len(self.datas))
        # print(len(self.labels))
        # exit()

    def getGeneCell(self, df):
        print(df)
        cell_idx = df.shape[0]
        node_src = []
        node_dst = []
        x = []
        gene_cell_top_d = collections.defaultdict(list)
        # gene_gene_top_d = collections.defaultdict(list)
        for i in range(df.shape[0]):
            j_nonzero = np.nonzero(df.iloc[i, :].values)[0]
            if len(j_nonzero) == 0:
                continue
            # gene_emb = self.gene_emb[i]
            # cell_emb = self.cell_emb[j_nonzero]
            # print(gene_emb.shape)
            # print(cell_emb.shape)
            # pearson
            # corr = np.corrcoef(gene_emb, cell_emb)[0, 1:]
            # corr = np.abs(corr)
            # corr = np.linalg.norm(gene_emb - cell_emb, axis=1)
            # corr = cosine_similarity(gene_emb.reshape(1, -1), cell_emb)
            # corr = np.squeeze(corr)
            # print(corr.shape)
            # exit()

            # rpkm排序
            # corr = self.origin_data[i, :]
            # 获取前10个最相关的cell的索引
            # top_k = 600
            # top_k_idx = np.argsort(-corr)[:top_k]
            # print(j_nonzero)
            # print(top_k_idx)
            # print(j_nonzero[top_k_idx])
            # exit()
            # top_k_cell = top_k_idx
            # print(top_k_cell)
            # exit()
            self.geneHaveCell[i].extend(j_nonzero)
            # node_src.extend([i] * len(top_k_cell))
            # node_dst.extend(top_k_cell + cell_idx)
            # gene_cell_top_d[i].extend(j_nonzero[top_k_idx])
            # x.append(len(j_nonzero))
        # self.gene_cell_src = node_src
        # self.gene_cell_dst = node_dst
        # self.gene_cell_top_d = gene_cell_top_d
        # print(gene_cell_top_d)
        # exit()
        # print(len(node_src))
        # print((len(node_dst)))
        # print(x)
        # exit()

    def getGeneOrCellGraph(self, emb):
        # 欧式距离
        emb_distance = euclidean_distances(emb)  # cosine_similarity(self.gene_emb)
        # print(self.gene_emb_distance)
        # exit()
        # 对角线为0
        np.fill_diagonal(emb_distance, 0)

        def get_top_k(distances, k=10):
            # 获取每行排序后的索引
            sorted_indices = np.argsort(distances, axis=1)[:, ::-1]

            # 初始化全零数组
            top_k = np.zeros_like(distances)

            # 将每行前 k 个最大值的索引设置为 1
            for i in range(distances.shape[0]):
                top_k[i][sorted_indices[i, :k]] = 1

            return top_k

        # top 10% edge
        # self.gene_emb_distance = np.where(self.gene_emb_distance > np.percentile(self.gene_emb_distance, 99.9), 1, 0)
        emb_distance = get_top_k(emb_distance, k=10)
        # print(self.gene_emb_distance)
        # exit()
        src, dst = np.where(emb_distance == 1)
        src = list(src)
        dst = list(dst)
        return src, dst


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred).to(device)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


def mixup_celoss(x, y):
    eps = 1e-6
    return -torch.mean(y * torch.log(x + eps) + (1 - y) * torch.log(1 - x + eps))


# e = GeneData('../data_evaluation/single_cell_type/mHSC-E/ExpressionData.csv',
#              '../data_evaluation/single_cell_type/training_pairsmHSC_E.txt',
#              '../data_evaluation/single_cell_type/training_pairsmHSC_E.txtTF_divide_pos.txt',
#              TF_num=18, gene_emb_path='out/mHSC_E_all_users_w19.npy', cell_emb_path='out/mHSC_E_all_items_w19.npy')

'''
mHSC_E:
'../data_evaluation/single_cell_type/mHSC-E/ExpressionData.csv',
                 '../data_evaluation/single_cell_type/training_pairsmHSC_E.txt',
                 '../data_evaluation/single_cell_type/training_pairsmHSC_E.txtTF_divide_pos.txt',
mesc2:
'../data_evaluation/Time_data/scRNA_expression_data/mesc2_expression_data/',
                 '../data_evaluation/Time_data/database/mesc2_gene_pairs_400.txt',
                 '../data_evaluation/Time_data/database/mesc2_gene_pairs_400_num.txt',
'''
device = "cuda:1"


def main():
    # e = GeneData('../data_evaluation/Time_data/scRNA_expression_data/mesc2_expression_data/',
    #              '../data_evaluation/Time_data/database/mesc2_gene_pairs_400.txt',
    #              '../data_evaluation/Time_data/database/mesc2_gene_pairs_400_num.txt',
    #              TF_num=38, gene_emb_path='out/mesc2_all_users_w4.npy', cell_emb_path='out/mesc2_all_items_w4.npy',
    #              istime=False)
    TF_num = 38
    data_name = 'mesc2'
    drim_path = '../modelResult/TimeData/myModel_3/' + data_name + '_result_2/'
    save_dir = '../modelResult/TimeData/myModel_3/'
    # e = GeneData('../data_evaluation/single_cell_type/mHSC-L/ExpressionData.csv',
    #              '../data_evaluation/single_cell_type/training_pairs' + data_name + '.txt',
    #              '../data_evaluation/single_cell_type/training_pairs' + data_name + '.txtTF_divide_pos.txt',
    #              TF_num=18, gene_emb_path='../../gcmc/' + data_name + '_emb/user_out_vfinal_2.npy',
    #              cell_emb_path='../../gcmc/' + data_name + '_emb/movie_out_vfinal_2.npy',
    #              istime=False, gene_list_path='../data_evaluation/single_cell_type/' + data_name + '_geneName_map.txt',
    #              data_name=data_name, TF_random=True)
    # e = GeneData('../data_evaluation/' + data_name + '/bone_marrow_cell.h5',
    #              '../data_evaluation/' + data_name + '/gold_standard_for_TFdivide',
    #              '../data_evaluation/' + data_name + '/whole_gold_split_pos',
    #              TF_num=TF_num, gene_emb_path='../../gcmc/' + data_name + '_emb/user_out_vfinal_1.npy',
    #              cell_emb_path='../../gcmc/' + data_name + '_emb/movie_out_vfinal_1.npy',
    #              istime=False, gene_list_path='../data_evaluation/' + data_name + '/sc_gene_list.txt',
    #              ish5=True, data_name=data_name)


    # time data
    e = GeneData('../data_evaluation/Time_data/scRNA_expression_data/' + data_name + '_expression_data/',
                 '../data_evaluation/Time_data/DB_pairs_TF_gene/' + data_name + '_gene_pairs_400.txt',
                 '../data_evaluation/Time_data/DB_pairs_TF_gene/' + data_name + '_gene_pairs_400_num.txt',
                 TF_num=TF_num, gene_emb_path='../../gcmc/timeData/' + data_name + '_emb/user_out_v1.npy',
                 cell_emb_path='../../gcmc/timeData/' + data_name + '_emb/movie_out_v1.npy',
                 istime=True, gene_list_path='../data_evaluation/Time_data/DB_pairs_TF_gene/' + data_name + '_gene_list_ref.txt',
                 ish5=True, data_name=data_name)
    exit()
    # cross_file_path = '../data_evaluation/Time_data/DB_pairs_TF_gene/' + data_name + '_cross_validation_fold_divide.txt'
    # cross_index = []
    # with open(cross_file_path, 'r') as f:
    #     for line in f:
    #         cross_index.append([int(i) for i in line.strip().split(',')])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # e = GeneData('../data_evaluation/single_cell_type/mHSC-E/ExpressionData.csv',
    #              '../data_evaluation/single_cell_type/training_pairsmHSC_E.txt',
    #              '../data_evaluation/single_cell_type/training_pairsmHSC_E.txtTF_divide_pos.txt',
    #              TF_num=18, gene_emb_path='out/mHSC_E_all_users_w4.npy', cell_emb_path='out/mHSC_E_all_items_w4.npy',
    #              istime=False)

    # e = GeneData('../data_evaluation/bonemarrow/bone_marrow_cell.h5',
    #              '../data_evaluation/bonemarrow/gold_standard_for_TFdivide',
    #              '../data_evaluation/bonemarrow/whole_gold_split_pos',
    #              TF_num=13, gene_emb_path='../../gcmc/bonemarrow_emb/user_out_v1.npy',
    #              cell_emb_path='../../gcmc/bonemarrow_emb/movie_out_v1.npy',
    #              istime=False, ish5=True,
    #              gene_list_path='../data_evaluation/bonemarrow/sc_gene_list.txt',
    #              )
    # three-fold cross validation
    acc_all = []
    auc_all = []
    ap_all = []
    pre_all = []
    label_all = []
    y_test_predict = []
    y_test_true = []
    z_all = []
    epochs = 200
    start_time = time.time()

    # kf = KFold(n_splits=3)
    # test_index_set = []
    # train_index_set = []
    # for fold, (train_index, test_index) in enumerate(kf.split(e.datas)):
    #     train_index_set.append(train_index)
    #     test_index_set.append(test_index)
    #
    # print("train_index_set", train_index_set)
    # print("test_index_set", test_index_set)
    # exit()
    # test_indel = [[4, 6, 9, 3, 1], [0, 2, 7, 12], [10, 5, 8, 11]]

    for fold in range(1, 4):
        count_set = [0]
        count_setx = 0
        test_index = fold
        test_TF = [i for i in range(int(np.ceil((test_index - 1) * 0.333333 * TF_num)),
                                    int(np.ceil(test_index * 0.333333 * TF_num)))]
        # test_TF = cross_index[fold - 1]
        fold_path = save_dir + str(test_TF) + '/'
        if not os.path.exists(fold_path):
            os.makedirs(fold_path)
        # test_TF = random.sample(list(range(TF_num)), round(TF_num / 3))
        # test_TF = test_indel[fold - 1]
        train_TF = [j for j in range(TF_num) if j not in test_TF]
        print("test_TF", test_TF)

        train_emb_datas = []
        train_h_datas = []
        # train_cell_datas = []
        train_labels = []
        # train_idx = []
        # train_node_src = []
        # train_node_dst = []
        for j in train_TF:
            print(j, e.datas[j].shape, e.labels[j].shape)
            train_emb_datas.append(e.datas[j])
            train_h_datas.append(e.h_datas[j])
            # train_cell_datas.append(e.cell_datas[j])
            train_labels.append(e.labels[j])
            # train_node_src.append(e.node_src[j])
            # train_node_dst.append(e.node_dst[j])
            # train_idx.append(e.idx[j])
        train_emb_datas = np.concatenate(train_emb_datas, axis=0)
        train_h_datas = np.concatenate(train_h_datas, axis=0)
        # train_cell_datas = np.concatenate(train_cell_datas, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)
        # train_node_src = np.concatenate(train_node_src, axis=0)
        # train_node_dst = np.concatenate(train_node_dst, axis=0)

        # print(train_node_src)
        # print(train_node_dst)

        train_emb_datas, val_emb_datas, train_h_datas, val_h_datas, train_labels, val_labels = \
            train_test_split(train_emb_datas,
                             train_h_datas,
                             train_labels,
                             test_size=0.2,
                             random_state=42)
        # train_emb_datas = np.concatenate([train_emb_datas, train_cell_datas], axis=0)
        # train_labels = np.concatenate([train_labels, train_labels], axis=0)
        # train_h_datas = np.concatenate([train_h_datas, train_h_datas], axis=0)
        test_emb_datas = []
        test_h_datas = []
        test_labels = []
        # test_node_src = []
        # test_node_dst = []
        z = [0]
        z_len = 0
        for j in test_TF:
            test_emb_datas.append(e.datas[j])
            test_h_datas.append(e.h_datas[j])
            test_labels.append(e.labels[j])
            # test_node_src.append(e.node_src[j])
            # test_node_dst.append(e.node_dst[j])
            z_len += len(e.datas[j])
            z.append(z_len)
        np.save(fold_path + 'z.npy', z)
        z_all.append(z)
        test_emb_datas = np.concatenate(test_emb_datas, axis=0)
        test_h_datas = np.concatenate(test_h_datas, axis=0)
        test_labels = np.concatenate(test_labels, axis=0)
        # test_node_src = np.concatenate(test_node_src, axis=0)
        # test_node_dst = np.concatenate(test_node_dst, axis=0)

        print('train', train_emb_datas.shape, train_h_datas.shape, train_labels.shape)
        print('val', val_emb_datas.shape, val_h_datas.shape, val_labels.shape)
        print('test', test_emb_datas.shape, test_h_datas.shape, test_labels.shape)
        print('=' * 100)

        # train normal
        # continue
        # print(train_node_src)
        # exit()
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(train_emb_datas).float(),
                                           torch.from_numpy(train_h_datas).float(),
                                           torch.from_numpy(train_labels).float()),
            batch_size=512, shuffle=True, num_workers=14, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(val_emb_datas).float(),
                                           torch.from_numpy(val_h_datas).float(),
                                           torch.from_numpy(val_labels).float()),
            batch_size=512, shuffle=False, num_workers=14)

        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(test_emb_datas).float(),
                                           torch.from_numpy(test_h_datas).float(),
                                           torch.from_numpy(test_labels).float()),
            batch_size=512, shuffle=False, num_workers=14)

        # transform = transforms.Compose([
        #     AddNoise(mean=0, std=0.1, p=0.5)
        # ])

        # val_loader = test_loader
        # train_loader = test_loader
        # train postive graph
        # train_pos_u = []
        # train_pos_v = []
        # train_neg_u = []
        # train_neg_v = []
        # for i in range(len(train_node_src)):
        #     if train_labels[i] == 1:
        #         train_pos_u.append(train_node_src[i])
        #         train_pos_v.append(train_node_dst[i])
        #     else:
        #         train_neg_u.append(train_node_src[i])
        #         train_neg_v.append(train_node_dst[i])

        # model = LinearModel().to(device)
        # model.m.load_state_dict(torch.load('con_linear.pth'))
        # for param in model.m.parameters():
        #     param.requires_grad = False
        model = LinearNet().to(device)
        # model = BestModel().to(device)
        # con_path = 'mHSC_E_con/con_linear.pth'
        # model.neighbor_model.load_state_dict(torch.load(con_path))
        # for param in model.neighbor_model.parameters():
        #     param.requires_grad = False
        # model = MMoE(feature_dim=1024, expert_dim=1024, n_expert=10, n_task=1, use_gate=True).to(device)
        # model = GCN(in_feats=1701, hidden_feats=256, out_feats=256).to(device)
        test_model = LinearNet().to(device)
        # test_model = BestModel().to(device)
        # test_model.neighbor_model.load_state_dict(torch.load(con_path))
        # for param in test_model.neighbor_model.parameters():
        #     param.requires_grad = True
        # test_model = MMoE(feature_dim=1024, expert_dim=1024, n_expert=10 , n_task=1, use_gate=True).to(device)
        # test_model = GCN(in_feats=1701, hidden_feats=256, out_feats=256).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6, verbose=True)
        # optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
        # criterion = nn.BCELoss()
        # criterion = LabelSmoothingLoss(classes=2, smoothing=0.3)
        # supConLoss = SupConLoss(temperature=0.07)
        # criterion = FocalLoss(gamma=2, alpha=0.75)
        # criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
        # nn.BCEWithLogitsLoss
        criterion = nn.BCEWithLogitsLoss()
        val_acc_best = 0.0
        model_weight_best = None
        early_stop = 0
        stop_num = 10
        for epoch in range(epochs):
            model.train()
            train_acc_sum = 0.0
            train_loss = 0.0
            train_ce_loss = 0.0
            train_suploss = 0.0
            for batch_idx, (emb_data, h_data, target) in enumerate(train_loader):
                emb_data = emb_data.to(device)
                # emb_data = transform(emb_data)
                h_data = h_data.to(device)
                target = target.to(device)

                optimizer.zero_grad()
                output = model(emb_data, h_data)
                output = output.squeeze()

                # SupConLoss
                # suploss = sup_constrive(hidden_emb, target, 0.07)

                # output = model(e.g.to(device), data[:, 0].long(), data[:, 1].long())
                # mmoe
                # loss1 = criterion(output[0], target)
                # loss2 = criterion(output[1], target)
                # loss = loss1 + loss2
                ce_loss = criterion(output, target)
                loss = ce_loss
                loss.backward()
                # nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()

                train_loss += loss.item()
                train_ce_loss += ce_loss.item()
                # train_suploss += suploss
                pred = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output))
                # pred = torch.argmax(output, dim=1)
                # target = torch.argmax(target, dim=1)
                # pred1 = torch.argmax(output[0], target)
                # pred2 = torch.argmax(output[1], target)
                # pred = pred1 + pred2
                # train_acc_sum1 = (pred1 == target).sum().item()
                # train_acc_sum2 =
                #

                train_acc_sum += (pred == target).sum().item()
            # scheduler.step()
            train_acc = train_acc_sum / len(train_loader.dataset)
            train_loss = train_loss / len(train_loader)
            train_ce_loss = train_ce_loss / len(train_loader)
            train_suploss = train_suploss / len(train_loader)

            model.eval()
            val_acc_sum = 0.0
            val_loss = 0.0
            pre = []
            label = []
            with torch.no_grad():
                for batch_idx, (emb_data, h_data, target) in enumerate(val_loader):
                    emb_data = emb_data.to(device)
                    h_data = h_data.to(device)
                    target = target.to(device).squeeze()

                    output = model(emb_data, h_data)
                    output = output.squeeze()
                    # output = model(e.g.to(device), data[:, 0].long(), data[:, 1].long())
                    loss = criterion(output, target)

                    val_loss += loss.item()
                    pred = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output))
                    # pred = torch.argmax(output, dim=1)
                    # target = torch.argmax(target, dim=1)
                    # pred1 = torch.argmax(output[0], target)
                    # pred2 = torch.argmax(output[1], target)
                    # pred = pred1 + pred2

                    pre.extend(output.cpu().numpy())
                    label.extend(target.cpu().numpy())

                    val_acc_sum += (pred == target).sum().item()

            val_acc = val_acc_sum / len(val_loader.dataset)
            val_loss = val_loss / len(val_loader)

            # one_hot_label = torch.eye(2)[label, :]
            one_hot_label = label
            # print(pre)
            # print(np.isnan(pre).any())
            # exit()
            val_auc = roc_auc_score(one_hot_label, pre)
            # val_ap = average_precision_score(one_hot_label, pre)
            precision, recall, thresholds = precision_recall_curve(one_hot_label, pre, pos_label=1)
            val_ap = auc(recall, precision)
            if val_acc > val_acc_best:
                model_weight_best = model.state_dict()
                val_acc_best = val_acc
                early_stop = 0
            else:
                early_stop += 1

            print('Epoch: {}, Train Loss: {:.4f},Train CELoss: {:.4f},Train SupLoss: {:.4f}, Train Acc: {:.4f}, '
                  'Val Loss: {:.4f}, Val Acc: {:.4f}, Val AUC: {:.4f}, Val AP: {:.4f}'.format(
                epoch, train_loss, train_ce_loss, train_suploss, train_acc, val_loss, val_acc, val_auc, val_ap
            ))
            if early_stop > stop_num:
                break

        # test
        # save model weight
        torch.save(model_weight_best, fold_path + 'model_weight_best.pth')
        test_model.load_state_dict(model_weight_best)
        test_model.eval()
        test_acc_sum = 0.0
        test_loss = 0.0
        pre = []
        origin_pre = []
        label = []
        # model.load_state_dict(model_weight_best)
        with torch.no_grad():
            for batch_idx, (emb_data, h_data, target) in enumerate(test_loader):
                emb_data = emb_data.to(device)
                h_data = h_data.to(device)
                target = target.to(device).squeeze()

                output = test_model(emb_data, h_data)
                output = output.squeeze()
                # origin_pre.extend(output.cpu().numpy())
                # output = model(e.g.to(device), data[:, 0].long(), data[:, 1].long())
                loss = criterion(output, target)

                test_loss += loss.item()
                pred = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output))
                # pred = torch.argmax(output, dim=1)
                # target = torch.argmax(target, dim=1)
                # pred1 = torch.argmax(output[0], target)
                # pred2 = torch.argmax(output[1], target)
                # pred = pred1 + pred2
                pre.extend(output.cpu().numpy())
                # pre.extend(pred.cpu().numpy())
                label.extend(target.cpu().numpy())

                test_acc_sum += (pred == target).sum().item()
        test_acc = test_acc_sum / len(test_loader.dataset)
        test_loss = test_loss / len(test_loader)

        # one_hot_label = torch.eye(2)[label, :]
        one_hot_label = label
        test_auc = roc_auc_score(one_hot_label, pre)
        precision, recall, thresholds = precision_recall_curve(one_hot_label, pre, pos_label=1)
        test_ap = auc(recall, precision)
        # test_ap = average_precision_score(one_hot_label, pre)
        pre_all.extend(pre)
        label_all.extend(label)
        y_test_predict.append(pre)
        y_test_true.append(label)
        acc_all.append(test_acc)
        auc_all.append(test_auc)
        ap_all.append(test_ap)

        np.save(fold_path + 'end_y_predict.npy', np.asarray(pre))
        np.save(fold_path + 'end_y_test.npy', np.asarray(label))
        print('Test Loss: {:.4f}, Test Acc: {:.4f}, Test AUC: {:.4f}, Test AP: {:.4f}'.format(
            test_loss, test_acc, test_auc, test_ap
        ))
    # np.save('z_all.npy', np.asarray(z_all))
    # exit()
    # one_hot_label_all = torch.eye(2)[label_all, :]
    one_hot_label_all = label_all
    final_auc = roc_auc_score(one_hot_label_all, pre_all)
    # final_ap = average_precision_score(one_hot_label_all, pre_all)
    precision, recall, thresholds = precision_recall_curve(one_hot_label_all, pre_all, pos_label=1)
    final_ap = auc(recall, precision)
    pre_all = np.where(np.asarray(pre_all) > 0.5, 1, 0)
    # np.savetxt('pre_all.csv', pre_all, delimiter=',')
    # final_acc = accuracy_score(np.argmax(one_hot_label_all, axis=1), np.argmax(pre_all, axis=1))
    final_acc = accuracy_score(label_all, pre_all)
    auc_index_all = []
    ap_index_all = []
    test_acc_all = []
    test_auc_all = []
    test_ap_all = []
    for fold in range(len(acc_all)):
        print('=' * 50)
        print('Fold {} Test Acc: {:.4f}, Test AUC: {:.4f}, Test AP: {:.4f}'.format(fold, acc_all[fold], auc_all[fold],
                                                                                   ap_all[fold]))
        test_predict = y_test_predict[fold]
        test_true = y_test_true[fold]
        z = z_all[fold]
        for i in range(len(z) - 1):
            test_predict_i = test_predict[z[i]:z[i + 1]]
            test_true_i = test_true[z[i]:z[i + 1]]
            # test_true_i = torch.eye(2)[test_true_i, :]
            test_auc_i = roc_auc_score(test_true_i, test_predict_i)
            # test_ap_i = average_precision_score(test_true_i, test_predict_i)
            precision, recall, thresholds = precision_recall_curve(test_true_i, test_predict_i, pos_label=1)
            test_ap_i = auc(recall, precision)
            test_predict_i = np.where(np.asarray(test_predict_i) > 0.5, 1, 0)
            # test_acc_i = accuracy_score(np.argmax(test_true_i, axis=1), np.argmax(test_predict_i, axis=1))
            test_acc_i = accuracy_score(test_true_i, test_predict_i)
            print('\tindex {} Test Acc: {:.4f}, Test AUC: {:.4f}, Test AP: {:.4f}'.format(i, test_acc_i, test_auc_i,
                                                                                          test_ap_i))
            auc_index_all.append(test_auc_i)
            ap_index_all.append(test_ap_i)
            test_acc_all.append(test_acc_i)
            test_auc_all.append(test_auc_i)
            test_ap_all.append(test_ap_i)
    print('Final ACC: {:.4f}, AUC: {:.4f}, AP:{:.4f}'.format(final_acc, final_auc, final_ap))
    print('Test Acc: {:.4f}, Test AUC: {:.4f}, Test AP: {:.4f}'.format(np.mean(test_acc_all), np.mean(test_auc_all),
                                                                       np.mean(test_ap_all)))
    print('Cost Time: {:.4f} s'.format(time.time() - start_time))
    # AUROC箱线图
    drim_auc = np.load(drim_path + 'AUROC_set.npy')
    drim_auc = np.expand_dims(drim_auc, axis=1)
    auc_index_all = np.asarray(auc_index_all)
    auc_index_all = np.expand_dims(auc_index_all, axis=1)
    all_auc = np.concatenate((drim_auc, auc_index_all), axis=1)
    # all_auc = all_auc.T
    # print(all_auc.shape)
    positions = [0, 0.16]
    plt.boxplot(all_auc, labels=['DRIM', 'My'], positions=positions)
    # plt.boxplot(auc_index_all, labels=['My'], positions=positions)

    plt.yticks(np.arange(0.1, 1, 0.1))
    plt.title('AUROC')
    plt.show()
    # AP箱线图
    drim_ap = np.load(drim_path + 'AUPRC_set.npy')
    drim_ap = np.expand_dims(drim_ap, axis=1)
    ap_index_all = np.asarray(ap_index_all)
    ap_index_all = np.expand_dims(ap_index_all, axis=1)
    all_ap = np.concatenate((drim_ap, ap_index_all), axis=1)
    # all_auc = all_auc.T
    print(all_ap.shape)
    plt.boxplot(all_ap, labels=['DRIM', 'My'], positions=positions)
    # plt.boxplot(ap_index_all, labels=['My'], positions=positions)
    plt.yticks(np.arange(0.1, 1, 0.1))
    plt.title('AUPRC')
    plt.show()


def main_label3():
    # e = GeneData('../data_evaluation/Time_data/scRNA_expression_data/mesc2_expression_data/',
    #              '../data_evaluation/Time_data/database/mesc2_gene_pairs_400.txt',
    #              '../data_evaluation/Time_data/database/mesc2_gene_pairs_400_num.txt',
    #              TF_num=38, gene_emb_path='out/mesc2_all_users_w4.npy', cell_emb_path='out/mesc2_all_items_w4.npy',
    #              istime=False)
    TF_num = 18
    data_name = 'mHSC_GM'
    drim_path = '../../DeepDRIM-main/DeepDRIM/' + data_name + '_result_label3/'
    # '/home/shilei/lishuo/DeepDRIM-main/DeepDRIM/mHSC_L_result_label3/AUROC_set.npy'
    save_dir = '../modelResult/myModel_label3/' + data_name + '_result/'
    e = GeneData('../data_evaluation/single_cell_type/mHSC-GM/ExpressionData.csv',
                 '../data_evaluation/single_cell_type/training_pairs' + data_name + '.txt',
                 '../data_evaluation/single_cell_type/training_pairs' + data_name + '.txtTF_divide_pos.txt',
                 TF_num=18, gene_emb_path='../../gcmc/' + data_name + '_emb/user_out_vfinal_1.npy',
                 cell_emb_path='../../gcmc/' + data_name + '_emb/movie_out_vfinal_1.npy',
                 istime=False, gene_list_path='../data_evaluation/single_cell_type/' + data_name + '_geneName_map.txt',
                 data_name=data_name, TF_random=True)
    # e = GeneData('../data_evaluation/' + data_name + '/bone_marrow_cell.h5',
    #              '../data_evaluation/' + data_name + '/gold_standard_for_TFdivide',
    #              '../data_evaluation/' + data_name + '/whole_gold_split_pos',
    #              TF_num=TF_num, gene_emb_path='../../gcmc/' + data_name + '_emb/user_out_vfinal_1.npy',
    #              cell_emb_path='../../gcmc/' + data_name + '_emb/movie_out_vfinal_1.npy',
    #              istime=False, gene_list_path='../data_evaluation/' + data_name + '/sc_gene_list.txt',
    #              ish5=True, data_name=data_name)

    # time data
    # e = GeneData('../data_evaluation/Time_data/scRNA_expression_data/' + data_name + '_expression_data/',
    #              '../data_evaluation/Time_data/DB_pairs_TF_gene/' + data_name + '_gene_pairs_400.txt',
    #              '../data_evaluation/Time_data/DB_pairs_TF_gene/' + data_name + '_gene_pairs_400_num.txt',
    #              TF_num=TF_num, gene_emb_path='../../gcmc/timeData/' + data_name + '_emb/user_out_vfinal_4.npy',
    #              cell_emb_path='../../gcmc/timeData/' + data_name + '_emb/movie_out_vfinal_4.npy',
    #              istime=True, gene_list_path='../data_evaluation/Time_data/DB_pairs_TF_gene/' + data_name + '_gene_list_ref.txt',
    #              ish5=True, data_name=data_name)
    # cross_file_path = '../data_evaluation/Time_data/DB_pairs_TF_gene/' + data_name + '_cross_validation_fold_divide.txt'
    # cross_index = []
    # with open(cross_file_path, 'r') as f:
    #     for line in f:
    #         cross_index.append([int(i) for i in line.strip().split(',')])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # e = GeneData('../data_evaluation/single_cell_type/mHSC-E/ExpressionData.csv',
    #              '../data_evaluation/single_cell_type/training_pairsmHSC_E.txt',
    #              '../data_evaluation/single_cell_type/training_pairsmHSC_E.txtTF_divide_pos.txt',
    #              TF_num=18, gene_emb_path='out/mHSC_E_all_users_w4.npy', cell_emb_path='out/mHSC_E_all_items_w4.npy',
    #              istime=False)

    # e = GeneData('../data_evaluation/bonemarrow/bone_marrow_cell.h5',
    #              '../data_evaluation/bonemarrow/gold_standard_for_TFdivide',
    #              '../data_evaluation/bonemarrow/whole_gold_split_pos',
    #              TF_num=13, gene_emb_path='../../gcmc/bonemarrow_emb/user_out_v1.npy',
    #              cell_emb_path='../../gcmc/bonemarrow_emb/movie_out_v1.npy',
    #              istime=False, ish5=True,
    #              gene_list_path='../data_evaluation/bonemarrow/sc_gene_list.txt',
    #              )
    # three-fold cross validation
    acc_all = []
    auc_all = []
    ap_all = []
    pre_all = []
    label_all = []
    y_test_predict = []
    y_test_true = []
    z_all = []
    epochs = 200
    start_time = time.time()

    # kf = KFold(n_splits=3)
    # test_index_set = []
    # train_index_set = []
    # for fold, (train_index, test_index) in enumerate(kf.split(e.datas)):
    #     train_index_set.append(train_index)
    #     test_index_set.append(test_index)
    #
    # print("train_index_set", train_index_set)
    # print("test_index_set", test_index_set)
    # exit()
    # test_indel = [[4, 6, 9, 3, 1], [0, 2, 7, 12], [10, 5, 8, 11]]

    for fold in range(1, 4):
        count_set = [0]
        count_setx = 0
        test_index = fold
        test_TF = [i for i in range(int(np.ceil((test_index - 1) * 0.333333 * TF_num)),
                                    int(np.ceil(test_index * 0.333333 * TF_num)))]
        # test_TF = cross_index[fold - 1]
        fold_path = save_dir + str(test_TF) + '/'
        if not os.path.exists(fold_path):
            os.makedirs(fold_path)
        # test_TF = random.sample(list(range(TF_num)), round(TF_num / 3))
        # test_TF = test_indel[fold - 1]
        train_TF = [j for j in range(TF_num) if j not in test_TF]
        print("test_TF", test_TF)

        train_emb_datas = []
        train_h_datas = []
        # train_cell_datas = []
        train_labels = []
        # train_idx = []
        # train_node_src = []
        # train_node_dst = []
        for j in train_TF:
            print(j, e.datas[j].shape, e.labels[j].shape)
            train_emb_datas.append(e.datas[j])
            train_h_datas.append(e.h_datas[j])
            # train_cell_datas.append(e.cell_datas[j])
            train_labels.append(e.labels[j])
            # train_node_src.append(e.node_src[j])
            # train_node_dst.append(e.node_dst[j])
            # train_idx.append(e.idx[j])
        train_emb_datas = np.concatenate(train_emb_datas, axis=0)
        train_h_datas = np.concatenate(train_h_datas, axis=0)
        # train_cell_datas = np.concatenate(train_cell_datas, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)
        # train_node_src = np.concatenate(train_node_src, axis=0)
        # train_node_dst = np.concatenate(train_node_dst, axis=0)

        # print(train_node_src)
        # print(train_node_dst)

        train_emb_datas, val_emb_datas, train_h_datas, val_h_datas, train_labels, val_labels = \
            train_test_split(train_emb_datas,
                             train_h_datas,
                             train_labels,
                             test_size=0.2,
                             random_state=42)
        # train_emb_datas = np.concatenate([train_emb_datas, train_cell_datas], axis=0)
        # train_labels = np.concatenate([train_labels, train_labels], axis=0)
        # train_h_datas = np.concatenate([train_h_datas, train_h_datas], axis=0)
        test_emb_datas = []
        test_h_datas = []
        test_labels = []
        # test_node_src = []
        # test_node_dst = []
        z = [0]
        z_len = 0
        for j in test_TF:
            test_emb_datas.append(e.datas[j])
            test_h_datas.append(e.h_datas[j])
            test_labels.append(e.labels[j])
            # test_node_src.append(e.node_src[j])
            # test_node_dst.append(e.node_dst[j])
            z_len += len(e.datas[j])
            z.append(z_len)
        np.save(fold_path + 'z.npy', z)
        z_all.append(z)
        test_emb_datas = np.concatenate(test_emb_datas, axis=0)
        test_h_datas = np.concatenate(test_h_datas, axis=0)
        test_labels = np.concatenate(test_labels, axis=0)
        # test_node_src = np.concatenate(test_node_src, axis=0)
        # test_node_dst = np.concatenate(test_node_dst, axis=0)

        print('train', train_emb_datas.shape, train_h_datas.shape, train_labels.shape)
        print('val', val_emb_datas.shape, val_h_datas.shape, val_labels.shape)
        print('test', test_emb_datas.shape, test_h_datas.shape, test_labels.shape)
        print('=' * 100)

        # train normal
        # continue
        # print(train_node_src)
        # exit()
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(train_emb_datas).float(),
                                           torch.from_numpy(train_h_datas).float(),
                                           torch.from_numpy(train_labels).long()),
            batch_size=512, shuffle=True, num_workers=14, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(val_emb_datas).float(),
                                           torch.from_numpy(val_h_datas).float(),
                                           torch.from_numpy(val_labels).long()),
            batch_size=512, shuffle=False, num_workers=14)

        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(test_emb_datas).float(),
                                           torch.from_numpy(test_h_datas).float(),
                                           torch.from_numpy(test_labels).long()),
            batch_size=512, shuffle=False, num_workers=14)

        # transform = transforms.Compose([
        #     AddNoise(mean=0, std=0.1, p=0.5)
        # ])

        # val_loader = test_loader
        # train_loader = test_loader
        # train postive graph
        # train_pos_u = []
        # train_pos_v = []
        # train_neg_u = []
        # train_neg_v = []
        # for i in range(len(train_node_src)):
        #     if train_labels[i] == 1:
        #         train_pos_u.append(train_node_src[i])
        #         train_pos_v.append(train_node_dst[i])
        #     else:
        #         train_neg_u.append(train_node_src[i])
        #         train_neg_v.append(train_node_dst[i])

        # model = LinearModel().to(device)
        # model.m.load_state_dict(torch.load('con_linear.pth'))
        # for param in model.m.parameters():
        #     param.requires_grad = False
        model = LinearNet_label3().to(device)
        # model = BestModel().to(device)
        # con_path = 'mHSC_E_con/con_linear.pth'
        # model.neighbor_model.load_state_dict(torch.load(con_path))
        # for param in model.neighbor_model.parameters():
        #     param.requires_grad = False
        # model = MMoE(feature_dim=1024, expert_dim=1024, n_expert=10, n_task=1, use_gate=True).to(device)
        # model = GCN(in_feats=1701, hidden_feats=256, out_feats=256).to(device)
        test_model = LinearNet_label3().to(device)
        # test_model = BestModel().to(device)
        # test_model.neighbor_model.load_state_dict(torch.load(con_path))
        # for param in test_model.neighbor_model.parameters():
        #     param.requires_grad = True
        # test_model = MMoE(feature_dim=1024, expert_dim=1024, n_expert=10 , n_task=1, use_gate=True).to(device)
        # test_model = GCN(in_feats=1701, hidden_feats=256, out_feats=256).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3 ,weight_decay=1e-6)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6, verbose=True)
        # optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
        criterion = nn.CrossEntropyLoss()
        # criterion = LabelSmoothingLoss(classes=2, smoothing=0.3)
        # supConLoss = SupConLoss(temperature=0.07)
        # criterion = FocalLoss(gamma=2, alpha=0.75)
        # criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
        # nn.BCEWithLogitsLoss
        # criterion = nn.BCEWithLogitsLoss()
        val_acc_best = 0.0
        model_weight_best = None
        early_stop = 0
        stop_num = 10
        for epoch in range(epochs):
            model.train()
            train_acc_sum = 0.0
            train_loss = 0.0
            train_ce_loss = 0.0
            train_suploss = 0.0
            for batch_idx, (emb_data, h_data, target) in enumerate(train_loader):
                emb_data = emb_data.to(device)
                # emb_data = transform(emb_data)
                h_data = h_data.to(device)
                target = target.to(device)

                optimizer.zero_grad()
                output = model(emb_data, h_data)
                output = output.squeeze()

                # SupConLoss
                # suploss = sup_constrive(hidden_emb, target, 0.07)

                # output = model(e.g.to(device), data[:, 0].long(), data[:, 1].long())
                # mmoe
                # loss1 = criterion(output[0], target)
                # loss2 = criterion(output[1], target)
                # loss = loss1 + loss2
                ce_loss = criterion(output, target)
                loss = ce_loss
                loss.backward()
                # nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()

                train_loss += loss.item()
                train_ce_loss += ce_loss.item()
                # train_suploss += suploss
                # pred = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output))
                pred = torch.argmax(output, dim=1)
                # target = torch.argmax(target, dim=1)
                # pred1 = torch.argmax(output[0], target)
                # pred2 = torch.argmax(output[1], target)
                # pred = pred1 + pred2
                # train_acc_sum1 = (pred1 == target).sum().item()
                # train_acc_sum2 =
                #

                train_acc_sum += (pred == target).sum().item()
            # scheduler.step()
            train_acc = train_acc_sum / len(train_loader.dataset)
            train_loss = train_loss / len(train_loader)
            train_ce_loss = train_ce_loss / len(train_loader)
            train_suploss = train_suploss / len(train_loader)

            model.eval()
            val_acc_sum = 0.0
            val_loss = 0.0
            pre = []
            label = []
            with torch.no_grad():
                for batch_idx, (emb_data, h_data, target) in enumerate(val_loader):
                    emb_data = emb_data.to(device)
                    h_data = h_data.to(device)
                    target = target.to(device).squeeze()

                    output = model(emb_data, h_data)
                    output = output.squeeze()
                    # output = model(e.g.to(device), data[:, 0].long(), data[:, 1].long())
                    loss = criterion(output, target)

                    val_loss += loss.item()
                    # pred = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output))
                    pred = torch.argmax(output, dim=1)
                    # target = torch.argmax(target, dim=1)
                    # pred1 = torch.argmax(output[0], target)
                    # pred2 = torch.argmax(output[1], target)
                    # pred = pred1 + pred2

                    pre.extend(output.cpu().numpy())
                    label.extend(target.cpu().numpy())

                    val_acc_sum += (pred == target).sum().item()

            val_acc = val_acc_sum / len(val_loader.dataset)
            val_loss = val_loss / len(val_loader)

            one_hot_label = torch.eye(3)[label, :]
            # one_hot_label = label
            # print(pre)
            # print(np.isnan(pre).any())
            # exit()
            val_auc = roc_auc_score(one_hot_label, pre)
            val_ap = average_precision_score(one_hot_label, pre)
            # precision, recall, thresholds = precision_recall_curve(one_hot_label, pre)
            # val_ap = auc(recall, precision)
            if val_acc > val_acc_best:
                model_weight_best = model.state_dict()
                val_acc_best = val_acc
                early_stop = 0
            else:
                early_stop += 1

            print('Epoch: {}, Train Loss: {:.4f},Train CELoss: {:.4f},Train SupLoss: {:.4f}, Train Acc: {:.4f}, '
                  'Val Loss: {:.4f}, Val Acc: {:.4f}, Val AUC: {:.4f}, Val AP: {:.4f}'.format(
                epoch, train_loss, train_ce_loss, train_suploss, train_acc, val_loss, val_acc, val_auc, val_ap
            ))
            if early_stop > stop_num:
                break

        # test
        # save model weight
        torch.save(model_weight_best, fold_path + 'model_weight_best.pth')
        test_model.load_state_dict(model_weight_best)
        test_model.eval()
        test_acc_sum = 0.0
        test_loss = 0.0
        pre = []
        origin_pre = []
        label = []
        # model.load_state_dict(model_weight_best)
        with torch.no_grad():
            for batch_idx, (emb_data, h_data, target) in enumerate(test_loader):
                emb_data = emb_data.to(device)
                h_data = h_data.to(device)
                target = target.to(device).squeeze()

                output = test_model(emb_data, h_data)
                output = output.squeeze()
                output = nn.Softmax(dim=1)(output)
                # origin_pre.extend(output.cpu().numpy())
                # output = model(e.g.to(device), data[:, 0].long(), data[:, 1].long())
                loss = criterion(output, target)

                test_loss += loss.item()
                # pred = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output))
                pred = torch.argmax(output, dim=1)
                # target = torch.argmax(target, dim=1)
                # pred1 = torch.argmax(output[0], target)
                # pred2 = torch.argmax(output[1], target)
                # pred = pred1 + pred2
                pre.extend(output.cpu().numpy())
                # pre.extend(pred.cpu().numpy())
                label.extend(target.cpu().numpy())

                test_acc_sum += (pred == target).sum().item()
        test_acc = test_acc_sum / len(test_loader.dataset)
        test_loss = test_loss / len(test_loader)

        one_hot_label = torch.eye(3)[label, :]
        # one_hot_label = label
        test_auc = roc_auc_score(one_hot_label, pre)
        # precision, recall, thresholds = precision_recall_curve(one_hot_label, pre)
        # test_ap = auc(recall, precision)
        test_ap = average_precision_score(one_hot_label, pre)
        pre_all.extend(pre)
        label_all.extend(label)
        y_test_predict.append(pre)
        y_test_true.append(label)
        acc_all.append(test_acc)
        auc_all.append(test_auc)
        ap_all.append(test_ap)

        np.save(fold_path + 'end_y_predict.npy', np.asarray(pre))
        np.save(fold_path + 'end_y_test.npy', np.asarray(label))
        print('Test Loss: {:.4f}, Test Acc: {:.4f}, Test AUC: {:.4f}, Test AP: {:.4f}'.format(
            test_loss, test_acc, test_auc, test_ap
        ))
    # np.save('z_all.npy', np.asarray(z_all))
    # exit()
    one_hot_label_all = torch.eye(3)[label_all, :]
    # one_hot_label_all = label_all
    final_auc = roc_auc_score(one_hot_label_all, pre_all)
    final_ap = average_precision_score(one_hot_label_all, pre_all)
    # precision, recall, thresholds = precision_recall_curve(one_hot_label_all, pre_all)
    # final_ap = auc(recall, precision)
    # pre_all = np.where(np.asarray(pre_all) > 0.5, 1, 0)
    pre_all = np.argmax(pre_all, axis=1)
    # np.savetxt('pre_all.csv', pre_all, delimiter=',')
    # final_acc = accuracy_score(np.argmax(one_hot_label_all, axis=1), np.argmax(pre_all, axis=1))
    final_acc = accuracy_score(label_all, pre_all)
    auc_index_all = []
    ap_index_all = []
    test_acc_all = []
    test_auc_all = []
    test_ap_all = []
    for fold in range(len(acc_all)):
        print('=' * 50)
        print('Fold {} Test Acc: {:.4f}, Test AUC: {:.4f}, Test AP: {:.4f}'.format(fold, acc_all[fold], auc_all[fold],
                                                                                   ap_all[fold]))
        test_predict = y_test_predict[fold]
        test_true = y_test_true[fold]
        z = z_all[fold]
        for i in range(len(z) - 1):
            test_predict_i = test_predict[z[i]:z[i + 1]]
            test_true_i = test_true[z[i]:z[i + 1]]
            test_true_i = torch.eye(3)[test_true_i, :]
            test_auc_i = roc_auc_score(test_true_i, test_predict_i)
            test_ap_i = average_precision_score(test_true_i, test_predict_i)
            # precision, recall, thresholds = precision_recall_curve(test_true_i, test_predict_i,)
            # test_ap_i = auc(recall, precision)
            # test_predict_i = np.where(np.asarray(test_predict_i) > 0.5, 1, 0)
            test_acc_i = accuracy_score(np.argmax(test_true_i, axis=1), np.argmax(test_predict_i, axis=1))
            # test_acc_i = accuracy_score(test_true_i, test_predict_i)
            print('\tindex {} Test Acc: {:.4f}, Test AUC: {:.4f}, Test AP: {:.4f}'.format(i, test_acc_i, test_auc_i,
                                                                                          test_ap_i))
            auc_index_all.append(test_auc_i)
            ap_index_all.append(test_ap_i)
            test_acc_all.append(test_acc_i)
            test_auc_all.append(test_auc_i)
            test_ap_all.append(test_ap_i)
    print('Final ACC: {:.4f}, AUC: {:.4f}, AP:{:.4f}'.format(final_acc, final_auc, final_ap))
    print(len(label_all), len(pre_all))
    print('Test Acc: {:.4f}, Test AUC: {:.4f}, Test AP: {:.4f}'.format(np.mean(test_acc_all), np.mean(test_auc_all),
                                                                       np.mean(test_ap_all)))
    print('Cost Time: {:.4f} s'.format(time.time() - start_time))
    # AUROC箱线图
    drim_auc = np.load(drim_path + 'AUROC_set.npy')
    drim_auc = np.expand_dims(drim_auc, axis=1)
    auc_index_all = np.asarray(auc_index_all)
    auc_index_all = np.expand_dims(auc_index_all, axis=1)
    all_auc = np.concatenate((drim_auc, auc_index_all), axis=1)
    # all_auc = all_auc.T
    # print(all_auc.shape)
    positions = [0, 0.16]
    plt.boxplot(all_auc, labels=['DRIM', 'My'], positions=positions)
    # plt.boxplot(auc_index_all, labels=['My'], positions=positions)

    plt.yticks(np.arange(0.1, 1, 0.1))
    plt.title('AUROC')
    plt.show()
    # # AP箱线图
    # drim_ap = np.load(drim_path + 'AUPRC_set.npy')
    # drim_ap = np.expand_dims(drim_ap, axis=1)
    # ap_index_all = np.asarray(ap_index_all)
    # ap_index_all = np.expand_dims(ap_index_all, axis=1)
    # all_ap = np.concatenate((drim_ap, ap_index_all), axis=1)
    # # all_auc = all_auc.T
    # print(all_ap.shape)
    # plt.boxplot(all_ap, labels=['DRIM', 'My'], positions=positions)
    # # plt.boxplot(ap_index_all, labels=['My'], positions=positions)
    # plt.yticks(np.arange(0.1, 1, 0.1))
    # plt.title('AUPRC')
    # plt.show()


def predict():
    import sklearn.metrics as metrics
    TF_num = 18
    data_name = 'hESC'
    drim_path = '../modelResult_500/DeepDRIM/' + data_name + '/'
    save_dir = '../modelResult_500/myModel/' + data_name + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    e = GeneData('../data_evaluation/single_cell_type/hESC/ExpressionData.csv',
                 '../data_evaluation/single_cell_type/training_pairs' + data_name + '.txt',
                 '../data_evaluation/single_cell_type/training_pairs' + data_name + '.txtTF_divide_pos.txt',
                 TF_num=18, gene_emb_path='../../gcmc/' + data_name + '_emb/user_out_vfinal_2.npy',
                 cell_emb_path='../../gcmc/' + data_name + '_emb/movie_out_vfinal_2.npy',
                 istime=False, gene_list_path='../data_evaluation/single_cell_type/' + data_name + '_geneName_map.txt',
                 data_name=data_name, TF_random=True)

    data_path = '../../dynGENIE3_python/BeelineData/500_' + data_name + '/ExpressionData.csv'
    data = pd.read_csv(data_path, header='infer', index_col=0)
    print(data)
    gene_500 = list(data.index)
    gene_500 = [gene.lower() for gene in gene_500]
    print(len(gene_500))

    AUROC_set = []
    AUPRC_set = []
    preds = []
    labels = []
    for fold in range(1, 4):
        test_index = fold
        test_TF = [i for i in range(int(np.ceil((test_index - 1) * 0.333333 * TF_num)),
                                    int(np.ceil(test_index * 0.333333 * TF_num)))]
        fold_path = save_dir + str(test_TF) + '/'
        # if not os.path.exists(fold_path):
        #     os.makedirs(fold_path)
        train_TF = [j for j in range(TF_num) if j not in test_TF]
        print("test_TF", test_TF)

        test_model = LinearNet().to(device)
        test_model.load_state_dict(
            torch.load('../modelResult/myModel/' + data_name + '_result/' + str(test_TF) + '/model_weight_best.pth'))
        test_model.eval()
        for j in test_TF:
            this_test_datas = e.datas[j]
            this_test_labels = e.labels[j]
            test_key_datas = e.gene_key_datas[j]

            test_datas = []
            test_labels = []
            for key_index, key in enumerate(test_key_datas):
                if key[0] in gene_500 and key[1] in gene_500:
                    test_datas.append(this_test_datas[key_index])
                    test_labels.append(this_test_labels[key_index])

            if len(test_datas) == 0:
                continue
            print("test_datas", len(test_datas))
            test_datas = np.asarray(test_datas)
            test_labels = np.asarray(test_labels)
            test_datas = torch.from_numpy(test_datas).float().to(device)
            test_labels = torch.from_numpy(test_labels).float().to(device)
            test_pred = test_model(test_datas)
            # test_pred = torch.nn.Sigmoid()(test_pred)
            test_pred = list(test_pred.cpu().detach().numpy().reshape(-1))
            test_labels = list(test_labels.cpu().detach().numpy().reshape(-1))
            print("test_pred", len(test_pred))
            print("test_labels", len(test_labels))
            preds.extend(test_pred)
            labels.extend(test_labels)
            fpr, tpr, thresholds = metrics.roc_curve(test_labels, test_pred, pos_label=1)
            auroc = np.trapz(tpr, fpr)
            # exit()
            AUROC_set.append(auroc)

            precision, recall, thresholds = metrics.precision_recall_curve(test_labels, test_pred, pos_label=1)
            auprc = metrics.auc(recall, precision)
            AUPRC_set.append(auprc)
    print(len(preds), len(labels))
    print(preds)
    print(labels)

    fpr, tpr, thresholds = metrics.roc_curve(labels, preds)
    final_auroc = np.trapz(tpr, fpr)
    precision, recall, thresholds = metrics.precision_recall_curve(labels, preds, pos_label=1)
    final_auprc = metrics.auc(recall, precision)

    our_dir = save_dir
    if not os.path.exists(our_dir):
        os.makedirs(our_dir)

    with open(our_dir + '/result.txt', 'w') as f:
        f.write('AUC: ' + str(final_auroc) + '\n')
        f.write('AUPRC: ' + str(final_auprc) + '\n')
        f.write('len_AUC ' + str(len(AUROC_set)) + '\n')
        f.write('len_samples ' + str(len(labels)) + '\n')
    np.save(our_dir + '/AUROC_set.npy', AUROC_set)
    np.save(our_dir + '/AUPRC_set.npy', AUPRC_set)
    print(len(AUROC_set))
    print(len(AUPRC_set))
    print(AUROC_set)
    print(AUPRC_set)

    drim_auc = np.load(drim_path + 'AUROC_set.npy')
    drim_auc = np.expand_dims(drim_auc, axis=1)
    auc_index_all = np.asarray(AUROC_set)
    auc_index_all = np.expand_dims(auc_index_all, axis=1)
    all_auc = np.concatenate((drim_auc, auc_index_all), axis=1)
    # all_auc = all_auc.T
    # print(all_auc.shape)
    positions = [0, 0.16]
    plt.boxplot(all_auc, labels=['DRIM', 'My'], positions=positions)
    plt.grid(axis='y')
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.savefig(our_dir + '/AUC_boxplot.png')
    plt.show()
    plt.close()

    # AP箱线图
    drim_ap = np.load(drim_path + 'AUPRC_set.npy')
    drim_ap = np.expand_dims(drim_ap, axis=1)
    ap_index_all = np.asarray(AUPRC_set)
    ap_index_all = np.expand_dims(ap_index_all, axis=1)
    all_ap = np.concatenate((drim_ap, ap_index_all), axis=1)
    # all_auc = all_auc.T
    # print(all_ap.shape)
    plt.boxplot(all_ap, labels=['DRIM', 'My'], positions=positions)
    # plt.boxplot(ap_index_all, labels=['My'], positions=positions)
    plt.yticks(np.arange(0.1, 1, 0.1))
    plt.title('AUPRC')
    plt.savefig(our_dir + '/AUPRC_boxplot.png')
    plt.show()



def main_graph():
    # e = GeneData('../data_evaluation/Time_data/scRNA_expression_data/mesc2_expression_data/',
    #              '../data_evaluation/Time_data/database/mesc2_gene_pairs_400.txt',
    #              '../data_evaluation/Time_data/database/mesc2_gene_pairs_400_num.txt',
    #              TF_num=38, gene_emb_path='out/mesc2_all_users_w4.npy', cell_emb_path='out/mesc2_all_items_w4.npy',
    #              istime=False)
    # e = GeneData('../data_evaluation/single_cell_type/mHSC-E/ExpressionData.csv',
    #              '../data_evaluation/single_cell_type/training_pairsmHSC_E.txt',
    #              '../data_evaluation/single_cell_type/training_pairsmHSC_E.txtTF_divide_pos.txt',
    #              TF_num=18, gene_emb_path='../../gcmc/user_out.npy', cell_emb_path='../../gcmc/movie_out.npy',
    #              istime=False)

    e = GeneData('../data_evaluation/single_cell_type/mHSC-E/ExpressionData.csv',
                 '../data_evaluation/single_cell_type/training_pairsmHSC_E.txt',
                 '../data_evaluation/single_cell_type/training_pairsmHSC_E.txtTF_divide_pos.txt',
                 TF_num=18, gene_emb_path='out/mHSC_E_all_users_w4.npy', cell_emb_path='out/mHSC_E_all_items_w4.npy',
                 istime=False)
    node_data = e.ndata['x']
    num_nodes = e.g.num_nodes()
    TF_num = 18
    # three-fold cross validation
    acc_all = []
    auc_all = []
    pre_all = []
    label_all = []
    y_test_predict = []
    y_test_true = []
    z_all = []
    epochs = 400
    start_time = time.time()

    test_indel = [[25, 30, 31, 18, 15, 26, 28, 33, 3, 4, 14, 17, 1],
                  [34, 32, 20, 27, 16, 29, 19, 35, 13, 22, 0, 5, 24],
                  [37, 36, 11, 2, 23, 12, 21, 8, 7, 10, 9, 6]]
    for fold in range(1, 4):
        test_index = fold
        test_TF = [i for i in range(int(np.ceil((test_index - 1) * 0.333333 * TF_num)),
                                    int(np.ceil(test_index * 0.333333 * TF_num)))]

        # test_TF = random.sample(list(range(TF_num)), round(TF_num / 3))
        # test_TF = test_indel[fold - 1]
        train_TF = [j for j in range(TF_num) if j not in test_TF]
        print("test_TF", test_TF)

        train_graph_datas = []
        train_labels = []
        for j in train_TF:
            print(j, e.datas[j].shape, e.labels[j].shape)
            train_graph_datas.append(e.datas[j])
            train_labels.append(e.labels[j])
            # train_idx.append(e.idx[j])
        train_graph_datas = np.concatenate(train_graph_datas, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)

        train_graph_datas, val_graph_datas, train_labels, val_labels = \
            train_test_split(train_graph_datas,
                             train_labels,
                             test_size=0.2,
                             random_state=42)

        test_graph_datas = []
        test_labels = []
        z = [0]
        z_len = 0
        for j in test_TF:
            test_graph_datas.append(e.datas[j])
            test_labels.append(e.labels[j])
            z_len += len(e.datas[j])
            z.append(z_len)
        z_all.append(z)

        test_graph_datas = np.concatenate(test_graph_datas, axis=0)
        test_labels = np.concatenate(test_labels, axis=0)

        print('train', len(train_graph_datas), train_labels.shape)
        print('val', len(val_graph_datas), val_labels.shape)
        print('test', len(test_graph_datas), test_labels.shape)
        print('=' * 100)

        # train normal
        train_data = GraphDataSet(train_graph_datas, train_labels)
        train_loader = GraphDataLoader(train_data, batch_size=512, shuffle=True, num_workers=14)

        val_data = GraphDataSet(val_graph_datas, val_labels)
        val_loader = GraphDataLoader(val_data, batch_size=512, shuffle=False, num_workers=14)

        test_data = GraphDataSet(test_graph_datas, test_labels)
        test_loader = GraphDataLoader(test_data, batch_size=512, shuffle=False, num_workers=14)
        # val_loader = test_loader

        model = GCN(
            num_layers=3,
            hidden_units=64,
            gcn_type='sage',
            pooling_type='sum',
            node_attributes=node_data,  # none
            edge_weights=None,  # none
            node_embedding=None,
            use_embedding=False,
            num_nodes=num_nodes,
            dropout=0.1
        ).to(device)
        test_model = GCN(
            num_layers=3,
            hidden_units=64,
            gcn_type='sage',
            pooling_type='sum',
            node_attributes=node_data,  # none
            edge_weights=None,  # none
            node_embedding=None,
            use_embedding=False,
            num_nodes=num_nodes,
            dropout=0.1
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.9, weight_decay=1e-6)
        # criterion = nn.BCELoss()
        criterion = nn.CrossEntropyLoss()
        # criterion = LabelSmoothingLoss(classes=2, smoothing=0.35)
        supConLoss = SupConLoss(temperature=0.07)
        # criterion = FocalLoss(gamma=2, alpha=0.5)
        val_acc_best = 0.0
        model_weight_best = None
        early_stop = 0
        stop_num = 10
        for epoch in range(epochs):
            model.train()
            train_acc_sum = 0.0
            train_loss = 0.0
            train_ce_loss = 0.0
            train_suploss = 0.0
            for batch_idx, (g, target) in enumerate(train_loader):
                g = g.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                output = model(g, g.ndata['z'], g.ndata[dgl.NID], g.edata[dgl.EID])
                loss = criterion(output, target)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.2)
                optimizer.step()

                train_loss += loss.item()
                # pred = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output))
                pred = torch.argmax(output, dim=1)
                train_acc_sum += (pred == target).sum().item()

            train_acc = train_acc_sum / len(train_loader.dataset)
            train_loss = train_loss / len(train_loader)

            model.eval()
            val_acc_sum = 0.0
            val_loss = 0.0
            pre = []
            label = []
            with torch.no_grad():
                for batch_idx, (g, target) in enumerate(val_loader):
                    g = g.to(device)
                    target = target.to(device).squeeze()
                    output = model(g, g.ndata['z'], g.ndata[dgl.NID], g.edata[dgl.EID])
                    loss = criterion(output, target)
                    val_loss += loss.item()
                    # pred = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output))
                    pred = torch.argmax(output, dim=1)

                    pre.extend(output.cpu().numpy())
                    label.extend(target.cpu().numpy())
                    val_acc_sum += (pred == target).sum().item()

            val_acc = val_acc_sum / len(val_loader.dataset)
            val_loss = val_loss / len(val_loader)

            one_hot_label = torch.eye(2)[label, :]
            # one_hot_label = label
            val_auc = roc_auc_score(one_hot_label, pre)

            if val_acc > val_acc_best:
                model_weight_best = model.state_dict()
                val_acc_best = val_acc
                early_stop = 0
            else:
                early_stop += 1

            print('Epoch: {}, Train Loss: {:.4f},Train CELoss: {:.4f},Train SupLoss: {:.4f}, Train Acc: {:.4f}, '
                  'Val Loss: {:.4f}, Val Acc: {:.4f}, Val AUC: {:.4f}'.format(
                epoch, train_loss, train_ce_loss, train_suploss, train_acc, val_loss, val_acc, val_auc
            ))
            if early_stop > stop_num:
                break

        # test
        # model = test_model
        # print(model_weight_best)
        # exit()
        test_model.load_state_dict(model_weight_best)
        test_model.eval()
        test_acc_sum = 0.0
        test_loss = 0.0
        pre = []
        label = []
        # model.load_state_dict(model_weight_best)
        with torch.no_grad():
            for batch_idx, (g, target) in enumerate(test_loader):
                g = g.to(device)
                target = target.to(device).squeeze()
                output = test_model(g, g.ndata['z'], g.ndata[dgl.NID], g.edata[dgl.EID])
                loss = criterion(output, target)
                test_loss += loss.item()
                # pred = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output))
                pred = torch.argmax(output, dim=1)
                pre.extend(output.cpu().numpy())
                # pre.extend(pred.cpu().numpy())
                label.extend(target.cpu().numpy())
                test_acc_sum += (pred == target).sum().item()
        # print(pre)
        # print(len(pre))
        # exit()

        test_acc = test_acc_sum / len(test_loader.dataset)
        test_loss = test_loss / len(test_loader)

        one_hot_label = torch.eye(2)[label, :]
        # one_hot_label = label
        test_auc = roc_auc_score(one_hot_label, pre)
        pre_all.extend(pre)
        label_all.extend(label)
        y_test_predict.append(pre)
        y_test_true.append(label)
        acc_all.append(test_acc)
        auc_all.append(test_auc)

        print('Test Loss: {:.4f}, Test Acc: {:.4f}, Test AUC: {:.4f}'.format(
            test_loss, test_acc, test_auc
        ))

    one_hot_label_all = torch.eye(2)[label_all, :]
    final_auc = roc_auc_score(one_hot_label_all, pre_all)

    final_acc = accuracy_score(label_all, np.argmax(pre_all, axis=1))

    auc_index_all = []
    test_acc_all = []
    test_auc_all = []
    for fold in range(len(acc_all)):
        print('=' * 50)
        print('Fold {} Test Acc: {:.4f}, Test AUC: {:.4f}'.format(fold, acc_all[fold], auc_all[fold]))
        test_predict = y_test_predict[fold]
        test_true = y_test_true[fold]
        z = z_all[fold]
        for i in range(len(z) - 1):
            test_predict_i = test_predict[z[i]:z[i + 1]]
            test_true_i = test_true[z[i]:z[i + 1]]
            test_true_i = torch.eye(2)[test_true_i, :]
            test_auc_i = roc_auc_score(test_true_i, test_predict_i)
            test_acc_i = accuracy_score(np.argmax(test_true_i, axis=1), np.argmax(test_predict_i, axis=1))
            print('\tindex {} Test Acc: {:.4f}, Test AUC: {:.4f}'.format(i, test_acc_i, test_auc_i))
            auc_index_all.append(test_auc_i)
            test_acc_all.append(test_acc_i)
            test_auc_all.append(test_auc_i)

    print('Final ACC: {:.4f}, AUC: {:.4f}'.format(final_acc, final_auc))
    print('Test Acc: {:.4f}, Test AUC: {:.4f}'.format(np.mean(acc_all), np.mean(auc_all)))
    print('Cost Time: {:.4f} s'.format(time.time() - start_time))
    # AUROC箱线图
    plt.boxplot(auc_index_all)
    plt.yticks(np.arange(0.1, 1, 0.1))
    plt.show()


def main_transformer():
    # e = GeneData('../data_evaluation/Time_data/scRNA_expression_data/mesc2_expression_data/',
    #              '../data_evaluation/Time_data/database/mesc2_gene_pairs_400.txt',
    #              '../data_evaluation/Time_data/database/mesc2_gene_pairs_400_num.txt',
    #              TF_num=38, gene_emb_path='out/mesc2_all_users_w4.npy', cell_emb_path='out/mesc2_all_items_w4.npy',
    #              istime=False)
    # e = GeneData('../data_evaluation/single_cell_type/mHSC-E/ExpressionData.csv',
    #              '../data_evaluation/single_cell_type/training_pairsmHSC_E.txt',
    #              '../data_evaluation/single_cell_type/training_pairsmHSC_E.txtTF_divide_pos.txt',
    #              TF_num=18, gene_emb_path='../../gcmc/emb/user_out_v1.npy', cell_emb_path='../../gcmc/emb/movie_out_v1.npy',
    #              istime=False)

    e = GeneData('../data_evaluation/single_cell_type/mHSC-E/ExpressionData.csv',
                 '../data_evaluation/single_cell_type/training_pairsmHSC_E.txt',
                 '../data_evaluation/single_cell_type/training_pairsmHSC_E.txtTF_divide_pos.txt',
                 TF_num=18, gene_emb_path='out/mHSC_E_all_users_w4.npy', cell_emb_path='out/mHSC_E_all_items_w4.npy',
                 istime=False)

    TF_num = 18
    # three-fold cross validation
    acc_all = []
    auc_all = []
    pre_all = []
    label_all = []
    y_test_predict = []
    y_test_true = []
    z_all = []
    epochs = 400
    start_time = time.time()

    test_indel = [[25, 30, 31, 18, 15, 26, 28, 33, 3, 4, 14, 17, 1],
                  [34, 32, 20, 27, 16, 29, 19, 35, 13, 22, 0, 5, 24],
                  [37, 36, 11, 2, 23, 12, 21, 8, 7, 10, 9, 6]]
    for fold in range(1, 4):
        test_index = fold
        test_TF = [i for i in range(int(np.ceil((test_index - 1) * 0.333333 * TF_num)),
                                    int(np.ceil(test_index * 0.333333 * TF_num)))]

        # test_TF = random.sample(list(range(TF_num)), round(TF_num / 3))
        # test_TF = test_indel[fold - 1]
        train_TF = [j for j in range(TF_num) if j not in test_TF]
        print("test_TF", test_TF)

        train_emb_datas = []
        train_h_datas = []
        train_labels = []
        train_idx = []
        train_node_src = []
        train_node_dst = []
        for j in train_TF:
            print(j, e.datas[j].shape, e.labels[j].shape)
            train_emb_datas.append(e.datas[j])
            train_h_datas.append(e.h_datas[j])
            train_labels.append(e.labels[j])
            train_node_src.append(e.node_src[j])
            train_node_dst.append(e.node_dst[j])
            # train_idx.append(e.idx[j])
        train_emb_datas = np.concatenate(train_emb_datas, axis=0)
        train_h_datas = np.concatenate(train_h_datas, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)
        train_node_src = np.concatenate(train_node_src, axis=0)
        train_node_dst = np.concatenate(train_node_dst, axis=0)

        print(train_node_src)
        print(train_node_dst)

        train_emb_datas, val_emb_datas, train_h_datas, val_h_datas, \
        train_node_src, val_node_src, train_node_dst, val_node_dst, train_labels, val_labels = \
            train_test_split(train_emb_datas,
                             train_h_datas,
                             train_node_src,
                             train_node_dst,
                             train_labels,
                             test_size=0.2,
                             random_state=42)
        # print(len(train_node_src))
        # print(train_node_src)
        # print(train_node_dst)
        # print(val_node_src)
        # print(val_node_dst)
        # print(np.sum(train_labels))
        # print(np.sum(val_labels))
        # exit()

        test_emb_datas = []
        test_h_datas = []
        test_labels = []
        test_node_src = []
        test_node_dst = []
        z = [0]
        z_len = 0
        for j in test_TF:
            test_emb_datas.append(e.datas[j])
            test_h_datas.append(e.h_datas[j])
            test_labels.append(e.labels[j])
            test_node_src.append(e.node_src[j])
            test_node_dst.append(e.node_dst[j])
            z_len += len(e.datas[j])
            z.append(z_len)
        z_all.append(z)
        # continue
        # np.save('z_all.npy', np.asarray(z_all))
        # exit()
        # print(z)
        # exit()
        test_emb_datas = np.concatenate(test_emb_datas, axis=0)
        test_h_datas = np.concatenate(test_h_datas, axis=0)
        test_labels = np.concatenate(test_labels, axis=0)
        test_node_src = np.concatenate(test_node_src, axis=0)
        test_node_dst = np.concatenate(test_node_dst, axis=0)

        print('train', train_emb_datas.shape, train_h_datas.shape, train_labels.shape)
        print('val', val_emb_datas.shape, val_h_datas.shape, val_labels.shape)
        print('test', test_emb_datas.shape, test_h_datas.shape, test_labels.shape)
        print('=' * 100)

        # train normal
        # continue
        # print(train_node_src)
        # exit()
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(train_emb_datas).float(),
                                           torch.from_numpy(train_h_datas).float(),
                                           torch.from_numpy(train_labels).long()),
            batch_size=512, shuffle=True, num_workers=14)

        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(val_emb_datas).float(),
                                           torch.from_numpy(val_h_datas).float(),
                                           torch.from_numpy(val_labels).long()),
            batch_size=512, shuffle=False, num_workers=14)

        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(test_emb_datas).float(),
                                           torch.from_numpy(test_h_datas).float(),
                                           torch.from_numpy(test_labels).long()),
            batch_size=512, shuffle=False, num_workers=14)
        # val_loader = test_loader
        # train postive graph
        train_pos_u = []
        train_pos_v = []
        train_neg_u = []
        train_neg_v = []
        for i in range(len(train_node_src)):
            if train_labels[i] == 1:
                train_pos_u.append(train_node_src[i])
                train_pos_v.append(train_node_dst[i])
            else:
                train_neg_u.append(train_node_src[i])
                train_neg_v.append(train_node_dst[i])
        # train_pos_g = dgl.graph((train_pos_u + train_pos_v, train_pos_v + train_pos_u), num_nodes=e.gene_emb.shape[0])
        # train_neg_g = dgl.graph((train_neg_u + train_neg_v, train_neg_v + train_neg_u), num_nodes=e.gene_emb.shape[0])
        # train_pos_g.ndata['feat'] = torch.from_numpy(e.origin_data).float()
        # print(train_pos_g)
        # print(train_pos_g.ndata['feat'].shape)
        # print(train_neg_g)
        # exit()

        # val postive graph
        # val_pos_u = []
        # val_pos_v = []
        # val_neg_u = []
        # val_neg_v = []
        # for i in range(len(val_node_src)):
        #     if val_labels[i] == 1:
        #         val_pos_u.append(val_node_src[i])
        #         val_pos_v.append(val_node_dst[i])
        #     else:
        #         val_neg_u.append(val_node_src[i])
        #         val_neg_v.append(val_node_dst[i])
        # val_pos_g = dgl.graph((val_pos_u + val_pos_v, val_pos_v + val_pos_u), num_nodes=e.gene_emb.shape[0])
        # val_neg_g = dgl.graph((val_neg_u + val_neg_v, val_neg_v + val_neg_u), num_nodes=e.gene_emb.shape[0])

        # test postive graph
        # test_pos_u = []
        # test_pos_v = []
        # test_neg_u = []
        # test_neg_v = []
        # for i in range(len(test_node_src)):
        #     if test_labels[i] == 1:
        #         test_pos_u.append(test_node_src[i])
        #         test_pos_v.append(test_node_dst[i])
        #     else:
        #         test_neg_u.append(test_node_src[i])
        #         test_neg_v.append(test_node_dst[i])
        # test_pos_g = dgl.graph((test_pos_u + test_pos_v, test_pos_v + test_pos_u), num_nodes=e.gene_emb.shape[0])
        # test_neg_g = dgl.graph((test_neg_u + test_neg_v, test_neg_v + test_neg_u), num_nodes=e.gene_emb.shape[0])

        # train gnn
        # train_pos_g = dgl.add_self_loop(train_pos_g)
        # train_neg_g = dgl.add_self_loop(train_neg_g)

        # gnn_main(train_pos_g, train_pos_g, train_neg_g, val_pos_g, val_neg_g, test_pos_g, test_neg_g,)
        # continue

        # model = LinearModel().to(device)
        # model.m.load_state_dict(torch.load('con_linear.pth'))
        # for param in model.m.parameters():
        #     param.requires_grad = False
        model = Transformer(feature_size=256, nhead=2, num_layers=3, dim_feedforward=512, dropout=0.25).to(device)
        # model.m.load_state_dict(torch.load('con_linear.pth'))
        # for param in model.m.parameters():
        #     param.requires_grad = False
        # model = MMoE(feature_dim=1024, expert_dim=1024, n_expert=10, n_task=1, use_gate=True).to(device)
        # model = GCN(in_feats=1701, hidden_feats=256, out_feats=256).to(device)
        # test_model = LinearModel().to(device)
        test_model = Transformer(feature_size=256, nhead=2, num_layers=3, dim_feedforward=512, dropout=0.25).to(device)
        # test_model = MMoE(feature_dim=1024, expert_dim=1024, n_expert=10 , n_task=1, use_gate=True).to(device)
        # test_model = GCN(in_feats=1701, hidden_feats=256, out_feats=256).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.9, weight_decay=1e-6)
        # criterion = nn.BCELoss()
        criterion = nn.CrossEntropyLoss()
        # criterion = LabelSmoothingLoss(classes=2, smoothing=0.35)
        supConLoss = SupConLoss(temperature=0.07)
        # criterion = FocalLoss(gamma=2, alpha=0.5)
        val_acc_best = 0.0
        model_weight_best = None
        early_stop = 0
        stop_num = 10
        for epoch in range(epochs):
            model.train()
            train_acc_sum = 0.0
            train_loss = 0.0
            train_ce_loss = 0.0
            train_suploss = 0.0
            for batch_idx, (emb_data, h_data, target) in enumerate(train_loader):
                emb_data = emb_data.to(device)
                h_data = h_data.to(device)
                target = target.to(device)

                optimizer.zero_grad()
                hidden_emb, output = model(emb_data, h_data)
                output = output.squeeze()

                # SupConLoss
                # suploss = sup_constrive(hidden_emb, target, 0.07)

                # output = model(e.g.to(device), data[:, 0].long(), data[:, 1].long())
                # mmoe
                # loss1 = criterion(output[0], target)
                # loss2 = criterion(output[1], target)
                # loss = loss1 + loss2
                ce_loss = criterion(output, target)
                loss = ce_loss
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()

                train_loss += loss.item()
                train_ce_loss += ce_loss.item()
                # train_suploss += suploss
                # pred = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output))
                pred = torch.argmax(output, dim=1)
                # pred1 = torch.argmax(output[0], target)
                # pred2 = torch.argmax(output[1], target)
                # pred = pred1 + pred2
                # train_acc_sum1 = (pred1 == target).sum().item()
                # train_acc_sum2 =
                #
                train_acc_sum += (pred == target).sum().item()

            train_acc = train_acc_sum / len(train_loader.dataset)
            train_loss = train_loss / len(train_loader)
            train_ce_loss = train_ce_loss / len(train_loader)
            train_suploss = train_suploss / len(train_loader)

            model.eval()
            val_acc_sum = 0.0
            val_loss = 0.0
            pre = []
            label = []
            with torch.no_grad():
                for batch_idx, (emb_data, h_data, target) in enumerate(val_loader):
                    emb_data = emb_data.to(device)
                    h_data = h_data.to(device)
                    target = target.to(device).squeeze()

                    hidden_emb, output = model(emb_data, h_data)
                    output = output.squeeze()
                    # output = model(e.g.to(device), data[:, 0].long(), data[:, 1].long())
                    loss = criterion(output, target)

                    val_loss += loss.item()
                    # pred = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output))
                    pred = torch.argmax(output, dim=1)
                    # pred1 = torch.argmax(output[0], target)
                    # pred2 = torch.argmax(output[1], target)
                    # pred = pred1 + pred2

                    pre.extend(output.cpu().numpy())
                    label.extend(target.cpu().numpy())

                    val_acc_sum += (pred == target).sum().item()

            val_acc = val_acc_sum / len(val_loader.dataset)
            val_loss = val_loss / len(val_loader)

            one_hot_label = torch.eye(2)[label, :]
            # one_hot_label = label
            val_auc = roc_auc_score(one_hot_label, pre)

            if val_acc > val_acc_best:
                model_weight_best = model.state_dict()
                val_acc_best = val_acc
                early_stop = 0
            else:
                early_stop += 1

            print('Epoch: {}, Train Loss: {:.4f},Train CELoss: {:.4f},Train SupLoss: {:.4f}, Train Acc: {:.4f}, '
                  'Val Loss: {:.4f}, Val Acc: {:.4f}, Val AUC: {:.4f}'.format(
                epoch, train_loss, train_ce_loss, train_suploss, train_acc, val_loss, val_acc, val_auc
            ))
            if early_stop > stop_num:
                break

        # test
        # model = test_model
        # print(model_weight_best)
        # exit()
        test_model.load_state_dict(model_weight_best)
        test_model.eval()
        test_acc_sum = 0.0
        test_loss = 0.0
        pre = []
        label = []
        # model.load_state_dict(model_weight_best)
        with torch.no_grad():
            for batch_idx, (emb_data, h_data, target) in enumerate(test_loader):
                emb_data = emb_data.to(device)
                h_data = h_data.to(device)
                target = target.to(device).squeeze()

                hidden_emb, output = test_model(emb_data, h_data)
                output = output.squeeze()
                # output = model(e.g.to(device), data[:, 0].long(), data[:, 1].long())
                loss = criterion(output, target)

                test_loss += loss.item()
                # pred = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output))
                pred = torch.argmax(output, dim=1)
                # pred1 = torch.argmax(output[0], target)
                # pred2 = torch.argmax(output[1], target)
                # pred = pred1 + pred2
                pre.extend(output.cpu().numpy())
                # pre.extend(pred.cpu().numpy())
                label.extend(target.cpu().numpy())

                test_acc_sum += (pred == target).sum().item()
        # print(pre)
        # print(len(pre))
        # exit()

        test_acc = test_acc_sum / len(test_loader.dataset)
        test_loss = test_loss / len(test_loader)

        one_hot_label = torch.eye(2)[label, :]
        # one_hot_label = label
        test_auc = roc_auc_score(one_hot_label, pre)
        pre_all.extend(pre)
        label_all.extend(label)
        y_test_predict.append(pre)
        y_test_true.append(label)
        acc_all.append(test_acc)
        auc_all.append(test_auc)

        print('Test Loss: {:.4f}, Test Acc: {:.4f}, Test AUC: {:.4f}'.format(
            test_loss, test_acc, test_auc
        ))
    # np.save('z_all.npy', np.asarray(z_all))
    # exit()
    one_hot_label_all = torch.eye(2)[label_all, :]
    final_auc = roc_auc_score(one_hot_label_all, pre_all)

    final_acc = accuracy_score(label_all, np.argmax(pre_all, axis=1))

    auc_index_all = []
    test_acc_all = []
    test_auc_all = []
    for fold in range(len(acc_all)):
        print('=' * 50)
        print('Fold {} Test Acc: {:.4f}, Test AUC: {:.4f}'.format(fold, acc_all[fold], auc_all[fold]))
        test_predict = y_test_predict[fold]
        test_true = y_test_true[fold]
        z = z_all[fold]
        for i in range(len(z) - 1):
            test_predict_i = test_predict[z[i]:z[i + 1]]
            test_true_i = test_true[z[i]:z[i + 1]]
            test_true_i = torch.eye(2)[test_true_i, :]
            test_auc_i = roc_auc_score(test_true_i, test_predict_i)
            test_acc_i = accuracy_score(np.argmax(test_true_i, axis=1), np.argmax(test_predict_i, axis=1))
            print('\tindex {} Test Acc: {:.4f}, Test AUC: {:.4f}'.format(i, test_acc_i, test_auc_i))
            auc_index_all.append(test_auc_i)
            test_acc_all.append(test_acc_i)
            test_auc_all.append(test_auc_i)
    print('Final ACC: {:.4f}, AUC: {:.4f}'.format(final_acc, final_auc))
    print('Test Acc: {:.4f}, Test AUC: {:.4f}'.format(np.mean(test_acc_all), np.mean(test_auc_all)))
    print('Cost Time: {:.4f} s'.format(time.time() - start_time))
    # AUROC箱线图
    plt.boxplot(auc_index_all)
    plt.yticks(np.arange(0.1, 1, 0.1))
    plt.show()


if __name__ == '__main__':
    # main_transformer()
    # main_lightgbm()
    main()
    # predict()
    # main_label3()