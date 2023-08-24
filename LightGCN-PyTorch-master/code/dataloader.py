"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Shuxian Bi (stanbi@mail.ustc.edu.cn),Jianbai Ye (gusye@mail.ustc.edu.cn)
Design Dataset here
Every dataset's index has to start at 0
"""
import os
from os.path import join
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import world
from world import cprint
from time import time
import math as m


class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")

    @property
    def n_users(self):
        raise NotImplementedError

    @property
    def m_items(self):
        raise NotImplementedError

    @property
    def trainDataSize(self):
        raise NotImplementedError

    @property
    def scores(self):
        raise NotImplementedError

    @property
    def testDict(self):
        raise NotImplementedError

    @property
    def allPos(self):
        raise NotImplementedError

    @property
    def allNeg(self):
        raise NotImplementedError

    def getUserItemFeedback(self, users, items):
        raise NotImplementedError

    def getUserPosItems(self, users):
        raise NotImplementedError

    def getUserNegItems(self, users):
        """
        not necessary for large dataset
        it's stupid to return all neg items in super large dataset
        """
        raise NotImplementedError

    def getSparseGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A = 
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError

    def getScores(self):
        raise NotImplementedError


class LastFM(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    LastFM dataset
    """

    def __init__(self, path="../data/lastfm"):
        # train or test
        cprint("loading [last fm]")
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        # self.n_users = 1892
        # self.m_items = 4489
        trainData = pd.read_table(join(path, 'data1.txt'), header=None)
        # print(trainData.head())
        testData = pd.read_table(join(path, 'test1.txt'), header=None)
        # print(testData.head())
        trustNet = pd.read_table(join(path, 'trustnetwork.txt'), header=None).to_numpy()
        # print(trustNet[:5])
        trustNet -= 1
        trainData -= 1
        testData -= 1
        self.trustNet = trustNet
        self.trainData = trainData
        self.testData = testData
        self.trainUser = np.array(trainData[:][0])
        self.trainUniqueUsers = np.unique(self.trainUser)
        self.trainItem = np.array(trainData[:][1])
        # self.trainDataSize = len(self.trainUser)
        self.testUser = np.array(testData[:][0])
        self.testUniqueUsers = np.unique(self.testUser)
        self.testItem = np.array(testData[:][1])
        self.Graph = None
        print(f"LastFm Sparsity : {(len(self.trainUser) + len(self.testUser)) / self.n_users / self.m_items}")

        # (users,users)
        self.socialNet = csr_matrix((np.ones(len(trustNet)), (trustNet[:, 0], trustNet[:, 1])),
                                    shape=(self.n_users, self.n_users))
        # (users,items), bipartite graph
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_users, self.m_items))

        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_users)))
        self.allNeg = []
        allItems = set(range(self.m_items))
        for i in range(self.n_users):
            pos = set(self._allPos[i])
            neg = allItems - pos
            self.allNeg.append(np.array(list(neg)))
        self.__testDict = self.__build_test()

    @property
    def n_users(self):
        return 1892

    @property
    def m_items(self):
        return 4489

    @property
    def trainDataSize(self):
        return len(self.trainUser)

    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def getSparseGraph(self):
        if self.Graph is None:
            user_dim = torch.LongTensor(self.trainUser)
            item_dim = torch.LongTensor(self.trainItem)

            first_sub = torch.stack([user_dim, item_dim + self.n_users])
            second_sub = torch.stack([item_dim + self.n_users, user_dim])
            index = torch.cat([first_sub, second_sub], dim=1)
            data = torch.ones(index.size(-1)).int()
            self.Graph = torch.sparse.IntTensor(index, data,
                                                torch.Size([self.n_users + self.m_items, self.n_users + self.m_items]))
            dense = self.Graph.to_dense()
            D = torch.sum(dense, dim=1).float()
            D[D == 0.] = 1.
            D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
            dense = dense / D_sqrt
            dense = dense / D_sqrt.t()
            index = dense.nonzero()
            data = dense[dense >= 1e-9]
            assert len(index) == len(data)
            self.Graph = torch.sparse.FloatTensor(index.t(), data, torch.Size(
                [self.n_users + self.m_items, self.n_users + self.m_items]))
            self.Graph = self.Graph.coalesce().to(world.device)
        return self.Graph

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    def getUserNegItems(self, users):
        negItems = []
        for user in users:
            negItems.append(self.allNeg[user])
        return negItems

    def __getitem__(self, index):
        user = self.trainUniqueUsers[index]
        # return user_id and the positive items of the user
        return user

    def switch2test(self):
        """
        change dataset mode to offer test data to dataloader
        """
        self.mode = self.mode_dict['test']

    def __len__(self):
        return len(self.trainUniqueUsers)


class Loader(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    gowalla dataset
    """

    def __init__(self, config=world.config, path="../data/gowalla"):
        # train or test
        cprint(f'loading [{path}]')
        self.split = config['A_split']
        self.folds = config['A_n_fold']
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        self.n_user = 0
        self.m_item = 0
        train_file = path + '/train.txt'
        test_file = path + '/test.txt'
        self.path = path
        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        self.traindataSize = 0
        self.testDataSize = 0

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(items))
                    trainItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.traindataSize += len(items)
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    testUniqueUsers.append(uid)
                    testUser.extend([uid] * len(items))
                    testItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.testDataSize += len(items)
        self.m_item += 1
        self.n_user += 1
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)

        self.Graph = None
        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        print(f"{world.dataset} Sparsity : {(self.trainDataSize + self.testDataSize) / self.n_users / self.m_items}")

        # (users,items), bipartite graph
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item))
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        # print(self.users_D[:10])
        # exit()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.__testDict = self.__build_test()
        print(f"{world.dataset} is ready to go")

    @property
    def n_users(self):
        return self.n_user

    @property
    def m_items(self):
        return self.m_item

    @property
    def trainDataSize(self):
        return self.traindataSize

    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def _split_A_hat(self, A):
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold * fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(world.device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def getSparseGraph(self):
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except:
                print("generating adjacency matrix")
                s = time()
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"costing {end - s}s, saved norm_mat...")
                sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)

            if self.split == True:
                self.Graph = self._split_A_hat(norm_adj)
                print("done split matrix")
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().to(world.device)
                print("don't split the matrix")
        return self.Graph

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    # def getUserNegItems(self, users):
    #     negItems = []
    #     for user in users:
    #         negItems.append(self.allNeg[user])
    #     return negItems


class GeneData(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    gowalla dataset
    """

    # "../data_evaluation/single_cell_type/mHSC-E/ExpressionData.csv"
    # ../data_evaluation/Time_data/scRNA_expression_data/mesc2_expression_data/
    def __init__(self, config=world.config,):
        # train or test
        super().__init__()
        path = config['data_path']
        time = config['time']
        cprint(f'loading [{path}]')
        self.split = config['A_split']
        self.folds = config['A_n_fold']
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        self.n_user = 0
        self.m_item = 0
        if not time:
            train_data = pd.read_csv(path, header='infer', index_col=0)
            # print(train_data)
            # exit()
        else:
            time_h5 = []
            files = os.listdir(path)
            for i in range(len(files)):
                time_pd = pd.read_hdf(path + 'RPKM_' + str(i) + '.h5', key='/RPKMs')
                # print(time_pd)
                # exit()
                time_h5.append(time_pd)
            train_data = pd.concat(time_h5, axis=0, ignore_index=True)
            train_data = train_data.T
            # print(train_data)
            # exit()
        self.dataset = 'mHSC_E' # 'mesc2'
        self.rpkm_value = train_data.values
        self.scores_df = train_data.T
        self.score_table = self.getScores()
        print('max', np.max(self.rpkm_value))
        print('min', np.min(self.rpkm_value))
        # print(self.score_table)
        # exit()
        # 后加的
        self.start_index = []
        self.end_index = []
        self.key_list = []
        self.gold_standard = {}
        self.getStartEndIndex('../data_evaluation/single_cell_type/training_pairsmHSC_E.txtTF_divide_pos.txt')
        self.getLabel('../data_evaluation/single_cell_type/training_pairsmHSC_E.txt')

        # print(self.scores.shape)
        # exit()
        test_data = train_data #pd.read_csv(path, header='infer', index_col=0)
        self.gene_to_idx = {gene.lower(): i for i, gene in enumerate(train_data.index)}
        # print(self.gene_to_idx)
        # exit()
        self.cell_to_idx = {cell: i for i, cell in enumerate(train_data.columns)}
        self.path = path
        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        edge_weight = []
        self.traindataSize = 0
        self.testDataSize = 0

        print(train_data)
        # zero_count = 0
        # for i in range(train_data.shape[0]):
        #     for j in range(train_data.shape[1]):
        #         if train_data.iloc[i, j] != 0.0:
        #             zero_count += 1
        # print('no zero', zero_count)

        for i in range(train_data.shape[0]):
            index_noZero = np.nonzero(train_data.iloc[i, :].values)[0]
            if len(index_noZero) == 0:
                continue
            trainUniqueUsers.append(i)
            trainUser.extend([i] * len(index_noZero))
            trainItem.extend(index_noZero)
            edge_weight.extend(train_data.iloc[i, index_noZero].values)
            self.m_item = max(self.m_item, max(index_noZero))
            self.n_user = max(self.n_user, i)
            self.traindataSize += len(index_noZero)

            # for j in range(train_data.shape[1]):
            #     if train_data.iloc[i, j] != 0.0:
            #         trainUser.append(i)
            #         trainItem.append(j)
            #         self.m_item = max(self.m_item, j)
            #         self.n_user = max(self.n_user, i)
            #         self.traindataSize += 1
        # print(len(trainUniqueUsers))
        # print(len(trainUser))
        # print(len(trainItem))
        # print(self.m_item)
        # print(self.n_user)
        # print(self.traindataSize)
        # _, _, _, _ = self.getTrainTest(18)
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)

        self.m_item += 2  # 多加一个，防止某个基因与所有细胞都有交互
        self.n_user += 1
        self.testUniqueUsers = self.trainUniqueUsers
        self.testUser = self.trainUser
        self.testItem = self.trainItem
        self.testDataSize = 0

        self.Graph = None
        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        print(f"{world.dataset} Sparsity : {(self.trainDataSize + self.testDataSize) / self.n_users / self.m_items}")

        # (users,items), bipartite graph
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item))
        # print(len(self.user_src), len(self.user_dst))
        # exit()
        # self.UserUserNet = csr_matrix((np.ones(len(self.user_src)), (self.user_src, self.user_dst)),
        #                               shape=(self.n_user, self.n_user))
        # self.UserItemNet = csr_matrix((edge_weight, (self.trainUser, self.trainItem)),
        #                               shape=(self.n_user, self.m_item))
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        # print(self.users_D[:10])
        # exit()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self._allNeg = self.getUserNegItems(list(range(self.n_user)))
        # print(len(self._allPos))
        # exit()
        self.__testDict = self.__build_test()
        print(f"{world.dataset} is ready to go")

    @property
    def n_users(self):
        return self.n_user

    @property
    def m_items(self):
        return self.m_item

    @property
    def trainDataSize(self):
        return self.traindataSize

    @property
    def testDict(self):
        return self.__testDict

    @property
    def scores(self):
        return self.score_table

    @property
    def allPos(self):
        return self._allPos

    @property
    def allNeg(self):
        return self._allNeg

    def _split_A_hat(self, A):
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold * fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(world.device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def getSparseGraph(self):
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.dataset + '_adj_mat.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except:
                print("generating adjacency matrix")
                s = time()
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                # user_R = self.UserUserNet.tolil()
                # print(R)
                # exit()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                # adj_mat[:self.n_users, :self.n_users] = user_R

                adj_mat = adj_mat.todok()
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)

                # norm_adj = adj_mat
                # print(norm_adj)
                # exit()
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"costing {end - s}s, saved norm_mat...")
                sp.save_npz(self.dataset + '_adj_mat.npz', norm_adj)

            if self.split == True:
                self.Graph = self._split_A_hat(norm_adj)
                print("done split matrix")
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().to(world.device)
                print("don't split the matrix")
        return self.Graph

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    def getUserNegItems(self, users):
        negItems = []
        for user in users:
            negItems.append(np.where(self.UserItemNet[user].toarray() == 0.0)[1])
        return negItems

    def getScores(self):
        rpkm = self.scores_df
        columns = rpkm.columns
        geneName_to_id = {}
        for i, geneName in enumerate(rpkm.columns):
            geneName_to_id[geneName] = i

        # 处理特征
        data_values = rpkm.values
        # # mask 0
        zero_index = np.where(data_values == 0)
        mask = np.ones_like(data_values)
        mask[zero_index] = 0
        mask = torch.tensor(mask.T, dtype=torch.float32)

        #
        means = []
        stds = []
        for i in range(data_values.shape[1]):
            tmp = data_values[:, i]
            if sum(tmp != 0) == 0:
                means.append(0)
                stds.append(1)
            else:
                means.append(tmp[tmp != 0].mean())
                stds.append(tmp[tmp != 0].std())

        means = np.array(means)
        stds = np.array(stds)
        stds[np.isnan(stds)] = 1
        stds[np.isinf(stds)] = 1
        means[np.isnan(stds)] = 0
        means[np.isinf(stds)] = 0
        stds[stds == 0] = 1
        data_values = (data_values - means) / (stds)
        data_values[np.isnan(data_values)] = 0
        data_values[np.isinf(data_values)] = 0
        data_values = np.maximum(data_values, -20)
        data_values = np.minimum(data_values, 20)
        input_feature = data_values.T

        input_feature = torch.tensor(input_feature, dtype=torch.float32)
        print(input_feature)
        # exit()
        return input_feature

    def build_bin_normalized_matrix(self):
        # if os.path.exists(source_path + 'bin-normalized-matrix.csv'):
        #     print('Bin normalized matrix already exists.')
        #     return
        class_num = 5
        data = self.scores_df
        zero_num = data[data == 0].count()
        print('zero num: ', np.sum(zero_num.values))
        gene_name = list(data)
        new_data = pd.DataFrame()
        new_data.index = data.index
        new_data_dict = {}
        for gene in gene_name:
            temp = data[gene]
            non_zero_element = np.log(temp[temp != 0].values)
            mean = np.mean(non_zero_element)
            tmin = np.min(non_zero_element)
            std = np.std(non_zero_element)
            tmax = np.max(non_zero_element)
            lower_bound = max(mean - 2 * std, tmin)
            upper_bound = min(mean + 2 * std, tmax)
            bucket_width = (upper_bound - lower_bound) / class_num
            temp = temp.apply(lambda x: 0 if x == 0 else m.floor((m.log(x) - lower_bound) / bucket_width))
            temp[temp >= class_num] = class_num - 1
            temp[temp < 0] = 0
            new_data_dict[gene] = temp

        new_data = pd.DataFrame(new_data_dict)
        new_data = torch.tensor(new_data.T.values, dtype=torch.float32)
        print(new_data)
        from collections import Counter
        print(Counter(new_data.flatten().numpy()))
        return new_data

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
        # print(len(TF_order))
        # print("TF_order", TF_order)
        # random.shuffle(TF_order)
        # print("TF_order", TF_order)
        # exit()
        # TF_order = list(TF_order)
        TF_order = [13, 45, 47, 44, 17, 27, 26, 25, 31, 19, 12, 4, 34,
                    8, 3, 6, 40, 41, 46, 15, 9, 16, 24, 33, 30, 0, 43, 32, 5,
                    29, 11, 36, 1, 21, 2, 37, 35, 23, 39, 10, 22, 18, 48, 20, 7, 42, 14, 28, 38]
        # print("TF_order", TF_order)
        index_start_list = np.asarray(self.start_index)
        index_end_list = np.asarray(self.end_index)
        index_start_list = index_start_list[TF_order]
        index_end_list = index_end_list[TF_order]
        # print("index_start_list", index_start_list)
        # print("index_end_list", index_end_list)
        # exit()
        # f = open('test.txt', 'w')
        # print(index_start_list)
        # print(index_end_list)
        select_data = []
        for i in range(TF_num):
            start = index_start_list[i]
            end = index_end_list[i]
            this_data = []
            for line in self.key_list[start:end]:
                label = self.gold_standard[line]
                gene1, gene2 = line.split(',')
                gene1_index = self.gene_to_idx[gene1]
                gene2_index = self.gene_to_idx[gene2]
                if int(label) != 2:
                    this_data.append([gene1_index, gene2_index, int(label)])
            select_data.append(this_data)
        # print(len(select_data))
        train_data = []
        test_data = []

        test_index = 1
        test_TF = [i for i in range(int(np.ceil((test_index - 1) * 0.333333 * TF_num)),
                                    int(np.ceil(test_index * 0.333333 * TF_num)))]
        # import random
        # random.seed(42)
        # test_TF = random.sample(list(range(TF_num)), round(TF_num / 3))
        print(test_TF)
        train_TF = [j for j in range(TF_num) if j not in test_TF]

        for i in train_TF:
            train_data.extend(select_data[i])
        for i in test_TF:
            test_data.extend(select_data[i])
        # print(test_TF)
        # print(len(train_data))
        # print(len(test_data))
        from sklearn.model_selection import train_test_split
        train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
        user_src0 = []
        user_dst0 = []
        for src, dst, label in train_data:
            if label == 1:
                user_src0.append(src)
                user_dst0.append(dst)

        user_src = user_src0 + user_dst0
        user_dst = user_dst0 + user_src0
        user_src = np.asarray(user_src)
        user_dst = np.asarray(user_dst)
        # print(user_src.shape)
        # print(user_dst.shape)
        self.user_src = user_src
        self.user_dst = user_dst

        return np.asarray(train_data), np.asarray(val_data), np.asarray(test_data), test_index


if __name__ == '__main__':
    gene = GeneData()
    gene.build_bin_normalized_matrix()
