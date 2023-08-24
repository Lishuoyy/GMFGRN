# =========================================================================
# Copyright (C) 2020-2023. The UltraGCN Authors. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# NOTICE: This program bundles some third-party utility functions (hit, ndcg, 
# RecallPrecision_ATk, MRRatK_r, NDCGatK_r, test_one_batch, getLabel) under
# the MIT License.
#
# Copyright (C) 2020 Xiang Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# =========================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import torch.utils.data as data
import scipy.sparse as sp
import os
import gc
import configparser
import time
import argparse
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import sys
from tqdm import tqdm
from utils import *


def data_param_prepare(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)

    params = {}

    embedding_dim = config.getint('Model', 'embedding_dim')
    params['embedding_dim'] = embedding_dim
    ii_neighbor_num = config.getint('Model', 'ii_neighbor_num')
    params['ii_neighbor_num'] = ii_neighbor_num
    model_save_path = config['Model']['model_save_path']
    params['model_save_path'] = model_save_path
    max_epoch = config.getint('Model', 'max_epoch')
    params['max_epoch'] = max_epoch

    params['enable_tensorboard'] = config.getboolean('Model', 'enable_tensorboard')

    initial_weight = config.getfloat('Model', 'initial_weight')
    params['initial_weight'] = initial_weight

    dataset = config['Training']['dataset']
    params['dataset'] = dataset
    train_file_path = config['Training']['train_file_path']
    gpu = config['Training']['gpu']
    params['gpu'] = gpu
    device = torch.device('cuda:' + params['gpu'] if torch.cuda.is_available() else "cpu")
    params['device'] = device
    lr = config.getfloat('Training', 'learning_rate')
    params['lr'] = lr
    batch_size = config.getint('Training', 'batch_size')
    params['batch_size'] = batch_size
    early_stop_epoch = config.getint('Training', 'early_stop_epoch')
    params['early_stop_epoch'] = early_stop_epoch
    w1 = config.getfloat('Training', 'w1')
    w2 = config.getfloat('Training', 'w2')
    w3 = config.getfloat('Training', 'w3')
    w4 = config.getfloat('Training', 'w4')
    params['w1'] = w1
    params['w2'] = w2
    params['w3'] = w3
    params['w4'] = w4
    negative_num = config.getint('Training', 'negative_num')
    negative_weight = config.getfloat('Training', 'negative_weight')
    params['negative_num'] = negative_num
    params['negative_weight'] = negative_weight

    gamma = config.getfloat('Training', 'gamma')
    params['gamma'] = gamma
    lambda_ = config.getfloat('Training', 'lambda')
    params['lambda'] = lambda_
    sampling_sift_pos = config.getboolean('Training', 'sampling_sift_pos')
    params['sampling_sift_pos'] = sampling_sift_pos

    test_batch_size = config.getint('Testing', 'test_batch_size')
    params['test_batch_size'] = test_batch_size
    topk = config.getint('Testing', 'topk')
    params['topk'] = topk

    test_file_path = config['Testing']['test_file_path']

    # dataset processing
    train_data, test_data, train_mat, user_num, item_num, constraint_mat, gene_to_idx, score = load_gene_data(
        train_file_path,
        test_file_path)
    train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=5)
    test_loader = data.DataLoader(list(range(user_num)), batch_size=test_batch_size, shuffle=False, num_workers=5)

    params['user_num'] = user_num
    params['item_num'] = item_num

    # mask matrix for testing to accelarate testing speed
    mask = torch.zeros(user_num, item_num)
    interacted_items = [[] for _ in range(user_num)]
    for (u, i) in train_data:
        mask[u][i] = -np.inf
        interacted_items[u].append(i)

    # test user-item interaction, which is ground truth
    test_ground_truth_list = [[] for _ in range(user_num)]
    for (u, i) in test_data:
        test_ground_truth_list[u].append(i)

    # Compute \Omega to extend UltraGCN to the item-item co-occurrence graph
    ii_cons_mat_path = './' + dataset + '_ii_constraint_mat'
    ii_neigh_mat_path = './' + dataset + '_ii_neighbor_mat'
    uu_cons_mat_path = './' + dataset + '_uu_constraint_mat'
    uu_neigh_mat_path = './' + dataset + '_uu_neighbor_mat'

    if os.path.exists(ii_cons_mat_path):
        ii_constraint_mat = pload(ii_cons_mat_path)
        ii_neighbor_mat = pload(ii_neigh_mat_path)
        uu_constraint_mat = pload(uu_cons_mat_path)
        uu_neighbor_mat = pload(uu_neigh_mat_path)
    else:
        ii_neighbor_mat, ii_constraint_mat = get_ii_constraint_mat(train_mat, ii_neighbor_num)
        pstore(ii_neighbor_mat, ii_neigh_mat_path)
        pstore(ii_constraint_mat, ii_cons_mat_path)
        uu_neighbor_mat, uu_constraint_mat = get_uu_constraint_mat(train_mat, ii_neighbor_num)
        pstore(uu_neighbor_mat, uu_neigh_mat_path)
        pstore(uu_constraint_mat, uu_cons_mat_path)
    # print(ii_constraint_mat)
    # print(ii_neighbor_mat)
    # print(uu_constraint_mat)
    # print(uu_neighbor_mat)
    # exit()
    return params, constraint_mat, ii_constraint_mat, ii_neighbor_mat, uu_constraint_mat, uu_neighbor_mat, \
           train_loader, test_loader, mask, test_ground_truth_list, interacted_items, gene_to_idx, score


def get_ii_constraint_mat(train_mat, num_neighbors, ii_diagonal_zero=False):
    print('Computing \\Omega for the item-item graph... ')
    A = train_mat.T.dot(train_mat)  # I * I

    n_items = A.shape[0]
    res_mat = torch.zeros((n_items, num_neighbors))
    res_sim_mat = torch.zeros((n_items, num_neighbors))
    if ii_diagonal_zero:
        A[range(n_items), range(n_items)] = 0
    items_D = np.sum(A, axis=0).reshape(-1)
    users_D = np.sum(A, axis=1).reshape(-1)

    beta_uD = (np.sqrt(users_D + 1) / users_D).reshape(-1, 1)
    # fill nan
    beta_uD[np.isnan(beta_uD)] = 0
    beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)

    all_ii_constraint_mat = torch.from_numpy(beta_uD.dot(beta_iD))
    for i in range(n_items - 1):
        row = all_ii_constraint_mat[i] * torch.from_numpy(A.getrow(i).toarray()[0])
        row_sims, row_idxs = torch.topk(row, num_neighbors)
        res_mat[i] = row_idxs
        res_sim_mat[i] = row_sims
        if i % 15000 == 0:
            print('i-i constraint matrix {} ok'.format(i))

    print('Computation \\Omega OK!')
    # print(res_mat.long())
    # print(res_sim_mat.float())
    # exit()
    return res_mat.long(), res_sim_mat.float()


def get_uu_constraint_mat(train_mat, num_neighbors, uu_diagonal_zero=False):
    print('Computing \\Omega for the user-user graph... ')
    A = train_mat.dot(train_mat.T)  # U * U

    n_users = A.shape[0]
    res_mat = torch.zeros((n_users, num_neighbors))
    res_sim_mat = torch.zeros((n_users, num_neighbors))
    if uu_diagonal_zero:
        A[range(n_users), range(n_users)] = 0
    items_D = np.sum(A, axis=0).reshape(-1)
    users_D = np.sum(A, axis=1).reshape(-1)

    beta_uD = (1 / np.sqrt(users_D + 1)).reshape(-1, 1)

    beta_iD = (np.sqrt(items_D + 1) / items_D).reshape(1, -1)
    # fill nan
    beta_iD[np.isnan(beta_iD)] = 0

    all_uu_constraint_mat = torch.from_numpy(beta_uD.dot(beta_iD))
    for i in range(n_users):
        row = all_uu_constraint_mat[i] * torch.from_numpy(A.getrow(i).toarray()[0])
        row_sims, row_idxs = torch.topk(row, num_neighbors)
        res_mat[i] = row_idxs
        res_sim_mat[i] = row_sims
        if i % 15000 == 0:
            print('u-u constraint matrix {} ok'.format(i))

    print('Computation \\Omega OK!')
    # print(res_mat.long())
    # print(res_sim_mat.float())
    # exit()
    return res_mat.long(), res_sim_mat.float()


def load_data(train_file, test_file):
    trainUniqueUsers, trainItem, trainUser = [], [], []
    testUniqueUsers, testItem, testUser = [], [], []
    n_user, m_item = 0, 0
    trainDataSize, testDataSize = 0, 0
    with open(train_file, 'r') as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                items = [int(i) for i in l[1:]]
                uid = int(l[0])
                trainUniqueUsers.append(uid)
                trainUser.extend([uid] * len(items))
                trainItem.extend(items)
                m_item = max(m_item, max(items))
                n_user = max(n_user, uid)
                trainDataSize += len(items)
    trainUniqueUsers = np.array(trainUniqueUsers)
    trainUser = np.array(trainUser)
    trainItem = np.array(trainItem)

    with open(test_file) as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                try:
                    items = [int(i) for i in l[1:]]
                except:
                    items = []
                uid = int(l[0])
                testUniqueUsers.append(uid)
                testUser.extend([uid] * len(items))
                testItem.extend(items)
                try:
                    m_item = max(m_item, max(items))
                except:
                    m_item = m_item
                n_user = max(n_user, uid)
                testDataSize += len(items)

    train_data = []
    test_data = []

    n_user += 1
    m_item += 1

    for i in range(len(trainUser)):
        train_data.append([trainUser[i], trainItem[i]])
    for i in range(len(testUser)):
        test_data.append([testUser[i], testItem[i]])
    train_mat = sp.dok_matrix((n_user, m_item), dtype=np.float32)

    for x in train_data:
        train_mat[x[0], x[1]] = 1.0

    # construct degree matrix for graphmf

    items_D = np.sum(train_mat, axis=0).reshape(-1)
    users_D = np.sum(train_mat, axis=1).reshape(-1)

    beta_uD = (np.sqrt(users_D + 1) / users_D).reshape(-1, 1)
    beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)

    constraint_mat = {"beta_uD": torch.from_numpy(beta_uD).reshape(-1),
                      "beta_iD": torch.from_numpy(beta_iD).reshape(-1)}

    return train_data, test_data, train_mat, n_user, m_item, constraint_mat


def prepare_score(df):
    rpkm = df
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
    return input_feature


def load_gene_data(train_file, test_file):
    trainUniqueUsers, trainItem, trainUser = [], [], []
    testUniqueUsers, testItem, testUser = [], [], []
    n_user, m_item = 0, 0
    trainDataSize, testDataSize = 0, 0
    train_data = pd.read_csv(train_file, header='infer', index_col=0)
    gene_to_idx = {gene: i for i, gene in enumerate(train_data.index)}
    cell_to_idx = {cell: i for i, cell in enumerate(train_data.columns)}
    score = prepare_score(train_data.T)

    for i in range(train_data.shape[0]):
        trainUniqueUsers.append(i)
        for j in range(train_data.shape[1]):
            if train_data.iloc[i, j] != 0.0:
                trainUser.append(i)
                trainItem.append(j)
                m_item = max(m_item, j)
                n_user = max(n_user, i)
                trainDataSize += 1

    trainUniqueUsers = np.array(trainUniqueUsers)
    trainUser = np.array(trainUser)
    trainItem = np.array(trainItem)

    m_item += 2  # 多加一个，防止某个基因与所有细胞都有交互
    n_user += 1
    testUniqueUsers = trainUniqueUsers
    testUser = trainUser
    testItem = trainItem
    testDataSize = 0

    train_data = []
    test_data = []

    # n_user += 1
    # m_item += 1

    for i in range(len(trainUser)):
        train_data.append([trainUser[i], trainItem[i]])
    # for i in range(len(testUser)):
    #     test_data.append([testUser[i], testItem[i]])
    train_mat = sp.dok_matrix((n_user, m_item), dtype=np.float32)

    for x in train_data:
        train_mat[x[0], x[1]] = 1.0

    # construct degree matrix for graphmf

    items_D = np.sum(train_mat, axis=0).reshape(-1)
    users_D = np.sum(train_mat, axis=1).reshape(-1)

    beta_uD = (np.sqrt(users_D + 1) / users_D).reshape(-1, 1)
    beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)

    constraint_mat = {"beta_uD": torch.from_numpy(beta_uD).reshape(-1),
                      "beta_iD": torch.from_numpy(beta_iD).reshape(-1)}

    return train_data, test_data, train_mat, n_user, m_item, constraint_mat, gene_to_idx, score


def pload(path):
    with open(path, 'rb') as f:
        res = pickle.load(f)
    print('load path = {} object'.format(path))
    return res


def pstore(x, path):
    with open(path, 'wb') as f:
        pickle.dump(x, f)
    print('store object in path = {} ok'.format(path))


def Sampling(pos_train_data, item_num, neg_ratio, interacted_items, sampling_sift_pos, no_interacted_items):
    # print(type(pos_train_data))
    # print(pos_train_data)
    # print(len(pos_train_data))
    # print(len(pos_train_data[0]))
    # exit()
    neg_candidates = np.arange(item_num)

    if sampling_sift_pos:
        neg_items = []
        for u in tqdm(pos_train_data[:, 0], total=len(pos_train_data[:, 0])):
            # probs = np.ones(item_num)
            # probs[interacted_items[u]] = 0
            # probs /= np.sum(probs)
            # u_neg_items = np.random.choice(neg_candidates, size=neg_ratio, p=probs, replace=True).reshape(1, -1)
            negItems = no_interacted_items[u]
            # print(len(negItems))
            u_neg_items = np.random.choice(no_interacted_items[u], size=neg_ratio, replace=True).reshape(1, -1)
            neg_items.append(u_neg_items)

        neg_items = np.concatenate(neg_items, axis=0)
    else:
        neg_items = np.random.choice(neg_candidates, (len(pos_train_data[0]), neg_ratio), replace=True)

    neg_items = torch.from_numpy(neg_items)

    return pos_train_data[:, 0], pos_train_data[:, 1], neg_items  # users, pos_items, neg_items


class UltraGCN(nn.Module):
    def __init__(self, params, constraint_mat, ii_constraint_mat, ii_neighbor_mat, uu_constraint_mat, uu_neighbor_mat):
        super(UltraGCN, self).__init__()
        self.user_num = params['user_num']
        self.item_num = params['item_num']
        self.embedding_dim = params['embedding_dim']
        self.w1 = params['w1']
        self.w2 = params['w2']
        self.w3 = params['w3']
        self.w4 = params['w4']

        self.negative_weight = params['negative_weight']
        self.gamma = params['gamma']
        self.lambda_ = params['lambda']

        self.user_embeds = nn.Embedding(self.user_num, self.embedding_dim)
        self.item_embeds = nn.Embedding(self.item_num, self.embedding_dim)

        self.constraint_mat = constraint_mat
        self.ii_constraint_mat = ii_constraint_mat
        self.ii_neighbor_mat = ii_neighbor_mat
        self.uu_constraint_mat = uu_constraint_mat
        self.uu_neighbor_mat = uu_neighbor_mat

        self.initial_weight = params['initial_weight']
        self.mse_u = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.mse_i = nn.Linear(self.embedding_dim, self.embedding_dim , bias=True)
        self.fc1 = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim, bias=True),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, 1, bias=True),
            # nn.Sigmoid()
        )
        self.initial_weights()

    def initial_weights(self):
        nn.init.normal_(self.user_embeds.weight, std=self.initial_weight)
        nn.init.normal_(self.item_embeds.weight, std=self.initial_weight)

    def get_omegas(self, users, pos_items, neg_items):
        device = self.get_device()
        if self.w2 > 0:
            pos_weight = torch.mul(self.constraint_mat['beta_uD'][users], self.constraint_mat['beta_iD'][pos_items]).to(
                device)
            pow_weight = self.w1 + self.w2 * pos_weight
        else:
            pos_weight = self.w1 * torch.ones(len(pos_items)).to(device)

        # users = (users * self.item_num).unsqueeze(0)
        if self.w4 > 0:
            neg_weight = torch.mul(torch.repeat_interleave(self.constraint_mat['beta_uD'][users], neg_items.size(1)),
                                   self.constraint_mat['beta_iD'][neg_items.flatten()]).to(device)
            neg_weight = self.w3 + self.w4 * neg_weight
        else:
            neg_weight = self.w3 * torch.ones(neg_items.size(0) * neg_items.size(1)).to(device)

        weight = torch.cat((pow_weight, neg_weight))
        return weight

    def cal_loss_L(self, users, pos_items, neg_items, omega_weight):
        device = self.get_device()
        user_embeds = self.user_embeds(users)
        pos_embeds = self.item_embeds(pos_items)
        neg_embeds = self.item_embeds(neg_items)

        pos_scores = (user_embeds * pos_embeds).sum(dim=-1)  # batch_size
        user_embeds = user_embeds.unsqueeze(1)
        neg_scores = (user_embeds * neg_embeds).sum(dim=-1)  # batch_size * negative_num

        # pos_cat = torch.cat((user_embeds, pos_embeds), dim=-1)
        # user_embeds = torch.repeat_interleave(user_embeds, 10, 0)
        # neg_embeds = neg_embeds.reshape(-1, 256)
        # neg_cat = torch.cat((user_embeds, neg_embeds), dim=-1)
        # pos_scores = self.fc1(pos_cat).squeeze()
        # neg_scores = self.fc1(neg_cat)
        # neg_scores = neg_scores.reshape(-1, 10)

        neg_labels = torch.zeros(neg_scores.size()).to(device)
        neg_loss = F.binary_cross_entropy_with_logits(neg_scores, neg_labels,
                                                      weight=omega_weight[len(pos_scores):].view(neg_scores.size()),
                                                      reduction='none').mean(dim=-1)

        pos_labels = torch.ones(pos_scores.size()).to(device)
        pos_loss = F.binary_cross_entropy_with_logits(pos_scores, pos_labels, weight=omega_weight[:len(pos_scores)],
                                                      reduction='none')
        # exit()

        loss = pos_loss + neg_loss * self.negative_weight
        # print('loss', loss)
        # exit()

        return loss.mean()

    def cal_loss_I(self, users, pos_items):
        device = self.get_device()
        neighbor_embeds = self.item_embeds(
            self.ii_neighbor_mat[pos_items].to(device))  # len(pos_items) * num_neighbors * dim
        sim_scores = self.ii_constraint_mat[pos_items].to(device)  # len(pos_items) * num_neighbors
        user_embeds = self.user_embeds(users).unsqueeze(1)

        loss = -sim_scores * (user_embeds * neighbor_embeds).sum(dim=-1).sigmoid().log()

        # loss = loss.sum(-1)
        return loss.mean()

    def cal_loss_U(self, users, pos_items):
        device = self.get_device()
        neighbor_embeds = self.user_embeds(
            self.uu_neighbor_mat[users].to(device))
        sim_scores = self.uu_constraint_mat[users].to(device)
        item_embeds = self.item_embeds(pos_items).unsqueeze(1)

        loss = -sim_scores * (item_embeds * neighbor_embeds).sum(dim=-1).sigmoid().log()

        return loss.mean()

    def mse_loss(self, users, pos_items, rpkms):
        device = self.get_device()
        user_embeds = self.user_embeds(users).to(device)
        pos_embeds = self.item_embeds(pos_items).to(device)

        # feat = torch.cat((user_embeds, pos_embeds), dim=-1)

        feat_u = self.mse_u(user_embeds)
        feat_i = self.mse_i(pos_embeds)
        #
        predi_rpkms = (feat_u * feat_i).sum(dim=-1)
        # predi_rpkms = (user_embeds * pos_embeds).sum(dim=-1)

        return F.mse_loss(predi_rpkms, rpkms)

        # return loss

    def norm_loss(self):
        loss = 0.0
        for parameter in self.parameters():
            loss += torch.sum(parameter ** 2)
        return loss / 2

    def forward(self, users, pos_items, neg_items, rpkms):
        omega_weight = self.get_omegas(users, pos_items, neg_items)

        cal_loss_l = self.cal_loss_L(users, pos_items, neg_items, omega_weight)
        norm_loss = self.gamma * self.norm_loss()
        cal_loss_i = self.lambda_ * self.cal_loss_I(users, pos_items)
        # cal_loss_u = self.lambda_ * self.cal_loss_U(users, pos_items)

        mse_loss = self.mse_loss(users, pos_items, rpkms)

        loss = cal_loss_l + norm_loss + cal_loss_i + mse_loss
        return loss, cal_loss_l, cal_loss_i, torch.tensor(0.0), mse_loss, norm_loss

    def test_foward(self, users):
        items = torch.arange(self.item_num).to(users.device)
        user_embeds = self.user_embeds(users)
        item_embeds = self.item_embeds(items)

        return user_embeds.mm(item_embeds.t())

    def get_device(self):
        return self.user_embeds.weight.device


########################### TRAINING #####################################

def train(model, optimizer, train_loader, test_loader, mask, test_ground_truth_list, interacted_items, params,
          gene_to_idx, score):
    all_items = list(range(params['item_num']))
    no_interacted_items = [[] for _ in range(params['user_num'])]
    for user in range(params['user_num']):
        no_interacted_items[user].extend(list(set(all_items) - set(interacted_items[user])))

    device = params['device']
    best_epoch, best_recall, best_ndcg = 0, 0, 0
    best_loss = 1e10
    early_stop_count = 0
    early_stop = False

    batches = len(train_loader.dataset) // params['batch_size']
    if len(train_loader.dataset) % params['batch_size'] != 0:
        batches += 1
    print('Total training batches = {}'.format(batches))

    if params['enable_tensorboard']:
        writer = SummaryWriter()

    dataset = torch.tensor(train_loader.dataset)
    for epoch in range(params['max_epoch']):
        start_time = time.time()
        sample_time = time.time()

        users, pos_items, neg_items = Sampling(dataset, params['item_num'], params['negative_num'], interacted_items,
                                               params['sampling_sift_pos'], no_interacted_items)
        user_pos_item_rpkm = score[users, pos_items]
        # user_pos_item_rpkm = torch.tensor(user_pos_item_rpkm).float()
        # print(user_pos_item_rpkm.shape)
        # exit()
        print('Sampling time: {}'.format(time.time() - sample_time))
        train_dataset = torch.utils.data.TensorDataset(users, pos_items, neg_items, user_pos_item_rpkm)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True,
                                                   num_workers=5)
        # exit()
        loss_all = 0.0
        cal_loss_l_all = 0.0
        cal_loss_i_all = 0.0
        cal_loss_u_all = 0.0
        mse_loss_all = 0.0
        norm_loss_all = 0.0
        model.train()
        for batch, x in tqdm(enumerate(train_loader), total=len(train_loader)):  # x: tensor:[users, pos_items]
            # users, pos_items, neg_items = Sampling(x, params['item_num'], params['negative_num'], interacted_items,
            #                                        params['sampling_sift_pos'])
            users, pos_items, neg_items, rpkms = x
            users = users.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)
            rpkms = rpkms.to(device)

            model.zero_grad()
            loss, cal_loss_l, cal_loss_i, cal_loss_u, mse_loss, norm_loss = model(users, pos_items, neg_items, rpkms)
            loss_all += loss.item()
            cal_loss_l_all += cal_loss_l.item()
            cal_loss_i_all += cal_loss_i.item()
            cal_loss_u_all += cal_loss_u.item()
            mse_loss_all += mse_loss.item()
            norm_loss_all += norm_loss.item()

            if params['enable_tensorboard']:
                writer.add_scalar("Loss/train_batch", loss, batches * epoch + batch)
            loss.backward()
            optimizer.step()

        train_time = time.strftime("%H: %M: %S", time.gmtime(time.time() - start_time))
        if params['enable_tensorboard']:
            writer.add_scalar("Loss/train_epoch", loss, epoch)

        need_test = True
        # if epoch < 50 and epoch % 5 != 0:
        #     need_test = False

        if need_test:
            # start_time = time.time()
            # F1_score, Precision, Recall, NDCG = test(model, test_loader, test_ground_truth_list, mask, params['topk'],
            #                                          params['user_num'])
            # if params['enable_tensorboard']:
            #     writer.add_scalar('Results/recall@20', Recall, epoch)
            #     writer.add_scalar('Results/ndcg@20', NDCG, epoch)
            # test_time = time.strftime("%H: %M: %S", time.gmtime(time.time() - start_time))
            #
            # print('The time for epoch {} is: train time = {}, test time = {}'.format(epoch, train_time, test_time))
            # print(
            #     "Loss = {:.5f}, F1-score: {:5f} \t Precision: {:.5f}\t Recall: {:.5f}\tNDCG: {:.5f}".format(loss.item(),
            #                                                                                                 F1_score,
            #                                                                                                 Precision,
            #                                                                                                 Recall,
            #                                                                                                 NDCG))
            model.eval()
            print('The time for epoch {} is: train time = {}'.format(epoch, train_time))
            print(
                "Loss = {:.5f}, cal_loss_l: {:5f} \t cal_loss_i: {:.5f}\t cal_loss_u: {:.5f}\t mse_loss: {"
                ":.5f}\tnorm_loss: {:.5f}".format(
                    loss_all / batches, cal_loss_l_all / batches, cal_loss_i_all / batches, cal_loss_u_all / batches,
                    mse_loss_all / batches,
                    norm_loss_all / batches))
            testEPR(model)
            testACC(model, gene_to_idx)
            # if Recall > best_recall:
            #     best_recall, best_ndcg, best_epoch = Recall, NDCG, epoch
            #     early_stop_count = 0
            # torch.save(model.state_dict(), params['model_save_path'])

            if loss.item() < best_loss:
                best_loss = loss.item()
                early_stop_count = 0
                torch.save(model.state_dict(), params['model_save_path'])
                all_users = model.user_embeds.weight.data.cpu().numpy()
                all_items = model.item_embeds.weight.data.cpu().numpy()
                np.save('./embs/mHSC_E_all_users_w3.npy', all_users)
                np.save('./embs/mHSC_E_all_items_w3.npy', all_items)
            else:
                early_stop_count += 1
                if early_stop_count == params['early_stop_epoch']:
                    early_stop = True

        if early_stop:
            print('##########################################')
            print('Early stop is triggered at {} epochs.'.format(epoch))
            print('Results:')
            print('best epoch = {}, best recall = {}, best ndcg = {}'.format(best_epoch, best_recall, best_ndcg))
            print('The best model is saved at {}'.format(params['model_save_path']))
            break

    writer.flush()

    print('Training end!')


########################### TESTING #####################################

def hit(gt_item, pred_items):
    if gt_item in pred_items:
        return 1
    return 0


def ndcg(gt_item, pred_items):
    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return np.reciprocal(np.log2(index + 2))
    return 0


def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k

    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall_n = np.where(recall_n != 0, recall_n, 1)
    recall = np.sum(right_pred / recall_n)
    precis = np.sum(right_pred) / precis_n
    return {'recall': recall, 'precision': precis}


def MRRatK_r(r, k):
    """
    Mean Reciprocal Rank
    """
    pred_data = r[:, :k]
    scores = np.log2(1. / np.arange(1, k + 1))
    pred_data = pred_data / scores
    pred_data = pred_data.sum(1)
    return np.sum(pred_data)


def NDCGatK_r(test_data, r, k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)


def test_one_batch(X, k):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = getLabel(groundTrue, sorted_items)
    ret = RecallPrecision_ATk(groundTrue, r, k)
    return ret['precision'], ret['recall'], NDCGatK_r(groundTrue, r, k)


def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')


def test(model, test_loader, test_ground_truth_list, mask, topk, n_user):
    users_list = []
    rating_list = []
    groundTrue_list = []

    with torch.no_grad():
        model.eval()
        for idx, batch_users in enumerate(test_loader):
            batch_users = batch_users.to(model.get_device())
            rating = model.test_foward(batch_users)
            rating = rating.cpu()
            rating += mask[batch_users]

            _, rating_K = torch.topk(rating, k=topk)
            rating_list.append(rating_K)

            groundTrue_list.append([test_ground_truth_list[u] for u in batch_users])

    X = zip(rating_list, groundTrue_list)
    Recall, Precision, NDCG = 0, 0, 0

    for i, x in enumerate(X):
        precision, recall, ndcg = test_one_batch(x, topk)
        Recall += recall
        Precision += precision
        NDCG += ndcg

    Precision /= n_user
    Recall /= n_user
    NDCG /= n_user
    F1_score = 2 * (Precision * Recall) / (Precision + Recall)

    return F1_score, Precision, Recall, NDCG


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, help='config file path')
    args = parser.parse_args()

    print('###################### UltraGCN ######################')

    print('Loading Configuration...')
    params, constraint_mat, ii_constraint_mat, ii_neighbor_mat, uu_constraint_mat, uu_neighbor_mat, train_loader, \
    test_loader, mask, test_ground_truth_list, interacted_items, gene_to_idx, score = data_param_prepare(
        args.config_file)

    print('Load Configuration OK, show them below')
    print('Configuration:')
    print(params)
    # exit()
    ultragcn = UltraGCN(params, constraint_mat, ii_constraint_mat, ii_neighbor_mat, uu_constraint_mat, uu_neighbor_mat)
    ultragcn = ultragcn.to(params['device'])
    optimizer = torch.optim.Adam(ultragcn.parameters(), lr=params['lr'])
    # model_weight = torch.load(params['model_save_path'])
    # ultragcn.load_state_dict(model_weight)
    # print('Load Model OK')

    train(ultragcn, optimizer, train_loader, test_loader, mask, test_ground_truth_list,
          interacted_items, params,
          gene_to_idx, score)

    print('END')
