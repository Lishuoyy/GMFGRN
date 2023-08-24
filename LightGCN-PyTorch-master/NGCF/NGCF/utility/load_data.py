# This file is based on the NGCF author's implementation
# <https://github.com/xiangwang1223/neural_graph_collaborative_filtering/blob/master/NGCF/utility/load_data.py>.
# It implements the data processing and graph construction.
import random as rd

import dgl

import numpy as np
import pandas as pd
import collections
from tqdm import tqdm
import torch

class Data(object):
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size

        train_file = path
        test_file = path + "/test.txt"
        train_data = pd.read_csv(path, header='infer', index_col=0)
        self.rpkm = self.getScores(train_data)
        # get number of users and items
        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0
        self.exist_users = []

        user_item_src = []
        user_item_dst = []
        edge_data = []

        self.train_items, self.test_set = collections.defaultdict(list), collections.defaultdict(list)

        # with open(train_file) as f:
        #     for l in f.readlines():
        #         if len(l) > 0:
        #             l = l.strip("\n").split(" ")
        #             items = [int(i) for i in l[1:]]
        #             uid = int(l[0])
        #             self.exist_users.append(uid)
        #             self.n_items = max(self.n_items, max(items))
        #             self.n_users = max(self.n_users, uid)
        #             self.n_train += len(items)
        #             for i in l[1:]:
        #                 user_item_src.append(uid)
        #                 user_item_dst.append(int(i))
        for i in range(train_data.shape[0]):
            self.exist_users.append(i)
            self.n_users = max(self.n_users, i)
            for j in range(train_data.shape[1]):
                if train_data.iloc[i, j] != 0.0:
                    user_item_src.append(i)
                    user_item_dst.append(j)
                    edge_data.append(train_data.iloc[i, j])
                    self.n_items = max(self.n_items, j)
                    self.n_train += 1
                    self.train_items[i].append(j)
        # with open(test_file) as f:
        #     for l in f.readlines():
        #         if len(l) > 0:
        #             l = l.strip("\n")
        #             try:
        #                 items = [int(i) for i in l.split(" ")[1:]]
        #             except Exception:
        #                 continue
        #             self.n_items = max(self.n_items, max(items))
        #             self.n_test += len(items)
        self.n_test = 0
        self.n_items += 2
        self.n_users += 1

        # neg items for every user
        self.train_neg_items = {}
        for u in self.train_items.keys():
            self.train_neg_items[u] = list(set(range(self.n_items)) - set(self.train_items[u]))

        self.print_statistics()
        # exit()

        # training positive items corresponding to each user; testing positive items corresponding to each user
        # self.train_items, self.test_set = {}, {}
        # with open(train_file) as f_train:
        #     with open(test_file) as f_test:
        #         for l in f_train.readlines():
        #             if len(l) == 0:
        #                 break
        #             l = l.strip("\n")
        #             items = [int(i) for i in l.split(" ")]
        #             uid, train_items = items[0], items[1:]
        #             self.train_items[uid] = train_items
        #
        #         for l in f_test.readlines():
        #             if len(l) == 0:
        #                 break
        #             l = l.strip("\n")
        #             try:
        #                 items = [int(i) for i in l.split(" ")]
        #             except Exception:
        #                 continue
        #
        #             uid, test_items = items[0], items[1:]
        #             self.test_set[uid] = test_items

        # construct graph from the train data and add self-loops
        user_selfs = [i for i in range(self.n_users)]
        item_selfs = [i for i in range(self.n_items)]

        data_dict = {
            ("user", "user_self", "user"): (user_selfs, user_selfs),
            ("item", "item_self", "item"): (item_selfs, item_selfs),
            ("user", "ui", "item"): (user_item_src, user_item_dst),
            ("item", "iu", "user"): (user_item_dst, user_item_src),
        }
        num_dict = {"user": self.n_users, "item": self.n_items}

        self.g = dgl.heterograph(data_dict, num_nodes_dict=num_dict)

    def sample(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [
                rd.choice(self.exist_users) for _ in range(self.batch_size)
            ]

        def sample_pos_items_for_u(u, num):
            # sample num pos items for u-th user
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            pos_rpkm_batch = []
            while True:
                if len(pos_batch) == num:
                    break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
                    pos_rpkm_batch.append(self.rpkm[u][pos_i_id])
            return pos_batch, pos_rpkm_batch

        def sample_neg_items_for_u(u, num):
            # sample num neg items for u-th user
            neg_items = []
            while True:
                if len(neg_items) == num:
                    break
                # neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                neg_id = np.random.choice(self.train_neg_items[u])
                if (
                        neg_id not in self.train_items[u]
                        and neg_id not in neg_items
                ):
                    neg_items.append(neg_id)
            return neg_items

        pos_items, neg_items = [], []
        pos_edge_datas = []
        for u in users:
            pos_i, pos_rpkm = sample_pos_items_for_u(u, 1)
            pos_items += pos_i
            pos_edge_datas += pos_rpkm
            # pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items, pos_edge_datas

    def get_num_users_items(self):
        return self.n_users, self.n_items

    def print_statistics(self):
        print("n_users=%d, n_items=%d" % (self.n_users, self.n_items))
        print("n_interactions=%d" % (self.n_train + self.n_test))
        print(
            "n_train=%d, n_test=%d, sparsity=%.5f"
            % (
                self.n_train,
                self.n_test,
                (self.n_train + self.n_test) / (self.n_users * self.n_items),
            )
        )

    def getScores(self, data):
        rpkm = data.T
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
