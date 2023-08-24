"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""
import world
import torch
from dataloader import BasicDataset
from torch import nn
import numpy as np
import torch.nn.functional as F


class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def getUsersRating(self, users):
        raise NotImplementedError


class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()

    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError

    def mse_loss(self, users, pos, neg, score):
        """
        Parameters:
            users: users list
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError

    def mse_loss2(self, users1, users2, labels):
        """
        Parameters:
            users: users list
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError


class PureMF(BasicModel):
    def __init__(self,
                 config: dict,
                 dataset: BasicDataset):
        super(PureMF, self).__init__()
        self.num_users = dataset.n_users
        self.num_items = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.__init_weight()

    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        print("using Normal distribution N(0,1) initialization for PureMF")

    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)

    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb = self.embedding_item(pos.long())
        neg_emb = self.embedding_item(neg.long())
        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1 / 2) * (users_emb.norm(2).pow(2) +
                              pos_emb.norm(2).pow(2) +
                              neg_emb.norm(2).pow(2)) / float(len(users))
        return loss, reg_loss

    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb * items_emb, dim=1)
        return self.f(scores)


class LightGCN(BasicModel):
    def __init__(self,
                 config: dict,
                 dataset: BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset: dataloader.BasicDataset = dataset
        self.rpkm = dataset.score_table
        # print(self.rpkm.shape)
        # exit()
        # self.user_pos = torch.tensor((dataset.allPos))
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_user.data = self.rpkm
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.edge_weight_fc = nn.Linear(self.rpkm.shape[1], self.latent_dim, bias=True)
        self.fc1 = nn.Sequential(
            nn.Linear(self.latent_dim * 2, self.latent_dim, bias=True),
            nn.ReLU(),
            nn.Linear(self.latent_dim, 3, bias=True),
            nn.Softmax(dim=1),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(self.latent_dim * 2, self.latent_dim, bias=True),
            nn.ReLU(),
            nn.Linear(self.latent_dim, 1, bias=True),
            # nn.Softmax(dim=1),
        )
        self.ac = nn.ReLU()
        self.w = []
        self.edge_fc = []
        for _ in range(self.n_layers):
            self.w.append(nn.Linear(self.latent_dim, self.latent_dim, bias=True))

        for _ in range(self.n_layers):
            self.edge_fc.append(nn.Linear(self.latent_dim, self.latent_dim, bias=True))
        self.w = nn.ModuleList(self.w)
        self.edge_fc = nn.ModuleList(self.edge_fc)

        # 可训练的稀疏adj
        self.train_adj = nn.Parameter(
            nn.Parameter(torch.FloatTensor(self.num_users + self.num_items, self.num_users + self.num_items),
                         requires_grad=True))
        self.train_adj.data = torch.ones([self.num_users + self.num_items, self.num_users + self.num_items]) / (
                self.num_users + self.num_items - 1) + (
                                      torch.rand((self.num_users + self.num_items) * (
                                              self.num_users + self.num_items)) * 0.0002).reshape(
            [self.num_users + self.num_items, self.num_users + self.num_items])
        self.user_adj = nn.Parameter(
            nn.Parameter(torch.FloatTensor(self.num_users, self.num_users), requires_grad=True))
        self.user_adj.data = torch.ones([self.num_users, self.num_users]) / (self.num_users - 1) + (
                torch.rand(self.num_users * self.num_users) * 0.0002).reshape(
            [self.num_users, self.num_users])
        if self.config['pretrain'] == 0:
            #             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
            #             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
            #             print('use xavier initilizer')
            # random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is already to go(dropout:{self.config['dropout']})")

        # print("save_txt")

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        # print(random_index)
        index = index[random_index]
        # print(len(index))
        # print(index)
        values = values[random_index] / keep_prob
        # print(values)
        # exit()
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def computer(self):
        """
        propagate methods for lightGCN
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        # return users_emb, items_emb
        # rpkm = self.rpkm.to(self.config['device'])
        # users_emb = self.edge_weight_fc(rpkm)

        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                # print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph

        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                # users_emb, items_emb = torch.split(all_emb, [self.num_users, self.num_items])
                # edge_emb = self.edge_fc[layer](rpkm_emb)
                # edge_emb = self.ac(edge_emb)
                # users_emb = torch.mul(users_emb, edge_emb)
                # all_emb = torch.cat([users_emb, items_emb])
                # g_droped转稠密图
                # g_droped = g_droped.to_dense()

                all_emb1 = torch.sparse.mm(g_droped, all_emb)
                # print(all_emb1)
                # exit()
            all_emb1 = self.w[layer](all_emb1)
            if layer != self.n_layers - 1:
                all_emb1 = self.ac(all_emb1)
                all_emb1 = all_emb1 + all_emb  # 残差连接
                # dropout
                # if self.training:
                #     all_emb1 = F.dropout(all_emb1, p=0.25, training=True)

            embs.append(all_emb1)
            all_emb = all_emb1
            # all_emb = self.ac(all_emb)
        embs = torch.stack(embs, dim=1)
        # embs = torch.cat(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        # light_out = embs[-1]
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        # print(users.shape)
        # print(items.shape)
        # exit()
        return users, items

    def attention(self, last_emb, all_emb):
        pass

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def getEmbedding1(self, users1, users2):
        all_users, all_items = self.computer()
        users_emb1 = all_users[users1]
        users_emb2 = all_users[users2]

        users_emb_ego1 = self.embedding_user(users1)
        users_emb_ego2 = self.embedding_user(users2)

        return users_emb1, users_emb2, users_emb_ego1, users_emb_ego2, all_items

    def bpr_loss(self, users, pos, neg):

        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss

    def mse_loss(self, users, pos, neg, score):
        # print(score)
        # exit()

        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())

        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))

        # pos_scores = torch.mul(users_emb, pos_emb)
        # pos_scores = torch.sum(pos_scores, dim=1)
        # neg_scores = torch.mul(users_emb, neg_emb)
        # neg_scores = torch.sum(neg_scores, dim=1)
        # bpr_loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        # maxi = nn.LogSigmoid()(pos_scores ** 2 - neg_scores ** 2)
        #
        # bpr_loss = -1 * torch.mean(maxi)

        # 对比学习Triplet Loss
        # triplet_loss = torch.nn.TripletMarginLoss(margin=1.0, p=2)
        # bpr_loss = triplet_loss(users_emb, pos_emb, neg_emb)

        # 二分类
        # pos_label = torch.ones_like(score)
        # neg_label = torch.zeros_like(score)
        # label = torch.cat([pos_label, neg_label], dim=0)
        # # #
        # pos_sample = torch.cat([users_emb, pos_emb], dim=1)
        # neg_sample = torch.cat([users_emb, neg_emb], dim=1)
        # all_sample = torch.cat([pos_sample, neg_sample], dim=0)
        # # #
        # predict = self.fc2(all_sample)
        # acc_sum = torch.sum((predict > 0.5).float() == label.float())
        # bce_fun = torch.nn.CrossEntropyLoss()
        # bpr_loss = bce_fun(predict, label.long())
        # acc_sum = torch.sum((predict.argmax(dim=1) == label.long()).float())

        # 多分类
        # predict1 = self.fc1(pos_sample)
        # bpr_loss1 = bce_fun(predict1, score.long())
        # acc_sum1 = torch.sum((predict1.argmax(dim=1) == score.long()).float())
        # mse
        pos_score = torch.cat([users_emb, pos_emb], dim=1)
        pos_scores = self.fc2(pos_score).squeeze()
        loss_fun = torch.nn.MSELoss()
        mse_loss = loss_fun(pos_scores, score)
        loss = mse_loss
        # mse_loss = torch.tensor(0.0).to(users_emb.device)
        # loss = mse_loss
        # mse_loss = torch.tensor(0.0).to(users_emb.device)
        # exit()
        # print(pos_scores[:10])
        # print(score[:10])
        # print(loss)
        # exit()

        # return loss, reg_loss, bpr_loss, mse_loss, acc_sum  # torch.tensor(0.0).to(users_emb.device)

        # return loss, reg_loss, bpr_loss, mse_loss, acc_sum
        zero = torch.tensor(0.0).to(users_emb.device)
        acc_sum = zero
        bpr_loss = zero
        return loss, reg_loss, bpr_loss, mse_loss, acc_sum, torch.tensor(0.0).to(users_emb.device)

    def mse_loss2(self, users1, users2, labels):
        # self.user_pos = self.user_pos.to(users1.device)
        (users_emb1, users_emb2, userEmb1, userEmb2, all_items) = self.getEmbedding1(users1.long(), users2.long())

        reg_loss = (1 / 2) * (userEmb1.norm(2).pow(2) +
                              userEmb2.norm(2).pow(2)) / float(len(users1))

        # 对比学习Triplet Loss
        # triplet_loss = torch.nn.TripletMarginLoss(margin=1.0, p=2)
        # bpr_loss = triplet_loss(users_emb, pos_emb, neg_emb)

        # 二分类
        labels = labels.long()
        all_emb = torch.cat([users_emb1, users_emb2], dim=1)
        output = self.fc1(all_emb)  # .squeeze()
        bce_fun = torch.nn.CrossEntropyLoss()
        bpr_loss = bce_fun(output, labels)
        predict = torch.argmax(output, dim=1)
        acc_sum = torch.sum(predict == labels)

        # 内积
        # inner_pro = torch.mul(users_emb1, users_emb2)
        # output = torch.sum(inner_pro, dim=1)
        # output = torch.sigmoid(output)
        # # print(output.shape)
        # bce_fun = torch.nn.BCELoss()
        # bpr_loss = bce_fun(output, labels)
        # # print(bpr_loss)
        #
        # predict = (output > 0.5).float()
        # acc_sum = torch.sum(predict == labels)
        # exit()

        return reg_loss, bpr_loss, acc_sum, output  # torch.tensor(0.0).to(users_emb.device)3

    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        # all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma
