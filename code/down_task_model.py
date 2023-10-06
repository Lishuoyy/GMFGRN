import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import *
import numpy as np
# import random
import dgl.function as fn
from losses import SupConLoss
import torch
# from torchsummary import summary
from test import NonLocalBlock, ContextBlock
from textCNN import CNNClassifier


def set_seed(seed_num):
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)
    np.random.seed(seed_num)
    # random.seed(seed_num)


set_seed(3407)


class GCN1(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_feats)
        self.conv2 = GraphConv(hidden_feats, hidden_feats)
        self.conv3 = GraphConv(hidden_feats, out_feats)
        self.fc = nn.Sequential(
            nn.Linear(2 * out_feats, out_feats),
            nn.ReLU(),
            nn.Linear(out_feats, 2),
            nn.Softmax(dim=1),
        )

    def computer(self, g):
        h = self.conv1(g, g.ndata['x'])
        h = F.relu(h)
        h = self.conv2(g, h)
        h = F.relu(h)
        h = self.conv3(g, h)
        return h

    def forward(self, g, gene1, gene2):
        h = self.computer(g)
        x = torch.cat((h[gene1], h[gene2]), dim=1)
        x = self.fc(x)
        return x


class NonLocalBlock1D(nn.Module):
    def __init__(self, in_channels, inter_channels=None):
        super(NonLocalBlock1D, self).__init__()

        if inter_channels is None:
            inter_channels = in_channels // 2

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        self.g = nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                           padding=0)
        self.theta = nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                               padding=0)
        self.phi = nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                             padding=0)
        self.out_conv = nn.Conv1d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1,
                                  stride=1, padding=0)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channels, length = x.size()

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f = self.softmax(f)

        y = torch.matmul(f, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        y = self.out_conv(y)

        return y + x


class SqueezeBlock(nn.Module):
    def __init__(self, exp_size, divide=2):
        super(SqueezeBlock, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(exp_size, exp_size // divide),
            nn.ReLU(inplace=True),
            nn.Linear(exp_size // divide, exp_size),
            h_sigmoid()
        )

    def forward(self, x):
        batch, channels, len = x.size()
        out = F.avg_pool1d(x, kernel_size=len).view(batch, -1)
        out = self.dense(out)
        out = out.view(batch, channels, 1)
        # out = hard_sigmoid(out)
        return out * x


class SqueezeBlock2D(nn.Module):
    def __init__(self, exp_size, divide=4):
        super(SqueezeBlock2D, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(exp_size, exp_size // divide),
            nn.ReLU(inplace=True),
            nn.Linear(exp_size // divide, exp_size),
            h_sigmoid()
        )

    def forward(self, x):
        batch, channels, height, width = x.size()
        out = F.avg_pool2d(x, kernel_size=[height, width]).view(batch, -1)
        out = self.dense(out)
        out = out.view(batch, channels, 1, 1)
        # out = hard_sigmoid(out)
        return out * x


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.m = nn.Sequential(
            # nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.Linear(256, 128),
            # nn.LeakyReLU(),
            # nn.Dropout(0.5),
            nn.Linear(512, 256),
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=1),
        )

    def forward(self, x, h):
        x = self.m(x)
        x = self.fc(x)
        return None, x


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.m = nn.Sequential(
            nn.Conv1d(21, 32, 3, padding=1),
            nn.LeakyReLU(),
            # nn.Conv1d(32, 32, 3, padding=1),
            # nn.LeakyReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(0.25),

            # nn.Conv1d(32, 64, 3, padding=1),
            # nn.LeakyReLU(),
            # nn.Conv1d(64, 64, 3, padding=1),
            # nn.LeakyReLU(),
            # nn.MaxPool1d(2),
            # nn.Dropout(0.25),

            # nn.Conv1d(64, 128, 3, padding=1),
            # nn.LeakyReLU(),
            # nn.Conv1d(128, 128, 3, padding=1),
            # nn.LeakyReLU(),
            # nn.MaxPool1d(2),
            # nn.Dropout(0.25),

            # NonLocalBlock1D(32),
            # nn.LeakyReLU(),
            # nn.Dropout(0.25),

            nn.Flatten(),
            nn.Linear(4096, 2048),
            nn.LeakyReLU(),
            nn.Dropout(0.25),
            nn.Linear(2048, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x, y):
        x = self.m(x)
        return x


# class FNN()
class LinearNet(nn.Module):
    def __init__(self, emb_dim=256):
        super(LinearNet, self).__init__()
        self.ac = nn.ReLU()
        self.neighbor_model = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            self.ac,
            nn.Dropout(0.5),
            nn.Linear(emb_dim, emb_dim),
            # nn.ReLU(),
            # nn.Dropout(0.25),
            # nn.Linear(256, 256),
        )
        self.cell_model = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            self.ac,
            nn.Dropout(0.5),
            nn.Linear(emb_dim, emb_dim),
            # nn.ReLU(),
            # nn.Dropout(0.25),
            # nn.Linear(256, 256),
        )
        self.drop = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.gene1_fc = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim * 2),
            self.ac,
            nn.Dropout(0.5),
            nn.Linear(emb_dim * 2, emb_dim),
        )
        self.gene2_fc = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim * 2),
            self.ac,
            nn.Dropout(0.5),
            nn.Linear(emb_dim * 2, emb_dim),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.Linear(256, 256),
        )
        self.fc = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            self.ac,
            nn.Dropout(0.5),
            nn.Linear(emb_dim, emb_dim),
            self.ac,
            nn.Dropout(0.5),
            nn.Linear(emb_dim, 1),
            # nn.Sigmoid()
            # nn.Softmax(dim=1)
        )
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, h=None):
        gene = []
        for i in range(0, 2):
            neighbour_emb = x[:, i, :]
            neighbour_emb = self.flatten(neighbour_emb)
            neighbour_emb_1 = self.neighbor_model(neighbour_emb)
            # neighbour_emb_1 += neighbour_emb
            # neighbour_emb = neighbour_emb.reshape(neighbour_emb.shape[0], 16, 4, 4)
            # neighbour_emb = self.ac(neighbour_emb)
            # neighbour_emb = self.drop(neighbour_emb)
            gene.append(neighbour_emb_1)
        # gene = torch.cat([*gene], dim=1)
        # gene = self.ac(gene)
        # gene = self.drop(gene)
        # gene = self.gene_fc(gene)
        #
        cell = []
        for i in range(0, 2):
            neighbour_emb = x[:, i + 2, :]
            neighbour_emb = self.flatten(neighbour_emb)
            neighbour_emb_1 = self.cell_model(neighbour_emb)
            # neighbour_emb_1 += neighbour_emb
            cell.append(neighbour_emb_1)
        # cell = torch.cat([*cell], dim=1)
        # cell = self.ac(cell)
        # cell = self.drop(cell)
        # cell = self.cell_fc(cell)
        # #
        gene1_cell = torch.cat([gene[0], cell[0]], dim=1)
        gene2_cell = torch.cat([gene[1], cell[1]], dim=1)
        # gene1_cell = torch.cat([gene[0], gene[2]], dim=1)
        # gene2_cell = torch.cat([gene[1], gene[3]], dim=1)
        gene1_cell_1 = self.gene1_fc(gene1_cell)
        gene2_cell_1 = self.gene2_fc(gene2_cell)
        # gene2_cell_1 += gene2_cell
        # gene1_cell_1 += gene1_cell
        gene_cell = torch.cat([gene1_cell_1, gene2_cell_1], dim=1)
        # gene_cell = self.ac(gene_cell)
        # gene_cell = self.drop(gene_cell)

        x1 = self.fc(gene_cell)
        # 点积
        # x1 = torch.sum(gene1_cell * gene2_cell, dim=1)
        return x1


class LinearNet_label3(nn.Module):
    def __init__(self):
        super(LinearNet_label3, self).__init__()
        self.ac = nn.ReLU()
        self.neighbor_model = nn.Sequential(
            nn.Linear(256, 256),
            self.ac,
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            # nn.ReLU(),
            # nn.Dropout(0.25),
            # nn.Linear(256, 256),
        )
        self.cell_model = nn.Sequential(
            nn.Linear(256, 256),
            self.ac,
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            # nn.ReLU(),
            # nn.Dropout(0.25),
            # nn.Linear(256, 256),
        )
        self.drop = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.gene1_fc = nn.Sequential(
            nn.Linear(512, 512),
            self.ac,
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.Linear(256, 256),
        )
        self.gene2_fc = nn.Sequential(
            nn.Linear(512, 512),
            self.ac,
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.Linear(256, 256),
        )
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            self.ac,
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            self.ac,
            nn.Dropout(0.5),
            nn.Linear(256, 3),
            # nn.Sigmoid()
            # nn.LogSoftmax(dim=1)
        )
        # self.position_weight1 = nn.Parameter(torch.Tensor([256]))
        # self.position_weight2 = nn.Parameter(torch.Tensor([256]))
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, h=None):
        gene = []
        for i in range(0, 2):
            neighbour_emb = x[:, i, :]
            neighbour_emb = self.flatten(neighbour_emb)
            neighbour_emb_1 = self.neighbor_model(neighbour_emb)
            # neighbour_emb_1 += neighbour_emb
            # neighbour_emb = neighbour_emb.reshape(neighbour_emb.shape[0], 16, 4, 4)
            # neighbour_emb = self.ac(neighbour_emb)
            # neighbour_emb = self.drop(neighbour_emb)
            gene.append(neighbour_emb_1)
        # gene = torch.cat([*gene], dim=1)
        # gene = self.ac(gene)
        # gene = self.drop(gene)
        # gene = self.gene_fc(gene)
        #
        cell = []
        for i in range(0, 2):
            neighbour_emb = x[:, i + 2, :]
            neighbour_emb = self.flatten(neighbour_emb)
            neighbour_emb_1 = self.cell_model(neighbour_emb)
            # neighbour_emb_1 += neighbour_emb
            cell.append(neighbour_emb_1)
        # cell = torch.cat([*cell], dim=1)
        # cell = self.ac(cell)
        # cell = self.drop(cell)
        # cell = self.cell_fc(cell)
        # #
        gene1_cell = torch.cat([gene[0], cell[0]], dim=1)
        gene2_cell = torch.cat([gene[1], cell[1]], dim=1)

        gene1_cell_1 = self.gene1_fc(gene1_cell)
        gene2_cell_1 = self.gene2_fc(gene2_cell)
        # gene2_cell_1 += gene2_cell
        # gene1_cell_1 += gene1_cell

        gene_cell = torch.cat([gene1_cell_1, gene2_cell_1], dim=1)
        # gene_cell1 = torch.cat([gene2_cell_1, gene1_cell_1], dim=1)
        # gene_cell = self.ac(gene_cell)
        # gene_cell = self.drop(gene_cell)

        x1 = self.fc(gene_cell)
        # x2 = self.fc(gene_cell1)
        # 点积
        # x1 = torch.sum(x1 * x2, dim=1)
        return x1


class LinearNet_nei(nn.Module):
    def __init__(self,emb_dim):
        super(LinearNet_nei, self).__init__()
        self.ac = nn.ReLU()
        self.neighbor_model = nn.Sequential(
            nn.Linear(512, 256),
            self.ac,
            nn.Dropout(0.25),
            nn.Linear(256, 256),
            # nn.ReLU(),
            # nn.Dropout(0.25),
            # nn.Linear(256, 256),
        )
        self.main_model = nn.Sequential(
            nn.Linear(512, 256),
            self.ac,
            nn.Dropout(0.25),
            nn.Linear(256, 256),
            # nn.ReLU(),
            # nn.Dropout(0.25),
            # nn.Linear(256, 256),
        )
        self.drop = nn.Dropout(0.25)
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(5888, 512),
            self.ac,
            nn.Dropout(0.25),
            nn.Linear(512, 128),
            self.ac,
            nn.Dropout(0.25),
            nn.Linear(128, 1),
            # nn.Sigmoid()
            # nn.Softmax(dim=1)
        )
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, h=None):
        main_emb = x[:, 0, :]
        main_emb = self.flatten(main_emb)
        main_emb = self.main_model(main_emb)
        nei_embs = []
        for i in range(1, x.shape[1]):
            neighbour_emb = x[:, i, :]
            neighbour_emb = self.flatten(neighbour_emb)
            neighbour_emb_1 = self.neighbor_model(neighbour_emb)
            # neighbour_emb_1 += neighbour_emb
            # neighbour_emb = neighbour_emb.reshape(neighbour_emb.shape[0], 16, 4, 4)
            # neighbour_emb = self.ac(neighbour_emb)
            # neighbour_emb = self.drop(neighbour_emb)
            nei_embs.append(neighbour_emb_1)
        # gene = torch.cat([*gene], dim=1)
        nei_embs = torch.cat([*nei_embs], dim=1)
        nei_embs = self.ac(nei_embs)
        nei_embs = self.drop(nei_embs)
        # gene = self.gene_fc(gene)
        all_embs = torch.cat([main_emb, nei_embs], dim=1)

        x1 = self.fc(all_embs)
        # 点积
        # x1 = torch.sum(gene1_cell * gene2_cell, dim=1)
        return x1


class CNNC(nn.Module):
    def __init__(self):
        super(CNNC, self).__init__()
        self.m = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Flatten(),
            nn.Linear(2048, 512),
        )

    def forward(self, h):
        h = self.m(h)
        return h


class DRIM(nn.Module):
    def __init__(self):
        super(DRIM, self).__init__()
        self.single_model = CNNC()
        self.neighbour_model = CNNC()
        self.linear = LinearNet()
        self.drop = nn.Dropout(0.5)
        self.ag = nn.ReLU()
        self.fc1 = nn.Linear(512 * 23, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, linear_x, x):
        # single_img = x[:, 0, :, :].unsqueeze(1)
        # neighbour_img = x[:, 1:, :, :]
        # single_emb = self.single_model(single_img)
        # neighbour_emb = []
        # for i in range(neighbour_img.shape[1]):
        #     neighbour_emb.append(self.neighbour_model(neighbour_img[:, i, :, :].unsqueeze(1)))
        # neighbour_emb = torch.cat(neighbour_emb, dim=1)
        # x = torch.cat([single_emb, neighbour_emb], dim=1)
        # x = self.drop(x)
        # x = self.fc1(x)
        # x = self.ag(x)
        # x = self.drop(x)
        # x = self.fc2(x)
        # x = self.ag(x)
        # x = self.drop(x)
        linear_x = self.linear(linear_x)
        # x = torch.cat([x, linear_x], dim=1)
        # x = self.fc3(x)
        x = nn.Softmax(dim=1)(linear_x)
        return x


x = torch.randn(2, 23, 32, 32)
h = torch.randn(2, 4, 256)
model = LinearNet()
print(model(h, x).shape)


# exit()

# exit()
#

class BestModel(nn.Module):
    def __init__(self):
        super(BestModel, self).__init__()
        self.m = nn.Sequential(
            nn.Conv1d(4, 8, 3, padding=1, stride=1),
            nn.ReLU(),
            # nn.Conv1d(8, 8, 3, padding=1, stride=2),
            # nn.LeakyReLU(),
            nn.Dropout(0.25),
            ResidualBlock(8, 8),

            # _NonLocalBlockND(8),
            nn.ReLU(),
            nn.Dropout(0.25),

            # nn.Conv1d(8, 8, 3, padding=1, stride=2),
            # nn.LeakyReLU(),
            # nn.Dropout(0.25),

            # ResidualBlock(8, 8),

            # # _NonLocalBlockND(8),
            # nn.LeakyReLU(),
            # nn.Dropout(0.25),

            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 2),

        )
        # self.classifier = nn.Sequential(
        #     nn.Linear(1024, 2),
        #     # nn.Sigmoid()
        #     nn.Softmax(dim=1)
        # )

    def forward(self, x, y=None):
        x1 = self.m(x)
        # x2 = self.classifier(x1)
        return x1


class BestModel2(nn.Module):
    def __init__(self):
        super(BestModel2, self).__init__()
        self.gene_emb = nn.Embedding(4762, 256)
        # self.cell_emb = nn.Embedding(1071, 256)
        nn.init.normal_(self.gene_emb.weight, std=0.1)
        self.m = nn.Sequential(
            nn.Conv1d(2, 8, 3, padding=1, stride=1),
            nn.LeakyReLU(),
            # nn.Conv1d(8, 8, 3, padding=1, stride=2),
            # nn.LeakyReLU(),
            nn.Dropout(0.25),
            ResidualBlock(8, 8),

            # _NonLocalBlockND(8),
            nn.LeakyReLU(),
            nn.Dropout(0.25),

            # nn.Conv1d(8, 8, 3, padding=1, stride=2),
            # nn.LeakyReLU(),
            # nn.Dropout(0.25),

            # ResidualBlock(8, 8),

            # # _NonLocalBlockND(8),
            # nn.LeakyReLU(),
            # nn.Dropout(0.25),

            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(),
            nn.Dropout(0.25),

        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 2),
            # nn.Softmax(dim=1)
        )

    def forward(self, x, y):
        x = x.long()
        y = y.long()
        x_emb = self.gene_emb(x)
        x_emb = x_emb.unsqueeze(1)
        y_emb = self.gene_emb(y)
        y_emb = y_emb.unsqueeze(1)
        x = torch.cat((x_emb, y_emb), dim=1)
        # print(x.shape)
        # print(x_emb.shape)
        # exit()
        x = self.m(x)
        x2 = self.classifier(x)
        return None, x2


class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        # self.conv1 = SAGEConv(in_feats, h_feats, "mean")
        # self.conv2 = SAGEConv(h_feats, h_feats, "mean")
        # self.conv3 = SAGEConv(h_feats, h_feats, "mean")
        # self.conv1 = GATConv(in_feats, h_feats, num_heads=2, activation=F.relu)
        # self.conv2 = GATConv(h_feats * 2, h_feats, num_heads=2, activation=F.relu)
        # self.conv3 = GATConv(h_feats * 2, h_feats, num_heads=2, activation=F.relu)
        self.conv1 = GraphConv(in_feats, 1024, activation=F.relu)
        self.conv2 = GraphConv(1024, 512, activation=F.relu)
        self.conv3 = GraphConv(512, 512, activation=F.relu)

    def forward(self, g, in_feat):
        # print(in_feat.shape)
        h = self.conv1(g, in_feat)
        # h = torch.reshape(h, (h.shape[0], -1))
        # print(h.shape)
        # h = F.relu(h)
        h = F.dropout(h, p=0.25, training=self.training)
        h = self.conv2(g, h)
        # h = torch.reshape(h, (h.shape[0], -1))
        # h = F.relu(h)
        h = F.dropout(h, p=0.25, training=self.training)
        h = self.conv3(g, h)
        # h = torch.reshape(h, (h.shape[0], -1))
        # print(h.shape)
        return h


class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)

    def apply_edges(self, edges):
        """
        Computes a scalar score for each edge of the given graph.

        Parameters
        ----------
        edges :
            Has three members ``src``, ``dst`` and ``data``, each of
            which is a dictionary representing the features of the
            source nodes, the destination nodes, and the edges
            themselves.

        Returns
        -------
        dict
            A dictionary of new edge features.
        """
        h = torch.cat([edges.src["h"], edges.dst["h"]], 1)
        return {"score": F.sigmoid(self.W2(F.dropout(F.relu(self.W1(h)), p=0.25, training=self.training)).squeeze(1))}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            g.apply_edges(self.apply_edges)
            return g.edata["score"]


class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v("h", "h", "score"))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata["score"][:, 0]


class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=1, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # channel数减半，减少计算量
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        # 定义1x1卷积形式的embeding层
        # 从上到下相当于Transformer里的q，k，v的embeding
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

        # output embeding和Batch norm
        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

    def forward(self, x, return_nl_map=False):
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """

        batch_size = x.size(0)

        # 相当于计算value
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # 相当于计算query
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        # 相当于计算key
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        # 计算attention map
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        # output
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        # 残差连接
        z = W_y + x

        if return_nl_map:
            return z, f_div_C
        return z


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=1, stride=1),
            # nn.Conv1d(in_channels, out_channels, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.Dropout(0.25),
            # nn.Conv1d(out_channels, out_channels, 3, padding=1, stride=1),
            # nn.LeakyReLU(),
            # nn.Dropout(0.5),
        )
        self.se = SqueezeBlock(out_channels)

    def forward(self, x):
        out = self.conv(x)
        out = self.se(out)
        return out + x


class GCN(nn.Module):
    """
    GCN Model
    Attributes:
        num_layers(int): num of gcn layers
        hidden_units(int): num of hidden units
        gcn_type(str): type of gcn layer, 'gcn' for GraphConv and 'sage' for SAGEConv
        pooling_type(str): type of graph pooling to get subgraph representation
                           'sum' for sum pooling and 'center' for center pooling.
        node_attributes(Tensor, optional): node attribute
        edge_weights(Tensor, optional): edge weight
        node_embedding(Tensor, optional): pre-trained node embedding
        use_embedding(bool, optional): whether to use node embedding. Note that if 'use_embedding' is set True
                             and 'node_embedding' is None, will automatically randomly initialize node embedding.
        num_nodes(int, optional): num of nodes
        dropout(float, optional): dropout rate
        max_z(int, optional): default max vocab size of node labeling, default 1000.
    """

    def __init__(
            self,
            num_layers,
            hidden_units,
            gcn_type="gcn",
            pooling_type="sum",
            node_attributes=None,
            edge_weights=None,
            node_embedding=None,
            use_embedding=False,
            num_nodes=None,
            dropout=0.5,
            max_z=1000,
    ):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling_type = pooling_type
        self.use_attribute = False if node_attributes is None else True
        self.use_embedding = use_embedding
        self.use_edge_weight = False if edge_weights is None else True

        self.z_embedding = nn.Embedding(max_z, hidden_units)
        if node_attributes is not None:
            self.node_attributes_lookup = nn.Embedding.from_pretrained(
                node_attributes
            )
            self.node_attributes_lookup.weight.requires_grad = False
        if edge_weights is not None:
            self.edge_weights_lookup = nn.Embedding.from_pretrained(
                edge_weights
            )
            self.edge_weights_lookup.weight.requires_grad = False
        if node_embedding is not None:
            self.node_embedding = nn.Embedding.from_pretrained(node_embedding)
            self.node_embedding.weight.requires_grad = False
        elif use_embedding:
            self.node_embedding = nn.Embedding(num_nodes, hidden_units)

        initial_dim = hidden_units
        if self.use_attribute:
            initial_dim += self.node_attributes_lookup.embedding_dim
        if self.use_embedding:
            initial_dim += self.node_embedding.embedding_dim

        self.layers = nn.ModuleList()
        if gcn_type == "gcn":
            self.layers.append(
                GraphConv(initial_dim, hidden_units, allow_zero_in_degree=True)
            )
            for _ in range(num_layers - 1):
                self.layers.append(
                    GraphConv(
                        hidden_units, hidden_units, allow_zero_in_degree=True
                    )
                )
        elif gcn_type == "sage":
            self.layers.append(
                SAGEConv(initial_dim, hidden_units, aggregator_type="gcn")
            )
            for _ in range(num_layers - 1):
                self.layers.append(
                    SAGEConv(hidden_units, hidden_units, aggregator_type="gcn")
                )
        else:
            raise ValueError("Gcn type error.")

        self.linear_1 = nn.Linear(hidden_units, hidden_units)
        self.linear_2 = nn.Linear(hidden_units, 2)
        if pooling_type == "sum":
            self.pooling = SumPooling()
        elif pooling_type == "avg":
            self.pooling = AvgPooling()
        elif pooling_type == "max":
            self.pooling = MaxPooling()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, g, z, node_id=None, edge_id=None):
        """
        Args:
            g(DGLGraph): the graph
            z(Tensor): node labeling tensor, shape [N, 1]
            node_id(Tensor, optional): node id tensor, shape [N, 1]
            edge_id(Tensor, optional): edge id tensor, shape [E, 1]
        Returns:
            x(Tensor): output tensor
        """

        z_emb = self.z_embedding(z)

        if self.use_attribute:
            x = self.node_attributes_lookup(node_id)
            x = torch.cat([z_emb, x], 1)
        else:
            x = z_emb

        if self.use_edge_weight:
            edge_weight = self.edge_weights_lookup(edge_id)
        else:
            edge_weight = None

        if self.use_embedding:
            n_emb = self.node_embedding(node_id)
            x = torch.cat([x, n_emb], 1)

        for layer in self.layers[:-1]:
            x = layer(g, x, edge_weight=edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](g, x, edge_weight=edge_weight)

        x = self.pooling(g, x)
        x = F.relu(self.linear_1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.softmax(self.linear_2(x), dim=1)

        return x


class CNN2D(nn.Module):
    def __init__(self):
        super(CNN2D, self).__init__()
        self.m = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(2, 5), stride=2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(2, 5), stride=1),
            nn.ReLU(),
            nn.Dropout(0.2),

            # nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.Dropout(0.2),
            # nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1),
            # nn.ReLU(),
            # nn.Dropout(0.2),

            nn.Flatten(),
            nn.Linear(488, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
            # nn.ReLU(),
            nn.Softmax(dim=1)

        )

    def forward(self, x, y=None):
        x = self.m(x)
        return None, x


# x = torch.rand(10, 4, 256)
# model = LinearNet()
# print(model(x).shape)
# exit()

# class LinearNet1(nn.Module):
#     def __init__(self):
#         super(LinearNet1, self).__init__()
#         self.m = LinearNet()
#         self.classifier = nn.Linear(512, 1)
#         self.sg = nn.Sigmoid()
#
#     def forward(self, x):
#         x = self.m(x)
#         x = self.classifier(x)
#         x = self.sg(x)
#         return x


if __name__ == '__main__':
    model = BestModel().to('cuda')
    x = torch.rand(10, 4, 256).to('cuda')
    z = torch.rand(10, 1, 16, 16).to('cuda')
    x1, x2 = model(x, z)
    print(x2.shape)
    # summary(model, (4, 256))
    # model = MMoE(feature_dim=1024, expert_dim=1024, n_expert=4, n_task=1, use_gate=True)
    # x = torch.rand(10, 4, 256)
    # y = model(x)
    #
    nParams = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % nParams)

    # x1 = torch.reshape(x1, (10, 1, 2048))
    # labels = torch.randint(0, 2, (10, ))
    # loss = SupConLoss(temperature=0.07, contrast_mode='all')
    # print(loss(x1, labels))
