import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CNNClassifier(nn.Module):

    def __init__(self, num_kernels, kernel_size, num_class, stride=1, embedding_dim=256):
        super(CNNClassifier, self).__init__()

        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.num_class = num_class
        self.stride = stride
        # input: (B, 512, 312)

        self.conv_0 = nn.Conv2d(1, self.num_kernels, (self.kernel_size[0], embedding_dim), self.stride)
        self.conv_1 = nn.Conv2d(1, self.num_kernels, (self.kernel_size[1], embedding_dim), self.stride)
        self.conv_2 = nn.Conv2d(1, self.num_kernels, (self.kernel_size[2], embedding_dim), self.stride)

        # conv output size: [(w-k)+2p]/s+1
        # (batch, channel=1, seq_len, emb_size)
        self.fc = nn.Linear(162, self.num_class)
        self.dropout = nn.Dropout(0.25)

    def forward(self, text):
        #  text (batch, seq_len, emb_size)
        emb = text.unsqueeze(dim=1)  # (batch, channel=1, seq_len, emb_size)

        # after conv: (batch, num_kernels, seq_len - kernel_size[0] + 1, 1)
        conved0 = F.relu(self.conv_0(emb).squeeze(3))
        conved1 = F.relu(self.conv_1(emb).squeeze(3))
        conved2 = F.relu(self.conv_2(emb).squeeze(3))
        # print(conved0.shape)

        # pooled: (batch, n_channel)
        # pool0 = nn.MaxPool1d(conved0.shape[2], self.stride)
        # pool1 = nn.MaxPool1d(conved1.shape[2], self.stride)
        # pool2 = nn.MaxPool1d(conved2.shape[2], self.stride)

        pooled0 = nn.Flatten()(conved0)
        # print(pooled0.shape)
        pooled1 = nn.Flatten()(conved1)
        pooled2 = nn.Flatten()(conved2)


        # (batch, n_chanel * num_filters)
        cat_pool = torch.cat([pooled0, pooled1, pooled2], dim=1)
        # print(cat_pool.shape)
        fc = self.fc(cat_pool)
        # fc = nn.Softmax(dim=1)(fc)
        return fc

#
# model = CNNClassifier(num_kernels=3, kernel_size=[2, 3, 4], num_class=128, stride=1, embedding_dim=256)
# x = torch.rand(100, 100, 256)
# print(model(x).shape)
