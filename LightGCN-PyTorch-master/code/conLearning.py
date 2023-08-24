import os.path

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.nn import PairwiseDistance
# from torchvision import datasets
# dataloader
from torch.utils.data import DataLoader, TensorDataset
from train_downtask import GeneData
import random
from down_task_model import ResidualBlock
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


def set_seed(seed_num):
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)
    np.random.seed(seed_num)
    random.seed(seed_num)


set_seed(6)


# 对比学习
class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=1):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.pdist = PairwiseDistance(2)

    def forward(self, output1, output2, label):
        l2_dist = self.pdist.forward(output1, output2)
        l2_dist = l2_dist.float()
        label = label.float()
        loss_contrastive = torch.mean(label * torch.pow(l2_dist, 2) +
                                      (1 - label) * torch.pow(torch.clamp(self.margin - l2_dist, min=0.0), 2))

        return loss_contrastive


def my_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return torch.mean(torch.eq(y_true, torch.lt(y_pred, 0.5).type(y_true.dtype)).type(torch.FloatTensor))


def load_data():
    e = GeneData('../data_evaluation/single_cell_type/mHSC-L/ExpressionData.csv',
                 '../data_evaluation/single_cell_type/training_pairsmHSC_L.txt',
                 '../data_evaluation/single_cell_type/training_pairsmHSC_L.txtTF_divide_pos.txt',
                 TF_num=18, gene_emb_path='../../gcmc/mHSC_L_emb/user_out_v2.npy',
                 cell_emb_path='../../gcmc/mHSC_L_emb/movie_out_v2.npy',
                 istime=False)
    x_data = e.datas
    y_data = e.labels

    TF_num = 18
    # test_index = 1
    # test_TF = [i for i in range(int(np.ceil((test_index - 1) * 0.333333 * TF_num)),
    #                             int(np.ceil(test_index * 0.333333 * TF_num)))]
    # print(test_TF)
    # # test_TF = random.sample(list(range(TF_num)), round(TF_num / 3))
    # # test_TF = test_indel[fold - 1]
    # train_TF = [j for j in range(TF_num) if j not in test_TF]
    # x_data= []
    # y_data = []
    for j in range(TF_num):
        print(j, e.datas[j].shape, e.labels[j].shape)
        x_data.append(e.datas[j])
        y_data.append(e.labels[j])
        # train_idx.append(e.idx[j])

    x_data = np.concatenate(x_data, axis=0)
    y_data = np.concatenate(y_data, axis=0)
    x_data = x_data[:, 0, :]
    print(x_data.shape, y_data.shape)
    # exit()
    select_mum = 6000

    print(x_data.shape, 'x_data samples')
    print(y_data.shape, 'y_data samples')

    zero_index = np.where(y_data == 0)
    one_index = np.where(y_data != 0)

    positive00_list = []

    zero_rand_indeces = np.random.choice(
        zero_index[0], select_mum * 2, replace=False)
    first_y_data_zero = y_data[zero_rand_indeces]
    first_x_data_zero = x_data[zero_rand_indeces]

    positive00_list.append(first_x_data_zero)

    zero_rand_indeces = np.random.choice(
        zero_index[0], select_mum * 2, replace=False)
    second_y_data_zero = y_data[zero_rand_indeces]
    second_x_data_zero = x_data[zero_rand_indeces]
    positive00_list.append(second_x_data_zero)
    positive00_list = np.array(positive00_list, dtype='float16')
    print(positive00_list.shape)

    positive_list = positive00_list
    positive_label = [1 for _ in range(select_mum * 2)]
    positive_label = np.array(positive_label)
    print(positive_list.shape)

    negative01_list = []
    zero_rand_indeces = np.random.choice(
        zero_index[0], select_mum, replace=False)
    first_y_data_zero = y_data[zero_rand_indeces]
    first_x_data_zero = x_data[zero_rand_indeces]
    negative01_list.append(first_x_data_zero)

    one_rand_indeces = np.random.choice(
        one_index[0], select_mum, replace=False)
    second_y_data_one = y_data[one_rand_indeces]
    second_x_data_one = x_data[one_rand_indeces]
    negative01_list.append(second_x_data_one)
    negative01_list = np.array(negative01_list, dtype='float16')
    print(negative01_list.shape)

    negative10_list = []
    one_rand_indeces = np.random.choice(
        one_index[0], select_mum, replace=False)
    first_y_data_one = y_data[one_rand_indeces]
    first_x_data_one = x_data[one_rand_indeces]
    negative10_list.append(first_x_data_one)

    zero_rand_indeces = np.random.choice(
        zero_index[0], select_mum, replace=False)
    second_y_data_zero = y_data[zero_rand_indeces]
    second_x_data_zero = x_data[zero_rand_indeces]
    negative10_list.append(second_x_data_zero)
    nagetive10_list = np.array(negative10_list, dtype='float16')
    print(nagetive10_list.shape)

    negative_list = np.concatenate((negative01_list, nagetive10_list), axis=1)
    negative_label = [0 for _ in range(select_mum * 2)]
    negative_label = np.array(negative_label)
    print(negative_list.shape)

    n = len(negative_label)
    n1 = int(np.ceil(len(negative_label) * 2 / 3))
    # print(positive_list.shape)
    # exit()

    positive_train = positive_list[:, 0:n1, ]
    posi_label_train = positive_label[0:n1]
    positive_test = positive_list[:, n1 + 1:n]
    posi_label_test = positive_label[n1 + 1:n]
    negative_train = negative_list[:, 0:n1]
    nega_label_train = negative_label[0:n1]
    negative_test = negative_list[:, n1 + 1:n, ]
    nega_label_test = negative_label[n1 + 1:n]

    train_data = np.concatenate((positive_train, negative_train), axis=1)
    print("train_data shape:", train_data.shape)
    train_label = np.concatenate((posi_label_train, nega_label_train), axis=0)
    print("train_label shape", train_label.shape)

    # train_data, val_data, train_label, val_label = train_test_split(train_data, train_label, test_size=0.2, random_state=42)
    # print("train_data shape:", train_data.shape)
    # print("val_data shape:", val_data.shape)
    test_data = np.concatenate((positive_test, negative_test), axis=1)
    test_label = np.concatenate((posi_label_test, nega_label_test), axis=0)

    index = [i for i in range(len(train_label))]
    random.shuffle(index)
    x_train = train_data[:, index, ]
    y_train = train_label[index]

    index = [i for i in range(len(test_label))]
    random.shuffle(index)
    x_test = test_data[:, index, ]
    y_test = test_label[index]

    # print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    return x_train, np.array(y_train).astype('float16'), x_test, np.array(y_test).astype('float16')


class LinearNet(nn.Module):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.m = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        x = self.m(x)
        return x


# model = LinearNet()
# x = torch.rand(10, 512)
# y = model(x)
# print(y.shape)
# exit()


def con_main():
    device = 'cuda:1'
    x_train, y_train, x_test, y_test = load_data()
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    x_train1 = x_train[0, :, :, ]
    x_train2 = x_train[1, :, :, ]
    x_train = TensorDataset(torch.from_numpy(x_train1, ), torch.from_numpy(x_train2), torch.from_numpy(y_train))
    x_test1 = x_test[0, :, :, ]
    x_test2 = x_test[1, :, :, ]
    x_test = TensorDataset(torch.from_numpy(x_test1), torch.from_numpy(x_test2), torch.from_numpy(y_test))
    train_loader = DataLoader(x_train, batch_size=256, shuffle=True)
    test_loader = DataLoader(x_test, batch_size=256, shuffle=False)

    model = LinearNet()
    model = model.to(device)
    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_test_acc = 0
    for epoch in range(200):
        model.train()
        epoch_loss = 0
        train_correct = 0

        for i, (x1, x2, y) in enumerate(train_loader):
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            output1 = model(x1.float())
            output2 = model(x2.float())
            loss = criterion(output1, output2, y)
            epoch_loss += loss.item()
            dist = F.pairwise_distance(output1, output2)
            pred = torch.where(dist < 0.5, torch.ones_like(dist), torch.zeros_like(dist))

            train_correct += (pred == y).sum().item()
            # train_acc =
            loss.backward()
            optimizer.step()
        # print('Epoch: {}, Train Loss: {:.4f}, Train Accuracy: {:.4f}'.format(epoch, epoch_loss / len(train_loader),
        #                                                                train_correct / len(train_loader.dataset)))
        model.eval()
        test_loss = 0
        with torch.no_grad():
            correct = 0
            total = 0
            for i, (x1, x2, y) in enumerate(test_loader):
                x1 = x1.to(device)
                x2 = x2.to(device)
                y = y.to(device)
                output1 = model(x1.float())
                output2 = model(x2.float())
                loss = criterion(output1, output2, y)
                test_loss += loss.item()
                dist = F.pairwise_distance(output1, output2)
                pred = dist < 0.5
                correct += (pred == y).sum().item()
                total += y.size(0)
            # print('Epoch: {}, Test loss: {:.4f}, Test Accuracy: {:.4f}'.format(epoch, test_loss/ len(test_loader), correct / total))
            if correct / total > best_test_acc:
                best_test_acc = correct / total
                path = 'mHSC_L_con/'
                if not os.path.exists(path):
                    os.makedirs(path)
                torch.save(model.m.state_dict(), path + 'con_linear.pth')
            print(
                'Epoch: {}, Train Loss: {:.4f}, Train Accuracy: {:.4f}, Test loss: {:.4f}, Test Accuracy: {:.4f}, Test Best_Acc: {:.4f}'.format(
                    epoch,
                    epoch_loss / len(
                        train_loader),
                    train_correct / len(
                        train_loader.dataset),
                    test_loss / len(
                        test_loader),
                    correct / len(test_loader.dataset),
                    best_test_acc
                )
            )



if __name__ == '__main__':
    # main()

    con_main()
