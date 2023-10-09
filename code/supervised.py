import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, precision_recall_curve, auc
from sklearn.model_selection import train_test_split, KFold

import random
import time
from model import LinearNet
import os
from data import GeneData
import argparse


def set_seed(seed_num):
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)
    # np.random.seed(seed_num)
    random.seed(seed_num)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed_num)


set_seed(114514)


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="0",
        type=int,
        help="Running device. E.g `--device 0`, if using cpu, set `--device -1`",
    )
    parser.add_argument(
        "--data_name",
        default="mHSC_E",
        type=str,
        help="The dataset name: mHSC_E, mHSC_GM, mHSC_L, timeData/hesc1...",
    )
    parser.add_argument(
        "--rpkm_path",
        default="../data_evaluation/single_cell_type/mHSC-E/ExpressionData.csv",
        type=str,
    )
    parser.add_argument(
        "--label_path",
        default="../data_evaluation/single_cell_type/training_pairsmHSC_E.txt",
        type=str,
    )
    parser.add_argument(
        "--divide_path",
        default="../data_evaluation/single_cell_type/training_pairsmHSC_E.txtTF_divide_pos.txt",
        type=str,
    )

    parser.add_argument(
        "--TF_num",
        default=18,
        type=int,
    )
    parser.add_argument("--is_time", default=False, action="store_true")
    parser.add_argument("--is_h5", default=False, action="store_true")
    parser.add_argument(
        "--gene_list_path",
        default="../data_evaluation/single_cell_type/mHSC_E_geneName_map.txt",
        type=str,
    )
    parser.add_argument("--TF_random", default=False, action="store_true")
    args = parser.parse_args()
    args.device = (
        torch.device(args.device) if args.device >= 0 else torch.device("cpu")
    )

    return args


device = "cuda:1"


def main():
    TF_num = args.TF_num
    data_name = args.data_name
    save_dir = '../results/' + data_name + '/'

    e = GeneData(args.rpkm_path,
                 args.label_path,
                 args.divide_path,
                 TF_num=args.TF_num,
                 gene_emb_path='../embeddings/' + data_name + '/gene_embedding.npy',
                 cell_emb_path='../embeddings/' + data_name + '/cell_embedding.npy',
                 istime=args.is_time, gene_list_path=args.gene_list_path,
                 data_name=args.data_name, TF_random=args.TF_random, ish5=args.is_h5)

    cross_file_path = '../data_evaluation/Time_data/DB_pairs_TF_gene/' + data_name.lower() + '_cross_validation_fold_divide.txt'
    cross_index = []
    if args.is_time:
        with open(cross_file_path, 'r') as f:
            for line in f:
                cross_index.append([int(i) for i in line.strip().split(',')])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

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

    for fold in range(1, 4):
        count_set = [0]
        count_setx = 0
        test_index = fold

        test_TF = [i for i in range(int(np.ceil((test_index - 1) * 0.333333 * TF_num)),
                                    int(np.ceil(test_index * 0.333333 * TF_num)))]
        if args.is_time:
            test_TF = cross_index[fold - 1]
        fold_path = save_dir + str(test_TF) + '/'
        if not os.path.exists(fold_path):
            os.makedirs(fold_path)

        train_TF = [j for j in range(TF_num) if j not in test_TF]
        print("test_TF", test_TF)

        train_emb_datas = []
        train_labels = []
        for j in train_TF:
            print(j, e.datas[j].shape, e.labels[j].shape)
            train_emb_datas.append(e.datas[j])
            train_labels.append(e.labels[j])
        train_emb_datas = np.concatenate(train_emb_datas, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)

        train_emb_datas, val_emb_datas, train_labels, val_labels = \
            train_test_split(train_emb_datas,
                             train_labels,
                             test_size=0.2,
                             random_state=42)

        test_emb_datas = []
        test_labels = []
        z = [0]
        z_len = 0
        for j in test_TF:
            test_emb_datas.append(e.datas[j])
            test_labels.append(e.labels[j])
            z_len += len(e.datas[j])
            z.append(z_len)
        np.save(fold_path + 'z.npy', z)
        z_all.append(z)
        test_emb_datas = np.concatenate(test_emb_datas, axis=0)
        test_labels = np.concatenate(test_labels, axis=0)

        print('train', train_emb_datas.shape, train_labels.shape)
        print('val', val_emb_datas.shape, val_labels.shape)
        print('test', test_emb_datas.shape, test_labels.shape)
        print('=' * 100)

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(train_emb_datas).float(),
                                           torch.from_numpy(train_labels).float()),
            batch_size=512, shuffle=True, num_workers=14, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(val_emb_datas).float(),
                                           torch.from_numpy(val_labels).float()),
            batch_size=512, shuffle=False, num_workers=14)

        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(test_emb_datas).float(),
                                           torch.from_numpy(test_labels).float()),
            batch_size=512, shuffle=False, num_workers=14)

        model = LinearNet().to(device)
        test_model = LinearNet().to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
        criterion = nn.BCEWithLogitsLoss()
        val_acc_best = 0.0
        model_weight_best = None
        early_stop = 0
        stop_num = 10

        for epoch in range(epochs):
            model.train()
            train_acc_sum = 0.0
            train_loss = 0.0

            for batch_idx, (emb_data, target) in enumerate(train_loader):
                emb_data = emb_data.to(device)
                target = target.to(device)

                optimizer.zero_grad()
                output = model(emb_data)
                output = output.squeeze()

                loss = criterion(output, target)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                pred = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output))

                train_acc_sum += (pred == target).sum().item()
            # scheduler.step()
            train_acc = train_acc_sum / len(train_loader.dataset)
            train_loss = train_loss / len(train_loader)

            model.eval()
            val_acc_sum = 0.0
            val_loss = 0.0
            pre = []
            label = []
            with torch.no_grad():
                for batch_idx, (emb_data, target) in enumerate(val_loader):
                    emb_data = emb_data.to(device)
                    target = target.to(device).squeeze()

                    output = model(emb_data)
                    output = output.squeeze()
                    loss = criterion(output, target)

                    val_loss += loss.item()
                    pred = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output))

                    pre.extend(output.cpu().numpy())
                    label.extend(target.cpu().numpy())

                    val_acc_sum += (pred == target).sum().item()

            val_acc = val_acc_sum / len(val_loader.dataset)
            val_loss = val_loss / len(val_loader)

            one_hot_label = label
            val_auc = roc_auc_score(one_hot_label, pre)
            precision, recall, thresholds = precision_recall_curve(one_hot_label, pre, pos_label=1)
            val_ap = auc(recall, precision)
            if val_acc > val_acc_best:
                model_weight_best = model.state_dict()
                val_acc_best = val_acc
                early_stop = 0
            else:
                early_stop += 1

            print('Epoch: {}, Train Loss: {:.4f},Train Acc: {:.4f}, '
                  'Val Loss: {:.4f}, Val Acc: {:.4f}, Val AUC: {:.4f}, Val AP: {:.4f}'.format(
                epoch, train_loss, train_acc, val_loss, val_acc, val_auc, val_ap
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
        label = []
        with torch.no_grad():
            for batch_idx, (emb_data, target) in enumerate(test_loader):
                emb_data = emb_data.to(device)
                target = target.to(device).squeeze()

                output = test_model(emb_data)
                output = output.squeeze()
                loss = criterion(output, target)

                test_loss += loss.item()
                pred = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output))
                pre.extend(output.cpu().numpy())
                label.extend(target.cpu().numpy())

                test_acc_sum += (pred == target).sum().item()
        test_acc = test_acc_sum / len(test_loader.dataset)
        test_loss = test_loss / len(test_loader)

        one_hot_label = label
        test_auc = roc_auc_score(one_hot_label, pre)
        precision, recall, thresholds = precision_recall_curve(one_hot_label, pre, pos_label=1)
        test_ap = auc(recall, precision)

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
    one_hot_label_all = label_all
    final_auc = roc_auc_score(one_hot_label_all, pre_all)

    precision, recall, thresholds = precision_recall_curve(one_hot_label_all, pre_all, pos_label=1)
    final_ap = auc(recall, precision)
    pre_all = np.where(np.asarray(pre_all) > 0.5, 1, 0)

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
            test_auc_i = roc_auc_score(test_true_i, test_predict_i)
            precision, recall, thresholds = precision_recall_curve(test_true_i, test_predict_i, pos_label=1)
            test_ap_i = auc(recall, precision)
            test_predict_i = np.where(np.asarray(test_predict_i) > 0.5, 1, 0)
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


if __name__ == '__main__':
    args = config()
    print(args)
    main()
