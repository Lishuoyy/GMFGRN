'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Design training and test process
'''
import world
import numpy as np
import torch
import utils
import dataloader
from pprint import pprint
from utils import timer
from time import time
from tqdm import tqdm
import model
import multiprocessing
from sklearn.metrics import roc_auc_score


CORES = multiprocessing.cpu_count() // 2


def BPR_train_original(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class

    with timer(name="Sample"):
        S = utils.UniformSample_original(dataset)

    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()
    scores = torch.Tensor(S[:, 3]).float()

    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    scores = scores.to(world.device)

    # train_dataset = torch.utils.data.TensorDataset(users, posItems, negItems, scores)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=world.config['bpr_batch_size'],
    #                                            shuffle=True, num_workers=14)
    users, posItems, negItems, scores = utils.shuffle(users, posItems, negItems, scores)

    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.
    reg_aver_loss = 0.
    bpr_aver_loss = 0.
    mse_aver_loss = 0.
    acc_aver = 0.
    acc_aver1 = 0.
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg,
          batch_score)) in tqdm(enumerate(utils.minibatch(users, posItems, negItems, scores, batch_size=world.config['bpr_batch_size'])),
                                total=len(users) // world.config['bpr_batch_size'] + 1):
        batch_users = batch_users.to(world.device)
        batch_pos = batch_pos.to(world.device)
        batch_neg = batch_neg.to(world.device)
        batch_score = batch_score.to(world.device)

        cri, reg_loss, bpr_loss, mse_loss, acc_sum, acc_sum1= bpr.stageOne(batch_users, batch_pos, batch_neg, batch_score)
        aver_loss += cri
        reg_aver_loss += reg_loss
        bpr_aver_loss += bpr_loss
        mse_aver_loss += mse_loss
        acc_aver += acc_sum
        acc_aver1 += acc_sum1
        if world.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
    aver_loss = aver_loss / total_batch
    reg_aver_loss = reg_aver_loss / total_batch
    bpr_aver_loss = bpr_aver_loss / total_batch
    mse_aver_loss = mse_aver_loss / total_batch
    acc_aver = acc_aver / (2 * len(users))
    acc_aver1 = acc_aver1 / (len(users))

    time_info = timer.dict()
    timer.zero()
    return f"loss{aver_loss:.4f}-reg_loss{reg_aver_loss:.4f}-bpr_loss{bpr_aver_loss:.4f}-" \
           f"mse_loss{mse_aver_loss:.4f}-acc_aver{acc_aver:.4f}-acc_aver1{acc_aver1:.4f}-{time_info}"


def BPR_train_gene(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class

    # with timer(name="Sample"):
    #     S = utils.UniformSample_original(dataset)

    # users = torch.Tensor(S[:, 0]).long()
    # posItems = torch.Tensor(S[:, 1]).long()
    # negItems = torch.Tensor(S[:, 2]).long()
    # scores = torch.Tensor(S[:, 3]).float()
    #
    # users = users.to(world.device)
    # posItems = posItems.to(world.device)
    # negItems = negItems.to(world.device)
    # scores = scores.to(world.device)
    # users, posItems, negItems, scores = utils.shuffle(users, posItems, negItems, scores)

    # 后加的
    train_data, val_data, test_data, _ = dataset.getTrainTest(18)

    users1 = torch.Tensor(train_data[:, 0]).long()
    users2 = torch.Tensor(train_data[:, 1]).long()
    label = torch.Tensor(train_data[:, 2]).float()
    users1 = users1.to(world.device)
    users2 = users2.to(world.device)
    label = label.to(world.device)


    users1, users2, label = utils.shuffle(users1, users2, label)

    total_batch = len(users1) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.
    reg_aver_loss = 0.
    bpr_aver_loss = 0.
    mse_aver_loss = 0.
    acc_aver = 0.
    for (batch_i,
         (batch_users1,
          batch_users2,
          batch_labels)) in enumerate(utils.minibatch(users1,
                                                           users2,
                                                           label,
                                                           batch_size=world.config['bpr_batch_size'])):
        cri, reg_loss, bpr_loss, acc_sum, output = bpr.stageOne2(batch_users1, batch_users2, batch_labels, 'train')
        aver_loss += cri
        reg_aver_loss += reg_loss
        bpr_aver_loss += bpr_loss
        acc_aver += acc_sum
        if world.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users1) / world.config['bpr_batch_size']) + batch_i)
    aver_loss = aver_loss / total_batch
    reg_aver_loss = reg_aver_loss / total_batch
    bpr_aver_loss = bpr_aver_loss / total_batch
    acc_aver = acc_aver / (len(users1))

    time_info = timer.dict()
    timer.zero()
    return f"loss{aver_loss:.4f}-reg_loss{reg_aver_loss:.4f}-bpr_loss{bpr_aver_loss:.4f}-" \
           f"acc_aver{acc_aver:.4f}-{time_info}"


def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue, r, k))
    return {'recall': np.array(recall),
            'precision': np.array(pre),
            'ndcg': np.array(ndcg)}


def Test(dataset, Recmodel, epoch, w=None, multicore=0):
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    testDict: dict = dataset.testDict
    Recmodel: model.LightGCN
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    max_K = max(world.topks)
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {'precision': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks))}
    with torch.no_grad():
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        # auc_record = []
        # ratings = []
        total_batch = len(users) // u_batch_size + 1
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)

            rating = Recmodel.getUsersRating(batch_users_gpu)
            # rating = rating.cpu()
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1 << 10)
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            # aucs = [ 
            #         utils.AUC(rating[i],
            #                   dataset, 
            #                   test_data) for i, test_data in enumerate(groundTrue)
            #     ]
            # auc_record.extend(aucs)
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))
        scale = float(u_batch_size / len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        # results['auc'] = np.mean(auc_record)
        if world.tensorboard:
            w.add_scalars(f'Test/Recall@{world.topks}',
                          {str(world.topks[i]): results['recall'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/Precision@{world.topks}',
                          {str(world.topks[i]): results['precision'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/NDCG@{world.topks}',
                          {str(world.topks[i]): results['ndcg'][i] for i in range(len(world.topks))}, epoch)
        if multicore == 1:
            pool.close()
        print(results)
        return results


def Test1(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None, flag=0):
    Recmodel = recommend_model
    Recmodel.eval()
    bpr: utils.BPRLoss = loss_class

    # with timer(name="Sample"):
    #     S = utils.UniformSample_original(dataset)

    # users = torch.Tensor(S[:, 0]).long()
    # posItems = torch.Tensor(S[:, 1]).long()
    # negItems = torch.Tensor(S[:, 2]).long()
    # scores = torch.Tensor(S[:, 3]).float()
    #
    # users = users.to(world.device)
    # posItems = posItems.to(world.device)
    # negItems = negItems.to(world.device)
    # scores = scores.to(world.device)
    # users, posItems, negItems, scores = utils.shuffle(users, posItems, negItems, scores)

    # 后加的
    train_data, val_data, test_data, fold = dataset.getTrainTest(18)
    if flag == 1:
        test_data = val_data
    users1 = torch.Tensor(test_data[:, 0]).long()
    users2 = torch.Tensor(test_data[:, 1]).long()
    label = torch.Tensor(test_data[:, 2]).float()
    users1 = users1.to(world.device)
    users2 = users2.to(world.device)
    label = label.to(world.device)
    users1, users2, label = utils.shuffle(users1, users2, label)

    total_batch = len(users1) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.
    reg_aver_loss = 0.
    bpr_aver_loss = 0.
    mse_aver_loss = 0.
    acc_aver = 0.
    pred = []
    for (batch_i,
         (batch_users1,
          batch_users2,
          batch_labels)) in enumerate(utils.minibatch(users1,
                                                           users2,
                                                           label,
                                                           batch_size=world.config['bpr_batch_size'])):
        cri, reg_loss, bpr_loss, acc_sum, output = bpr.stageOne2(batch_users1, batch_users2, batch_labels, 'test')
        aver_loss += cri
        reg_aver_loss += reg_loss
        bpr_aver_loss += bpr_loss
        acc_aver += acc_sum
        pred.extend(output.detach().cpu().numpy())
        if world.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users1) / world.config['bpr_batch_size']) + batch_i)
    aver_loss = aver_loss / total_batch
    reg_aver_loss = reg_aver_loss / total_batch
    bpr_aver_loss = bpr_aver_loss / total_batch
    acc_aver = acc_aver / (len(users1))
    one_hot_label = torch.eye(2)[label.long(), :]
    auc = roc_auc_score(one_hot_label.detach().cpu().numpy(), pred)
    time_info = timer.dict()
    timer.zero()
    if flag == 0:
        np.save('result/pred'+str(fold)+'.npy', pred)
        np.save('result/label'+str(fold)+'.npy', one_hot_label.detach().cpu().numpy())
    return f"loss{aver_loss:.4f}-reg_loss{reg_aver_loss:.4f}-bpr_loss{bpr_aver_loss:.4f}-" \
           f"acc_aver{acc_aver:.4f}-AUC{auc:.4f}-{time_info}", acc_aver


def geneTest(dataset, Recmodel, epoch, w=None, multicore=0):
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    testDict: dict = dataset.testDict
    Recmodel: model.LightGCN
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    max_K = max(world.topks)

    with torch.no_grad():
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        # auc_record = []
        # ratings = []
        total_batch = len(users) // u_batch_size + 1
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)

            rating = Recmodel.getUsersRating(batch_users_gpu)
            # rating = rating.cpu()
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1 << 10)
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            # aucs = [
            #         utils.AUC(rating[i],
            #                   dataset,
            #                   test_data) for i, test_data in enumerate(groundTrue)
            #     ]
            # auc_record.extend(aucs)
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
