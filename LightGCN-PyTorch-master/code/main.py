import os

import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
import pandas as pd

# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset

Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

# init tensorboard
if world.tensorboard:
    w: SummaryWriter = SummaryWriter(
        join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

# expr_file = "../../contrastive-predictive-coding-master/data_evaluation/single_cell_type/mHSC-E/ExpressionData.csv"
# label_path = "../../contrastive-predictive-coding-master/data_evaluation/single_cell_type/training_pairsmHSC_E.txt"

expr_file = '../data_evaluation/Time_data/scRNA_expression_data/mesc2_expression_data/'
label_path = '../data_evaluation/Time_data/database/mesc2_gene_pairs_400.txt'
istime = True

def testEPR(model, expr_file=expr_file, label_path=label_path):
    all_users, all_items = model.computer()
    all_users = all_users.detach().cpu()
    all_items = all_items.detach().cpu()

    users_emb = model.embedding_user.weight.data.detach().cpu()
    items_emb = model.embedding_item.weight.data.detach().cpu()

    user_pos = dataset.allPos

    A = torch.matmul(all_users, all_users.T).cpu().numpy()
    A = torch.sigmoid(torch.from_numpy(A)).numpy()

    truth_edges, Evaluate_Mask = getTrueEdges(expr_file, label_path, 1)
    ep, epr = evaluate(A, truth_edges, Evaluate_Mask)
    # return ep, epr
    print('Test EPR', epr)

# testEPR(Recmodel)
def testACC(model, label_path=label_path):
    all_users, all_items = model.computer()

    all_users = all_users.detach().cpu()
    all_items = all_items.detach().cpu()
    user_pos = dataset.allPos
    # for i in range(len(user_pos)):
    #     user_items = user_pos[i]
    #     user_items_emb = all_items[user_items]
    #     user_emb = torch.mean(user_items_emb, dim=0)
    #     # print(user_emb.shape)
    #     # print(all_users[i].shape)
    #     all_users[i] = (user_emb + all_users[i]) / 2.0
    A = torch.matmul(all_users, all_users.T).cpu().numpy()
    A = torch.sigmoid(torch.from_numpy(A)).numpy()
    gene_to_idx = dataset.gene_to_idx
    df = get_true_edge(label_path, 0)
    users = df['Gene1']
    users = [gene_to_idx[gene.lower()] for gene in users]
    items = df['Gene2']
    items = [gene_to_idx[gene.lower()] for gene in items]
    labels = df['Labels']

    test_dataset = torch.utils.data.TensorDataset(torch.LongTensor(users), torch.LongTensor(items),
                                                  torch.LongTensor(labels))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4096, shuffle=False)
    acc = 0
    for user, item, label in test_loader:
        user = user.numpy()
        item = item.numpy()
        label = label.numpy()
        pred = A[user, item]
        pred = np.where(pred > 0.5, 1, 0)
        acc += np.sum(pred == label)
    acc = acc / len(df)
    print('Test ACC', acc)


def get_true_edge(label_path, ctype):
    s = open(label_path)
    gene1s = []
    gene2s = []
    labels = []
    for line in s:
        separation = line.split('\t')
        geneA_name, geneB_name, label = separation[0], separation[1], separation[2]
        geneA_name = geneA_name.upper()
        geneB_name = geneB_name.upper()
        if ctype == 0:
            if int(label) != 2:
                gene1s.append(geneA_name)
                gene2s.append(geneB_name)
                labels.append(int(label))
        elif ctype == 1:
            if int(label) == 1:
                gene1s.append(geneA_name)
                gene2s.append(geneB_name)
                labels.append(int(label))
    s.close()
    # to csv
    df = pd.DataFrame({'Gene1': gene1s, 'Gene2': gene2s, 'Labels': labels})
    return df


def getTrueEdges(expr_file, label_path, ctype):
    Ground_Truth = get_true_edge(label_path, ctype)
    if not istime:
        df = pd.read_csv(expr_file, header='infer', index_col=0)
    else:
        time_h5 = []
        files = os.listdir(expr_file)
        for i in range(len(files)):
            time_pd = pd.read_hdf(expr_file + 'RPKM_' + str(i) + '.h5', key='/RPKMs')
            # print(time_pd)
            # exit()
            time_h5.append(time_pd)
        train_data = pd.concat(time_h5, axis=0, ignore_index=True)
        df = train_data.T

    rpkm = df.T

    TF = set(Ground_Truth['Gene1'])
    # print('len TF', len(TF))
    All_gene = set(Ground_Truth['Gene1']) | set(Ground_Truth['Gene2'])
    # print('len All_gene', len(All_gene))
    num_genes, num_nodes = rpkm.shape[1], rpkm.shape[0]
    Evaluate_Mask = np.zeros([num_genes, num_genes])
    TF_mask = np.zeros([num_genes, num_genes])
    for i, item in enumerate(rpkm.columns):
        for j, item2 in enumerate(rpkm.columns):
            if i == j:
                continue
            if item2 in TF and item in All_gene:
                Evaluate_Mask[i, j] = 1
                # print(i, j)
            if item2 in TF:
                TF_mask[i, j] = 1
    # print(Evaluate_Mask)
    truth_df = pd.DataFrame(np.zeros([num_genes, num_genes]), index=rpkm.columns, columns=rpkm.columns)
    for i in range(Ground_Truth.shape[0]):
        truth_df.loc[Ground_Truth.iloc[i, 1], Ground_Truth.iloc[i, 0]] = 1
    # print(truth_df)

    A_truth = truth_df.values
    idx_rec, idx_send = np.where(A_truth)
    truth_edges = set(zip(idx_send, idx_rec))
    return truth_edges, Evaluate_Mask


def evaluate(A, truth_edges, Evaluate_Mask):
    num_nodes = A.shape[0]
    num_truth_edges = len(truth_edges)  # 4526
    A = abs(A)
    if Evaluate_Mask is None:
        Evaluate_Mask = np.ones_like(A) - np.eye(len(A))
    A = A * Evaluate_Mask
    A_val = list(np.sort(abs(A.reshape(-1, 1)), 0)[:, 0])
    A_val.reverse()

    cutoff_all = A_val[num_truth_edges]  # A_val[4526]

    A_indicator_all = np.zeros([num_nodes, num_nodes])
    A_indicator_all[abs(A) > cutoff_all] = 1
    idx_rec, idx_send = np.where(A_indicator_all)
    A_edges = set(zip(idx_send, idx_rec))
    overlap_A = A_edges.intersection(truth_edges)
    # print(len(overlap_A))
    # print(num_truth_edges ** 2)
    # print(np.sum(Evaluate_Mask))
    # print((num_truth_edges ** 2) / np.sum(Evaluate_Mask))
    # exit()
    return len(overlap_A), 1. * len(overlap_A) / ((num_truth_edges ** 2) / np.sum(Evaluate_Mask))

best_val_acc = 0.0
early_stopping = 0
stop_num = 10
try:
    for epoch in range(world.TRAIN_epochs):
        start = time.time()
        output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k, w=w)
        # val_info, val_acc = Procedure.Test1(dataset, Recmodel, bpr, epoch, neg_k=Neg_k, w=w, flag=1)
        # if val_acc > best_val_acc:
        #     best_val_acc = val_acc
        #     early_stopping = 0
        # else:
        #     early_stopping += 1
        print(f'EPOCH[{epoch + 1}/{world.TRAIN_epochs}] {output_information}')
        # print(f'VAL {val_info}')
        # if True:
        #     cprint("[TEST]")
        #     testEPR(Recmodel)
        #     testACC(Recmodel)
            # test_info, _ = Procedure.Test1(dataset, Recmodel, bpr, epoch, neg_k=Neg_k, w=w, flag=0)
            # print(f'TEST {test_info}')
        if early_stopping > stop_num:
            break
        torch.save(Recmodel.state_dict(), weight_file)
finally:
    if world.tensorboard:
        w.close()
