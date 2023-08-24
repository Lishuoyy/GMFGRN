import torch
import numpy as np

import pandas as pd

expr_file = "../../contrastive-predictive-coding-master/data_evaluation/single_cell_type/mHSC-E/ExpressionData.csv"
label_path = "../../contrastive-predictive-coding-master/data_evaluation/single_cell_type/training_pairsmHSC_E.txt"


def testEPR(model, expr_file=expr_file, label_path=label_path):
    users_emb = model.user_embeds.weight.data.detach().cpu()
    items_emb = model.item_embeds.weight.data.detach().cpu()
    all_users = users_emb
    all_items = items_emb
    #
    # np.save('out/mHSC_GM_all_users_w5.npy', all_users)
    # np.save('out/mHSC_GM_all_items_w5.npy', all_items)
    # np.save('out/mHSC_GM_users_emb_w5.npy', users_emb)
    # np.save('out/mHSC_GM_items_emb_w5.npy', items_emb)
    # exit()
    # user_pos = dataset.allPos
    # for i in range(len(user_pos)):
    #     user_items = user_pos[i]
    #     user_items_emb = all_items[user_items]
    #     user_emb = torch.mean(user_items_emb, dim=0)
    #     # print(user_emb.shape)
    #     # print(all_users[i].shape)
    #     all_users[i] = (user_emb + all_users[i]) / 2.0

    A = torch.matmul(all_users, all_users.T).cpu().numpy()
    A = torch.sigmoid(torch.from_numpy(A)).numpy()

    truth_edges, Evaluate_Mask = getTrueEdges(expr_file, label_path, 1)
    ep, epr = evaluate(A, truth_edges, Evaluate_Mask)
    # return ep, epr
    print('Test EPR', epr)


def testACC(model, gene_to_idx, label_path=label_path):
    users_emb = model.user_embeds.weight.data.detach().cpu()
    items_emb = model.item_embeds.weight.data.detach().cpu()
    all_users = users_emb
    all_items = items_emb
    # user_pos = dataset.allPos
    # for i in range(len(user_pos)):
    #     user_items = user_pos[i]
    #     user_items_emb = all_items[user_items]
    #     user_emb = torch.mean(user_items_emb, dim=0)
    #     # print(user_emb.shape)
    #     # print(all_users[i].shape)
    #     all_users[i] = (user_emb + all_users[i]) / 2.0
    A = torch.matmul(all_users, all_users.T).cpu().numpy()
    A = torch.sigmoid(torch.from_numpy(A)).numpy()

    df = get_true_edge(label_path, 0)
    users = df['Gene1']
    users = [gene_to_idx[gene] for gene in users]
    items = df['Gene2']
    items = [gene_to_idx[gene] for gene in items]
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

    df = pd.read_csv(expr_file, header='infer', index_col=0)

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
