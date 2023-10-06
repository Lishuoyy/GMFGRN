from __future__ import print_function
import os
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy import interp
import pandas as pd
import torch

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def load_data(indel_list, data_path,
              num_of_pair_ratio=1):  # cell type specific  ## random samples for reactome is not enough, need borrow some from keggp
    import numpy as np
    xxdata_list = []
    yydata = []
    zzdata = []
    count_set = [0]
    count_setx = 0
    for i in indel_list:  # len(h_tf_sc)): xdata_tf0.npy
        #     xdata = np.load(data_path + '/' + 'xdata_tf' + str(i) + '.npy')
        #     ydata = np.load(data_path + '/' + 'ydata_tf' + str(i) + '.npy')
        xdata = np.load(data_path + str(i) + '_xdata.npy')
        ydata = np.load(data_path + str(i) + '_ydata.npy')
        # zdata = np.load(data_path + str(i) + '_zdata.npy')

        num_of_pairs = round(num_of_pair_ratio * len(ydata))
        all_k_list = list(range(len(ydata)))
        select_k_list = all_k_list[0:num_of_pairs]
        for k in select_k_list:
            xxdata_list.append(xdata[k, :, :, :])
            yydata.append(ydata[k])
            # zzdata.append(zdata[k])
        count_setx = count_setx + len(ydata)
        count_set.append(count_setx)
        print(i, len(ydata))
    yydata_array = np.array(yydata)
    yydata_x = yydata_array.astype('int')
    print(np.array(xxdata_list).shape)
    return ((np.array(xxdata_list), yydata_x, count_set))


# data_path = '../contrastive-predictive-coding-master/my/mesc_1_representation/' ### 3D NEPDF folder
data_path = "dendritic_representation/"
# test_dir = '../Ablation/time_points/4'
test_dir = '../modelResult/myModel_3/dendritic_result_1'
save_dir = test_dir  ### the final performance folder
length_TF = 16

# cross_file_path = '../data_evaluation/Time_data/DB_pairs_TF_gene/hesc1_cross_validation_fold_divide.txt'
# cross_index = []
# with open(cross_file_path, 'r') as f:
#     for line in f:
#         cross_index.append([int(i) for i in line.strip().split(',')])
whole_data_TF = [i for i in range(length_TF)]
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
    # plt.grid()
AUC_set = []
s = open(save_dir + '/whole_RPKM_AUCs1+2.txt', 'w')
tprs = []
mean_fpr = np.linspace(0, 1, 100)

y_testy = np.empty([0])
y_predicty = np.empty([0, 1])
# count_setx = pd.read_table('/home/yey3/nn_project2/data/human_brain/pathways/kegg/unique_rand_labelx_num.txt',header=None)
# count_set = [i[0] for i in np.array(count_setx)]
count_set = [0]
for test_indel in range(1, 4):
    test_TF = [i for i in range(int(np.ceil((test_indel - 1) * 0.333333 * length_TF)),
                                int(np.ceil(test_indel * 0.333333 * length_TF)))]
    # test_TF = cross_index[test_indel - 1]
    print(test_TF, str(test_TF))
    # (x_testx, y_testx, count_setz) = load_data(test_TF, data_path)
    # print(len(y_testx),count_setz)
    y_predictyz = np.load(
        test_dir + '/' + str(test_TF) + '/end_y_predict.npy')  ### trained model for each fold cross validation
    y_testyz = np.load(
        test_dir + '/' + str(test_TF) + '/end_y_test.npy')  ### trained model for each fold cross validation
    count_setz = np.load(test_dir + '/' + str(test_TF) + '/z.npy')
    y_testy = np.concatenate((y_testy, y_testyz), axis=0)
    # print(y_predicty.shape, y_predictyz.shape)
    # print(y_predicty)
    # exit()
    y_predictyz = np.expand_dims(y_predictyz, axis=1)
    y_predicty = np.concatenate((y_predicty, y_predictyz), axis=0)
    count_set = count_set + [i + count_set[-1] if len(count_set) > 0 else i for i in count_setz[1:]]
    ############
print(len(count_set))
print(count_set)
print(y_testy)
# exit()
# y_testy = y_testy.astype(np.long)
# y_testy = np.eye(2)[y_testy, :]
# print(y_testy)
# exit()
# y_predicty = torch.sigmoid(torch.from_numpy(y_predicty)).numpy()
# y_predicty = np.where(y_predicty > 0.5, 1, 0)
# cm = metrics.confusion_matrix(y_testy, y_predicty)
# fp = cm[0][1]
# print(fp)
# exit()
###############whole performance

##################################
fig = plt.figure(figsize=(5, 5))
plt.plot([0, 1], [0, 1])
total_pair = 0
total_auc = 0
print(y_predicty.shape)
############
for jj in range(len(count_set) - 1):  # len(count_set)-1):
    if count_set[jj] < count_set[jj + 1]:
        print(test_indel, jj, count_set[jj], count_set[jj + 1])
        current_pair = count_set[jj + 1] - count_set[jj]
        total_pair = total_pair + current_pair
        y_test = y_testy[count_set[jj]:count_set[jj + 1]]
        y_predict = y_predicty[count_set[jj]:count_set[jj + 1]]
        # Score trained model.
        # print(y_test.shape, y_predict.shape)
        # exit()
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        # Print ROC curve
        plt.plot(fpr, tpr, color='0.5', lw=0.001, alpha=.2)
        auc = np.trapz(tpr, fpr)
        s.write(str(jj) + '\t' + str(count_set[jj]) + '\t' + str(count_set[jj + 1]) + '\t' + str(auc) + '\n')
        print('AUC:', auc)
        AUC_set.append(auc)
        total_auc = total_auc + auc * current_pair

fpr, tpr, thresholds = metrics.roc_curve(y_testy, y_predicty, pos_label=1)
auc = np.trapz(tpr, fpr)
s.write('final AUROC' + '\t' + str(auc) + '\n')
s.close()
np.save(save_dir + '/AUROC_set.npy', AUC_set)
mean_tpr = np.median(tprs, axis=0)
mean_tpr[-1] = 1.0
per_tpr = np.percentile(tprs, [25, 50, 75], axis=0)
mean_auc = np.trapz(mean_tpr, mean_fpr)
print("mean auc:", mean_auc)
plt.plot(mean_fpr, mean_tpr, 'k', lw=3, label='median ROC')
plt.title("{:.4f}".format(mean_auc), fontsize=15)
plt.fill_between(mean_fpr, per_tpr[0, :], per_tpr[2, :], color='g', alpha=.2, label='quantile')
plt.plot(mean_fpr, per_tpr[0, :], 'g', lw=3, alpha=.2)
plt.legend(loc='lower right', fontsize=15)
plt.ylim([0, 1])
plt.xlim([0, 1])
plt.grid()
plt.xlabel('FP', fontsize=15)
plt.ylabel('TP', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig(save_dir + '/whole_kegg_ROCs1+2_percentile.pdf')
del fig
fig = plt.figure(figsize=(3, 3))
plt.hist(AUC_set, bins=50)
plt.savefig(save_dir + '/whole_kegg_ROCs1+2_hist.pdf')
del fig
fig = plt.figure(figsize=(3, 3))
plt.boxplot(AUC_set)
plt.savefig(save_dir + '/whole_kegg_ROCs1+2_box.pdf')
del fig
############################
# AUPRC
AUC_set = []
s = open(save_dir + '/whole_RPKM_AUPRCs1+2.txt', 'w')
############
for jj in range(len(count_set) - 1):  # len(count_set)-1):
    if count_set[jj] < count_set[jj + 1]:
        print(test_indel, jj, count_set[jj], count_set[jj + 1])
        current_pair = count_set[jj + 1] - count_set[jj]
        total_pair = total_pair + current_pair
        y_test = y_testy[count_set[jj]:count_set[jj + 1]]
        y_predict = y_predicty[count_set[jj]:count_set[jj + 1]]
        # Score trained model.
        precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_predict, pos_label=1)
        auc = metrics.auc(recall, precision)
        s.write(str(jj) + '\t' + str(count_set[jj]) + '\t' + str(count_set[jj + 1]) + '\t' + str(auc) + '\n')
        print('AUC:', auc)
        AUC_set.append(auc)

precision, recall, thresholds = metrics.precision_recall_curve(y_testy, y_predicty, pos_label=1)
auc = metrics.auc(recall, precision)
s.write('final AUPRC' + '\t' + str(auc) + '\n')
s.close()
np.save(save_dir + '/AUPRC_set.npy', AUC_set)