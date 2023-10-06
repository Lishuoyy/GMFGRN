from __future__ import print_function
# import keras
# from keras.preprocessing.image import ImageDataGenerator
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation, Flatten
# from keras.layers import Conv2D, MaxPooling2D
# from keras.optimizers import SGD
# from keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy import interp
# import seaborn as sns
import pandas as pd
import torch

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# sns.set_style("whitegrid")


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
# test_dir = '../Ablation/time_points/4'
test_dir = '../modelResult/TimeData/myModel_3/hesc2_result_1'
save_dir = test_dir  ### the final performance folder
length_TF = 98

cross_file_path = '../data_evaluation/Time_data/DB_pairs_TF_gene/hesc2_cross_validation_fold_divide.txt'


cross_index = []
with open(cross_file_path, 'r') as f:
    for line in f:
        cross_index.append([int(i) for i in line.strip().split(',')])

# timedata
gene_pairs_path = 'hesc2_representation/gene_pairs.txt'
divide_pos_path = 'hesc2_representation/divide_pos.txt'
gene_pairs = []
divide_pos = []
with open(gene_pairs_path, 'r') as f:
    for line in f:
        gene_pairs.append(line.strip().split('\t')[0])

with open(divide_pos_path, 'r') as f:
    for line in f:
        divide_pos.append(int(line.strip()))
print(divide_pos)
#

whole_data_TF = [i for i in range(length_TF)]
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
    # plt.grid()
AUC_set = []
tprs = []
mean_fpr = np.linspace(0, 1, 100)

y_testy = np.empty([0])
y_predicty = np.empty([0, 1])
# count_setx = pd.read_table('/home/yey3/nn_project2/data/human_brain/pathways/kegg/unique_rand_labelx_num.txt',header=None)
# count_set = [i[0] for i in np.array(count_setx)]
count_set = [0]
tf_name = []
tf_name_len = []
for test_indel in range(1, 4):
    # test_TF = [i for i in range(int(np.ceil((test_indel - 1) * 0.333333 * length_TF)),
    #                             int(np.ceil(test_indel * 0.333333 * length_TF)))]
    test_TF = cross_index[test_indel - 1]

    print(test_TF, str(test_TF))
    # (x_testx, y_testx, count_setz) = load_data(test_TF, data_path)
    # print(len(y_testx),count_setz)
    y_predictyz = np.load(
        test_dir + '/' + str(test_TF) + '/end_y_predict.npy')  ### trained model for each fold cross validation
    y_testyz = np.load(
        test_dir + '/' + str(test_TF) + '/end_y_test.npy')  ### trained model for each fold cross validation
    count_setz = np.load(test_dir + '/' + str(test_TF) + '/z.npy')
    ## timedata
    for i in test_TF:
        start_index = divide_pos[i]
        end_index = divide_pos[i + 1]
        tf_name_len.append(len(gene_pairs[start_index:end_index]))
        # print(start_index, end_index)
        # print(gene_pairs[start_index:end_index])
        # exit()
        tf_name.append(gene_pairs[start_index])
    #
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
print(tf_name_len)
print(len(tf_name))
print(tf_name)
df = pd.DataFrame({'tf_name': tf_name})
df.to_excel('../modelResult/TF_name.xlsx', index=False)
