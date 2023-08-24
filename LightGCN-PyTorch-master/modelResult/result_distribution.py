import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve


# CNNC_path = 'CNNC/'
# bonemarrow, dendritic, mesc1 AUROC and AUPRC
def firstPlt(model_set, model_path, data_set, data_name, colors, save_name,index, flag):
    data_len = len(data_set)
    box_position = []
    row_index = 1
    for i in range(1, len(model_set) + 1):
        sublist = []
        for j in range(data_len):
            sublist.append(row_index + j * (((len(model_set) - 1) * 0.3) + 0.6))
        box_position.append(sublist)
        row_index += 0.22
    print(box_position)
    # exit()
    model_num = len(model_set)
    model_value = [[] for i in range(model_num)]

    for i, data in enumerate(data_set):
        for j in range(len(model_path)):
            if j == len(model_path) - 1:
                if flag == 1:
                    model_value[j].append(np.load(model_path[j] + data + '_result_final/' + 'AUROC_set.npy'))
                else:
                    model_value[j].append(np.load(model_path[j] + data + '_result_final/' + 'AUPRC_set.npy'))
            else:
                if flag == 1:
                    model_value[j].append(np.load(model_path[j] + data + '_result/' + 'AUROC_set.npy'))
                else:
                    model_value[j].append(np.load(model_path[j] + data + '_result/' + 'AUPRC_set.npy'))
            # if flag == 1:
            #     model_value[j].append(np.load(model_path[j] + data + '/' + 'AUROC_set.npy'))
            # else:
            #     model_value[j].append(np.load(model_path[j] + data + '/' + 'AUPRC_set.npy'))

    value = np.array(model_value).T
    # 保留 3 位小数
    value_1 = np.stack(value[3])
    print(value_1)
    print(value_1.shape)
    print(data_name)
    print(model_set)

    # 存储为 excel
    df = pd.DataFrame(value_1.T, columns=model_set)
    df.to_excel('time_AUPRC1.xlsx')
    exit()

    # # TF_name
    path = "mesc1_representation"
    df_name = pd.read_csv('../code/'+path+'/gene_pairs.txt', sep='\t', header=None)
    df_index = pd.read_csv('../code/'+path+'/divide_pos.txt', sep='\t', header=None)
    name = df_name[0].values.reshape(-1)
    index = df_index.values.reshape(-1)
    index = index[:-1]
    tfs = name[index]
    print(name)
    print(name.shape)
    print(index)
    print(index.shape)
    print(tfs)
    assert len(tfs) == np.unique(tfs).shape[0]
    df_tfs_name = pd.DataFrame(tfs, columns=['TF_name'])
    df_tfs_name.to_excel('TF_name.xlsx', index=False)
    exit()
    plt.rc('font', size=16)
    # plt.grid(axis='y')
    fig = plt.figure(figsize=(12, 4))
    # 分组箱线图，行为数据集data_set,  label为3个模型方法名称
    ax,ax1 = fig.subplots(1, 2)
    # AUROC
    boxes = []

    for i in range(model_num):
        boxes.append(ax.boxplot(model_value[i], positions=box_position[i], widths=0.2, patch_artist=True,
                                boxprops=dict(facecolor=colors[i], color='black'),
                                whiskerprops=dict(color='black'), sym='.',
                                medianprops={'color': 'black'}))

    # box1 = ax.boxplot(model_auroc[0], positions=box_position[0], widths=0.2, patch_artist=True,
    #                   boxprops=dict(facecolor='red', color='black'), whiskerprops=dict(color='black'))
    # box2 = ax.boxplot(model_auroc[0], positions=box_position[1], widths=0.2, patch_artist=True,
    #                   boxprops=dict(facecolor='green', color='black'), whiskerprops=dict(color='black'))
    # box3 = ax.boxplot(model_auroc[0], positions=box_position[2], widths=0.2, patch_artist=True,
    #                   boxprops=dict(facecolor='blue', color='black'), whiskerprops=dict(color='black'))
    # 添加标签
    # ax.grid(axis='y')
    ax.set_xticks(box_position[len(box_position) // 2])
    ax.set_xticklabels(data_name, rotation=15,)
    # ax.text(-0.25, 1.15, index, transform=ax.transAxes, fontsize=24, fontweight='bold', va='top', ha='left')

    # ax.set_xlabel('Dataset')
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    if flag==1:
        ax.set_ylabel('AUROC')
    else:
        ax.set_ylabel('AUPRC')

    # AUPRC
    # boxes1 = []
    # for i in range(model_num):
    #     boxes1.append(ax1.boxplot(model_auprc[i], positions=box_position[i], widths=0.2, patch_artist=True,
    #                               boxprops=dict(facecolor=colors[i], color='black'), whiskerprops=dict(color='black'),
    #                               sym='.', medianprops={'color': 'black'}))
    # box1 = ax1.boxplot(myModel_auprc, positions=box_position[0], widths=0.2, patch_artist=True,
    #                    boxprops=dict(facecolor='red', color='black'), whiskerprops=dict(color='black'))
    # box2 = ax1.boxplot(DeepDRIM_auprc, positions=box_position[1], widths=0.2, patch_artist=True,
    #                    boxprops=dict(facecolor='green', color='black'), whiskerprops=dict(color='black'))
    # box3 = ax1.boxplot(CNNC_auprc, positions=box_position[2], widths=0.2, patch_artist=True,
    #                    boxprops=dict(facecolor='blue', color='black'), whiskerprops=dict(color='black'))

    # 添加标签
    # ax1.grid(axis='y')
    # ax1.set_xticks(box_position[len(box_position) // 2])
    # ax1.set_xticklabels(data_name, rotation=15,)
    #
    # # ax1.set_xticklabels(model_set, rotation=25, )
    # ax1.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    # ax1.set_ylim()
    # ax1.set_ylabel('AUPRC')

    boxes1 = []
    # print(len(boxes))
    # print(model_num
    #       )
    # exit()
    for i in range(model_num):
        boxes1.append(boxes[i]["boxes"][0])
    legend = ax.legend(boxes1, model_set, loc='upper left', bbox_to_anchor=(1.01, 1))
    # legend = plt.legend([box1["boxes"][0], box2["boxes"][0], box3["boxes"][0]], ['myModel', 'DeepDRIM', 'CNNC'],
    #                     loc='upper left', bbox_to_anchor=(1.01, 1))
    for patch in legend.get_patches():
        patch.set_facecolor(patch.get_facecolor())

    plt.savefig(save_name, bbox_inches='tight')
    plt.show()


# eight data
# model_set = ['GMFGRN', 'DeepDRIM', 'CNNC', 'MI', 'PCC']
# model_set = model_set[::-1]
# model_path = ['myModel_3/', 'DeepDRIM/', 'CNNC/', 'MI/', 'PCC/']
# model_path = model_path[::-1]
# # data_set = ['bonemarrow', 'dendritic', 'mesc_1']
# # data_name = ['boneMarrow', 'dendritic', 'mESC(1)']
# data_set = ['bonemarrow', 'dendritic', 'mesc_1', 'hESC', 'mESC_2', 'mHSC_E', 'mHSC_GM', 'mHSC_L']
# data_name = ['boneMarrow', 'dendritic', 'mESC(1)','hESC', 'mESC(2)', 'mHSC(E)', 'mHSC(GM)', 'mHSC(L)']
# colors = ['#AF58BA', '#009ADE', '#00CD6C', '#F28522', "#E32977"]
# colors = colors[::-1]
# # colors = ['#AF58BA', '#009ADE', '#00CD6C', '#F28522', "#E32977"]
# firstPlt(model_set, model_path, data_set, data_name, colors,
#          save_name='eight_dataset.pdf',index='A',flag=1)

# five data 500
# model_set = ['GMFGRN', 'DeepDRIM', 'CNNC', 'SINCERITIES', 'SCODE', 'GENIE3', 'PIDC']
# model_set = model_set[::-1]
# model_path = ['../modelResult_500/myModel/', '../modelResult_500/DeepDRIM/', '../modelResult_500/CNNC/',
#               '../modelResult_500/SINCERITIES/',
#               '../modelResult_500/SCODE/', '../modelResult_500/GENIE3/', '../modelResult_500/PIDC/']
# model_path = model_path[::-1]
#
# data_set = ['hESC', 'mESC', 'mHSC_E', 'mHSC_GM', 'mHSC_L']
# data_name = ['hESC', 'mESC(2)', 'mHSC(E)', 'mHSC(GM)', 'mHSC(L)']
# colors = ['#AF58BA', '#009ADE', '#00CD6C', '#F28522', "#E32977", '#C5E1EF', '#FF69B4']
# colors = colors[::-1]
# firstPlt(model_set, model_path, data_set, data_name, colors,
#          save_name='Figure2C_AUPRC.pdf',index='C',flag=0)

# # time data
model_set = ['GMFGRN', 'dynDeepDRIM', 'TDL-LSTM', 'TDL-3DCNN', 'MI', 'PCC', 'dynGENIE3']
model_set = model_set[::-1]
model_path = ['TimeData/myModel_3/', 'TimeData/dynDeepDRIM/', 'TimeData/TDL-LSTM/', 'TimeData/TDL-3DCNN/',
              'TimeData/MI/', 'TimeData/PCC/', 'TimeData/dynGENIE3/']
model_path = model_path[::-1]
data_set = ['mesc1', 'mesc2', 'hesc1', 'hesc2']
data_name = ['mESC1', 'mESC2', 'hESC1', 'hESC2']
colors = ['#AF58BA', '#009ADE', '#00CD6C', '#06592A', '#F28522', "#E32977", '#FFF3B2']
colors = colors[::-1]
firstPlt(model_set, model_path, data_set, data_name, colors,
         save_name='Figure5A_AUROC.pdf',index='A',flag=0)
