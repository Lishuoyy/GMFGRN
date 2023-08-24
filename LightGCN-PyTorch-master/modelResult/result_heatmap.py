import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve


# CNNC_path = 'CNNC/'
# bonemarrow, dendritic, mesc1 AUROC and AUPRC
def firstPlt(model_set, auroc, auprc, data_name, save_name, index):
    auroc = auroc[::-1]
    auprc = auprc[::-1]
    auroc = np.array(auroc).T
    auprc = np.array(auprc).T
    print(auroc)
    print(auprc)

    # 热力图
    # plt.rc('font', size=16)
    fig = plt.figure(figsize=(9, 3), dpi=300)
    ax, ax1 = fig.subplots(1, 2)
    # fig, ax = plt.subplots()
    # AUROC
    heatmap = ax.imshow(auroc, cmap='inferno', aspect='auto')
    ax.tick_params(axis='x', labeltop=True, labelbottom=False, )
    ax.xaxis.set_ticks_position('top')
    # ax.tick_params(axis='y', length=0)
    ax.set_xticks(np.arange(len(model_set)))
    ax.set_xticklabels(model_set, rotation=25, )
    ax.set_yticks(np.arange(len(data_name)))
    ax.set_yticklabels(data_name)
    # ax.text(-0.15, 1.35, index, transform=ax.transAxes, fontsize=16, fontweight='bold', va='top', ha='left')
    # fig.text(0.02, 1.03, index, transform=fig.transFigure, fontsize=16, fontweight='bold', va='top', ha='left')

    # 显示颜色条
    # cbar = plt.colorbar(heatmap)
    ax.set_title('AUROC', fontweight='bold')

    # 填充数字
    for i in range(len(data_name)):
        for j in range(len(model_set)):
            text = ax.text(j, i, '%.3f' % auroc[i, j], ha='center', va='center', color='black')

    # AUPRC
    heatmap = ax1.imshow(auprc, cmap='inferno', aspect='auto')
    ax1.tick_params(axis='x', labeltop=True, labelbottom=False, )
    ax1.xaxis.set_ticks_position('top')
    ax1.tick_params(axis='y', length=0)
    ax1.set_xticks(np.arange(len(model_set)))
    ax1.set_xticklabels(model_set, rotation=25, )
    ax1.set_yticks([])
    # ax1.set_yticklabels(data_name)
    # 显示颜色条
    # cbar = plt.colorbar(heatmap)
    ax1.set_title('AUPRC', fontweight='bold')
    # 填充数字
    for i in range(len(data_name)):
        for j in range(len(model_set)):
            text = ax1.text(j, i, '%.3f' % auprc[i, j], ha='center', va='center', color='black')
    plt.tight_layout()
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.show()


eight_data_auroc = [[0.6322, 0.5848, 0.5069, 0.5534, 0.6162, 0.5759, 0.5194, 0.5098],
                    [0.7025, 0.6746, 0.7071, 0.6363, 0.7538, 0.7007, 0.6997, 0.6932],
                    [0.7233, 0.6538, 0.6817, 0.6513, 0.7866, 0.6955, 0.7003, 0.7003],
                    [0.7926, 0.6906, 0.7357, 0.7143, 0.8856, 0.7992, 0.7967, 0.8068],
                    [0.8054, 0.7102, 0.7961, 0.7386, 0.8882, 0.8103, 0.8131, 0.8156]]
eight_data_auprc = [[0.6647, 0.6212, 0.5815, 0.5400, 0.6030, 0.5729, 0.5211, 0.5051],
                    [0.6846, 0.6690, 0.7033, 0.5938, 0.7558, 0.6575, 0.6516, 0.6444],
                    [0.7084, 0.6437, 0.6871, 0.6008, 0.7268, 0.6463, 0.6524, 0.6543],
                    [0.7805, 0.6759, 0.7344, 0.6547, 0.8249, 0.7339, 0.7302, 0.7458],
                    [0.8013, 0.6998, 0.7702, 0.6806, 0.8231, 0.7682, 0.7582, 0.7825]]

five_data_500_auroc = [[0.4306, 0.5208, 0.4989, 0.4964, 0.4889],
                       [0.5604, 0.4475, 0.4942, 0.4919, 0.4929],
                       [0.5032, 0.4813, 0.4885, 0.5336, 0.5141],
                       [0.5137, 0.4775, 0.5194, 0.4955, 0.5544],
                       [0.5658, 0.6584, 0.5614, 0.5935, 0.6049],
                       [0.6269, 0.7599, 0.7227, 0.7238, 0.7519],
                       [0.6420, 0.7363, 0.7368, 0.7377, 0.7539]]
five_data_500_auprc = [[0.5638, 0.6574, 0.6134, 0.6003, 0.4116],
                       [0.6704, 0.6057, 0.6241, 0.5897, 0.4233],
                       [0.6088, 0.6293, 0.6069, 0.6028, 0.4246],
                       [0.6402, 0.6134, 0.6466, 0.5974, 0.4936],
                       [0.6771, 0.7349, 0.6581, 0.6705, 0.4928],
                       [0.7203, 0.8042, 0.7457, 0.7263, 0.6698],
                       [0.7010, 0.7851, 0.7780, 0.7494, 0.6739]]
time_data_auroc = [[0.5136, 0.4984, 0.5005, 0.5018],
                   [0.5487, 0.5075, 0.6053, 0.5753],
                   [0.6284, 0.7127, 0.7095, 0.6319],
                   [0.5983, 0.7263, 0.7068, 0.6374, ],
                   [0.6127, 0.7320, 0.7035, 0.6469, ],
                   [0.7165, 0.8145, 0.7592, 0.6938, ],
                   [0.6908, 0.8202, 0.7632, 0.7177]]
time_data_auprc = [[0.5164, 0.4913, 0.5014, 0.5037],
                   [0.5392, 0.5198, 0.5889, 0.5685],
                   [0.5891, 0.6538, 0.6614, 0.5941],
                   [0.5659, 0.6660, 0.6587, 0.5918],
                   [0.5701, 0.6746, 0.6500, 0.6083],
                   [0.6774, 0.7579, 0.7200, 0.6592],
                   [0.6456, 0.7619, 0.7106, 0.6720]]

# eight data
model_set = ['GMFGRN', 'DeepDRIM', 'CNNC', 'MI', 'PCC']
# data_set = ['bonemarrow', 'dendritic', 'mesc_1', 'hESC', 'mESC_2', 'mHSC_E', 'mHSC_GM', 'mHSC_L']
data_name = ['boneMarrow', 'dendritic', 'mESC(1)', 'hESC', 'mESC(2)', 'mHSC(E)', 'mHSC(GM)', 'mHSC(L)']
firstPlt(model_set, eight_data_auroc, eight_data_auprc, data_name,
         save_name='eight_data_heatmap_1.pdf',index='A')

# five data 500
# model_set = ['GMFGRN', 'DeepDRIM', 'CNNC', 'SINCERITIES', 'GENIE3', 'SCODE', 'PIDC']
# # data_set = ['bonemarrow', 'dendritic', 'mesc_1', 'hESC', 'mESC_2', 'mHSC_E', 'mHSC_GM', 'mHSC_L']
# data_name = ['hESC', 'mESC(2)', 'mHSC(E)', 'mHSC(GM)', 'mHSC(L)']
# firstPlt(model_set, five_data_500_auroc, five_data_500_auprc, data_name,
#          save_name='five_data_500_heatmap_1.pdf',index='B')

# time data
# model_set = ['GMFGRN', 'dynDeepDRIM', 'TDL-LSTM', 'TDL-3DCNN', 'MI', 'PCC', 'dynGENIE3']
# data_name = ['mESC1', 'mESC2', 'hESC1', 'hESC2']
# firstPlt(model_set, time_data_auroc, time_data_auprc, data_name,
#          save_name='time_data_heatmap_1.pdf', index='B')
