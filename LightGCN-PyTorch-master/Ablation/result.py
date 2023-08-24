import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve
import matplotlib.ticker as mtick

# CNNC_path = 'CNNC/'
# bonemarrow, dendritic, mesc1 AUROC and AUPRC
def firstPlt(model_set, model_path, data_set, data_name, colors, save_name, title, index):
    data_len = len(data_set)
    box_position = []
    row_index = 1
    for i in range(1, len(model_set) + 1):
        sublist = []
        for j in range(data_len):
            sublist.append(row_index + j * (((len(model_set) - 1) * 0.3) + 0.4))
        box_position.append(sublist)
        row_index += 0.21
    print(box_position)
    # exit()
    model_num = len(model_set)
    model_value = [[] for i in range(model_num)]

    for i, data in enumerate(data_set):
        for j in range(len(model_path)):
            if j == 0:
                model_value[j].append(np.load(model_path[j] + 'bonemarrow_' + data + '_result/' + 'AUROC_set.npy'))
                # model_value[j].append(np.load(model_path[j] + data + '/' + 'AUROC_set.npy'))
            else:
                model_value[j].append(np.load(model_path[j] + 'bonemarrow_' + data + '_result/' + 'AUPRC_set.npy'))
                # model_value[j].append(np.load(model_path[j] + data + '/' + 'AUPRC_set.npy'))
            # model_auroc[j].append(np.load(model_path[j] + data + '/AUROC_set.npy'))
            # model_auprc[j].append(np.load(data + '_result/' + 'AUPRC_set.npy'))
            # model_auroc[j].append(np.load(model_path[j] + data + '/' + 'AUROC_set.npy'))
            # model_auprc[j].append(np.load(model_path[j] + data + '/' + 'AUPRC_set.npy'))
    # print(model_auroc[0])
    # exit()
    # 计算中位线的中心点
    medians1 = [np.median(d) for d in model_value[0]]
    medians2 = [np.median(d) for d in model_value[1]]
    plt.rc('font', size=14)
    fig = plt.figure(dpi=300, figsize=(7, 4))
    # 分组箱线图，行为数据集data_set,  label为3个模型方法名称
    ax = fig.subplots(1, 1)
    #########################
    # 箱线图
    boxes = []
    for i in range(model_num):
        boxes.append(ax.boxplot(model_value[i], positions=box_position[i], widths=0.2, patch_artist=True,
                                boxprops=dict(facecolor=colors[i], edgecolor='black'),
                                whiskerprops=dict(color='black'), sym='.',
                                medianprops={'color': 'black'}, showcaps=True))
    ax.plot(box_position[0], medians1, color='#F28522', marker='o', linestyle='-', linewidth=1, markersize=2)
    ax.plot(box_position[1], medians2, color='#E31A1C', marker='o', linestyle='-', linewidth=1, markersize=2)
    #######################
    # 折线阴影图
    #####################
    # y_lower = []
    # y_upper = []
    # for d in model_auroc[0]:
    #     std = np.std(d)
    #     se = std / np.sqrt(len(d))
    #     y_lower.append(np.median(d) - se)
    #     y_upper.append(np.median(d) + se)
    # x = [1, 2, 3, 4, 5]
    # print(x)
    # print(medians)
    # ax.plot(x, medians, '#AF58BA')
    # ax.fill_between(x, y_lower, y_upper, color='#AF58BA', alpha=0.2)
    #################3

    # 添加标签
    print(len(box_position))
    print(box_position[len(box_position) // 2])
    ax.set_xticks(np.array(box_position[len(box_position) // 2])-0.1)
    boxes1 = []
    for i in range(model_num):
        boxes1.append(boxes[i]["boxes"][0])
    ax.legend(boxes1, model_set, loc='upper left', bbox_to_anchor=(1.01, 1))
    # for patch in legend.get_patches():
    #     patch.set_facecolor(patch.get_facecolor())
    # ax.set_xlim(1,5)
    # plt.xlim(np.arange(len(x)))

    # ax.set_xlabel('Dataset')
    # ax.set_yticks([0.6, 0.7, 0.8, 0.9])
    # ax.set_ylabel('AUROC')
    # ax.text(-0.15, 1.1, index, transform=ax.transAxes, fontsize=24, fontweight='bold', va='top', ha='left')
    # ax.set_xlim(1)  # 左侧和右侧各添加 0.5 的间距
    x = np.arange(len(data_name))
    # ax.set_xticks(x+0.2)
    ax.set_xticklabels(data_name)
    ax.set_title(title, y=-0.25)
    plt.tight_layout()
    plt.savefig(save_name, bbox_inches='tight')
    plt.show()


def timePoint(model_set, model_path, data_set, data_name, colors, save_name, title, index):
    data_len = len(data_set)
    box_position = []
    row_index = 1
    for i in range(1, len(model_set) + 1):
        sublist = []
        for j in range(data_len):
            sublist.append(row_index + j * (((len(model_set) - 1) * 0.3) + 0.4))
        box_position.append(sublist)
        row_index += 0.21
    print(box_position)
    # exit()
    model_num = len(model_set)
    model_value = [[] for i in range(model_num)]

    for i, data in enumerate(data_set):
        for j in range(len(model_path)):
            if j == 0:
                # model_value[j].append(np.load(model_path[j] + 'bonemarrow_' + data + '_result/' + 'AUROC_set.npy'))
                model_value[j].append(np.load(model_path[j] + data + '/' + 'AUROC_set.npy'))
            else:
                # model_value[j].append(np.load(model_path[j] + 'bonemarrow_' + data + '_result/' + 'AUPRC_set.npy'))
                model_value[j].append(np.load(model_path[j] + data + '/' + 'AUPRC_set.npy'))
            # model_auroc[j].append(np.load(model_path[j] + data + '/AUROC_set.npy'))
            # model_auprc[j].append(np.load(data + '_result/' + 'AUPRC_set.npy'))
            # model_auroc[j].append(np.load(model_path[j] + data + '/' + 'AUROC_set.npy'))
            # model_auprc[j].append(np.load(model_path[j] + data + '/' + 'AUPRC_set.npy'))
    # print(model_auroc[0])
    # exit()
    # 计算中位线的中心点
    medians1 = [np.median(d) for d in model_value[0]]
    medians2 = [np.median(d) for d in model_value[1]]
    plt.rc('font', size=14)
    fig = plt.figure(dpi=300, figsize=(7, 4))
    # 分组箱线图，行为数据集data_set,  label为3个模型方法名称
    ax = fig.subplots(1, 1)
    #########################
    # 箱线图
    # boxes = []
    # for i in range(model_num):
    #     boxes.append(ax.boxplot(model_value[i], positions=box_position[i], widths=0.2, patch_artist=True,
    #                             boxprops=dict(facecolor=colors[i], edgecolor='black'),
    #                             whiskerprops=dict(color='black'), sym='.',
    #                             medianprops={'color': 'black'}, showcaps=True))
    # ax.plot(box_position[0], medians1, color='#F28522', marker='o', linestyle='-', linewidth=1, markersize=2)
    # ax.plot(box_position[1], medians2, color='#E31A1C', marker='o', linestyle='-', linewidth=1, markersize=2)
    #######################
    # 折线阴影图
    #####################
    y_lower1 = []
    y_upper1 = []
    for d in model_value[0]:
        std = np.std(d)
        se = std / np.sqrt(len(d))
        y_lower1.append(np.median(d) - se)
        y_upper1.append(np.median(d) + se)
    x = [1, 2, 3, 4, 5]
    print(x)
    print(medians1)
    ax.plot(x, medians1, colors[0])
    legend1 = ax.fill_between(x, y_lower1, y_upper1, color=colors[0], alpha=0.2)

    y_lower2 = []
    y_upper2 = []
    for d in model_value[1]:
        std = np.std(d)
        se = std / np.sqrt(len(d))
        y_lower2.append(np.median(d) - se)
        y_upper2.append(np.median(d) + se)
    print(x)
    print(medians2)
    ax.plot(x, medians2, colors[1])
    legend2 = ax.fill_between(x, y_lower2, y_upper2, color=colors[1], alpha=0.2)
    #################3

    # 添加标签

    # ax.set_xticks(box_position[len(box_position) // 2])
    # boxes1 = []
    # for i in range(model_num):
    #     boxes1.append(boxes[i]["boxes"][0])
    # ax.legend(boxes1, model_set, loc='upper left', bbox_to_anchor=(1.01, 1))
    # for patch in legend.get_patches():
    #     patch.set_facecolor(patch.get_facecolor())
    ax.set_xlim(1,5)
    # plt.xlim(np.arange(len(x)))
    ax.legend([legend1, legend2], model_set, loc='upper left', bbox_to_anchor=(1.01, 1))
    # ax.set_xlabel('Dataset')
    # ax.set_yticks([0.6, 0.7, 0.8, 0.9])
    # ax.set_ylabel('AUROC')
    # ax.text(-0.15, 1.1, index, transform=ax.transAxes, fontsize=24, fontweight='bold', va='top', ha='left')
    ax.set_xlim(1, 5)  # 左侧和右侧各添加 0.5 的间距
    ax.set_xticks(x)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    ax.set_xticklabels(data_name)
    ax.set_title(title, y=-0.25)
    plt.tight_layout()
    plt.savefig(save_name, bbox_inches='tight')
    plt.show()

colors = ['#AF58BA', '#3C93C2'] # 226E9C
# # link
# model_set = ['AUROC', 'AUPRC']
# model_path = ['link/', 'link/']
# data_set = ['0.0_link', '0.1_link', '0.2_link', '0.3_link', '0.4_link', '0.5_link', '0.6_link', '0.7_link', '0.8_link',
#             '0.9_link']
# data_name = ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', "0.7", "0.8", "0.9"]
# firstPlt(model_set, model_path, data_set, data_name, colors, save_name='Figure6C.pdf', title='Edge decay rate',
#          index='C')

# emb
# model_set = ['AUROC', 'AUPRC']
# model_path = ['emb/','emb/']
# data_set = ['64_emb', '128_emb', '256_emb', '512_emb', '1024_emb']
# data_name = ['64', '128', '256', '512', '1024']
# # colors = ['#AF58BA', '#FFF3B2']
# firstPlt(model_set, model_path, data_set, data_name, colors,
#          save_name='Figure6A.pdf', title='Embedding size', index='A')

# rating
# model_set = ['AUROC', 'AUPRC']
# model_path = ['rating/', 'rating/']
# data_set = ['1_rating', '5_rating', '10_rating', '15_rating', '20_rating', '25_rating']
# data_name = ['1', '5', '10', '15', '20', '25']
# # colors = ['#AF58BA', '#FFF3B2']
# firstPlt(model_set, model_path, data_set, data_name, colors, save_name='Figure6B.pdf',
#          title='Number of expression levels', index='B')

# # time points
model_set = ['AUROC', 'AUPRC']
model_path = ['time_points/', 'time_points/']
data_set = ['1', '2', '3', '4', '5']
data_name = ['1', '2', '3', '4', '5']
# colors = ['#AF58BA', '#FFF3B2']
timePoint(model_set, model_path, data_set, data_name, colors, save_name='Figure6D.pdf', title='Time points',
         index='D')
