import matplotlib.pyplot as plt
import numpy as np

# 柱形图
model_set = ['GMFGRN', 'dynDeepDRIM', 'TDL-LSTM', 'TDL-3DCNN', 'MI', 'PCC', 'dynGENIE3']
model_set = model_set[::-1]
colors = ['#AF58BA', '#009ADE', '#00CD6C', '#06592A', '#F28522', "#E32977", '#FFF3B2']
colors = colors[::-1]
data_name = ['mESC1', 'mESC2', 'hESC1', 'hESC2']
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
data = time_data_auprc
data = np.array(data)

x = np.arange(len(data_name))
width = 0.11


# 添加数值标签
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.3f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',rotation=90)


fig, ax = plt.subplots(figsize=(8,4), dpi=300)
ax.spines['top'].set_visible(False)
for i in range(len(model_set)):
    rects = ax.bar(x+i*width, data[i], width, label=model_set[i], color=colors[i])
    autolabel(rects)

ax.set_ylabel('AUROC')
ax.set_xticks(x+width*3)
ax.set_xticklabels(data_name, rotation=0)
ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1))


# fig.text(0.02, 1.03, 'C.', transform=fig.transFigure, fontsize=14, fontweight='bold', va='top', ha='left')
plt.tight_layout()
plt.savefig('result_timeBar_AUPRC.pdf', dpi=300, bbox_inches='tight')
plt.show()
