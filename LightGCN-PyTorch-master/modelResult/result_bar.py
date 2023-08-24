import matplotlib.pyplot as plt
import numpy as np

# 柱形图
model_set = ['GMFGRN', 'DeepDRIM', 'CNNC']
model_set = model_set[::-1]
data_name = ['boneMarrow', 'dendritic', 'mESC(1)', 'hESC', 'mESC(2)', 'mHSC(E)', 'mHSC(GM)', 'mHSC(L)']
data = [[12433, 5941, 38169, 20711, 34965, 8961, 9749, 7507],
        [7463, 5935, 29218, 15646, 19980, 5776, 6268, 6522],
        [6035, 4639, 22218, 14048, 18916, 6907, 6197, 6022]]
data = np.array(data)

x = np.arange(len(data_name))
width = 0.2
fig, ax = plt.subplots(figsize=(6, 3), dpi=300)
rects1 = ax.bar(x - width, data[0], width, label=model_set[0], color='#00CD6C')
rects2 = ax.bar(x, data[1], width, label=model_set[1], color='#009ADE')
rects3 = ax.bar(x + width, data[2], width, label=model_set[2], color='#AF58BA')



ax.set_ylabel('False positives')
ax.set_xticks(x)
ax.set_xticklabels(data_name, rotation=15)
ax.legend()

# 添加数值标签
# def autolabel(rects):
#     for rect in rects:
#         height = rect.get_height()
#         ax.annotate('{:.2f}'.format(height),
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),
#                     textcoords="offset points",
#                     ha='center', va='bottom')
#
# autolabel(rects1)
# autolabel(rects2)
# autolabel(rects3)
# fig.text(0.02, 1.03, 'C.', transform=fig.transFigure, fontsize=14, fontweight='bold', va='top', ha='left')
plt.tight_layout()
# plt.savefig('result_bar.pdf', dpi=300, bbox_inches='tight')
plt.show()