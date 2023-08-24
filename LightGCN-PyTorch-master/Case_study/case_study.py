import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch as th
taf1_network_path = 'hesc2_2/result_taf1.txt'
taf1_network = pd.read_csv(taf1_network_path, sep=',', header=None)
taf1_network.columns = ['tf', 'target', 'score']
taf1_network = taf1_network.iloc[:20,:]

# print(taf1_network)

tbp_network_path = 'hesc2_2/result_tbp.txt'
tbp_network = pd.read_csv(tbp_network_path, sep=',', header=None)
tbp_network.columns = ['tf', 'target', 'score']
tbp_network = tbp_network.iloc[:20,:]
# print(tbp_network)

# 柱形图
data_name = tbp_network['target'].tolist()
data = tbp_network['score'].tolist()
# print(len(data))
# exit()
x = np.arange(len(data_name))
width = 0.7
fig, ax = plt.subplots(figsize=(8, 3), dpi=300)
rects = ax.bar(x, data, width, color='#3C93C2') # AF58BA

ax.set_ylabel('Scores')
ax.set_ylim(0,10)
ax.set_xticks(x)
ax.set_xticklabels(data_name, rotation=45)
# ax.legend()

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
plt.savefig('case_study_tbp.pdf', dpi=300, bbox_inches='tight')
plt.show()



