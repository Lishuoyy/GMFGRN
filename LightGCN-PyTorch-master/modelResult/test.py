import matplotlib.pyplot as plt
import numpy as np

# 数据
categories = ['A', 'B', 'C', 'D']
groups = ['Group 1', 'Group 2', 'Group 3']  # 多个组的名称
data = np.array([[10, 15, 7, 12],
                 [8, 11, 9, 6],
                 [5, 9, 11, 8]])  # 每个组的数据

# 设置柱形宽度和间隔
bar_width = 0.2
index = np.arange(len(categories))

# 计算组之间的距离
group_distance = 0.5

# 创建分组柱形图
for i, group_data in enumerate(data):
    plt.bar(index + (i - len(groups)/2) * bar_width * len(groups) - group_distance/2, group_data, bar_width, label=groups[i])

# 设置标签、标题等
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Grouped Bar Chart')
plt.xticks(index, categories)
plt.legend()

# 调整布局
plt.tight_layout()

# 显示图像
plt.show()
