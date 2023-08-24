import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch as th
import igraph as ig
import networkx as nx
import nxviz as nv
# Read the data
network_path = 'hesc2/result_all_tfs.txt'
network = pd.read_csv(network_path, sep=',', header=None)
network.columns = ['tf', 'target', 'score']
# network['direction'] = ['undirected' for i in range(len(network))]

# table = network.iloc[:200,:]
# table.to_excel('hesc2/result_all_tfs_200.xlsx', index=False)
# print(table)
# print(table['weight'].max(), table['weight'].min())
# exit()



# Create a undirected graph
# G = nx.Graph()
# for i in range(100):
#     G.add_edge(network['tf'][i], network['target'][i], weight=network['score'][i])
#
#
# ax = nv.circos(G)
# ax.draw()
# plt.show()