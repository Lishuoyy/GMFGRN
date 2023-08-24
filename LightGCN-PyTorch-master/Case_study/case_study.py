import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch as th
# Read the data
taf1_network_path = 'hesc2_2/result_taf1.txt'
taf1_network = pd.read_csv(taf1_network_path, sep=',', header=None)
taf1_network.columns = ['tf', 'target', 'score']
# print(taf1_network)

tbp_network_path = 'hesc2_2/result_tbp.txt'
tbp_network = pd.read_csv(tbp_network_path, sep=',', header=None)
tbp_network.columns = ['tf', 'target', 'score']
# print(tbp_network)

# Merge the two network top 50
network = pd.concat([taf1_network.iloc[:50,:], tbp_network.iloc[:50,:]], ignore_index=True)
print(network)
# Create a undirected graph
G = nx.Graph()
for i in range(len(network)):
    G.add_edge(network['tf'][i], network['target'][i], weight=network['score'][i])

edge_colors = [G[u][v]['weight'] for u, v in G.edges()]
cmap = plt.cm.plasma
# use plt to draw the graph
plt.figure(figsize=(10, 10))
pos = nx.spring_layout(G)
# pos = nx.kamada_kaway_layout(G)
nx.draw_networkx_nodes(G, pos, node_size=500, node_color='r')
nx.draw_networkx_edges(G, pos, edge_color=edge_colors, edge_cmap = cmap,width=1.0, alpha=0.5)
nx.draw_networkx_labels(G, pos, font_size=10, font_color='b')
plt.axis('off')
plt.show()
