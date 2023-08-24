import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch as th
# Read the data
network_path = 'hesc2/result_all_tfs.txt'
network = pd.read_csv(network_path, sep=',', header=None)
network.columns = ['tf', 'target', 'score']
print(network)

# Create a undirected graph
G = nx.Graph()
for i in range(100):
    G.add_edge(network['tf'][i], network['target'][i], weight=network['score'][i])


M = G.number_of_edges()
node_sizes = 1
edge_colors = [G[u][v]['weight'] for u, v in G.edges()]
edge_alphas = [(5 + i) / (M + 4) for i in range(M)]
cmap = plt.cm.plasma

# use plt to draw the graph
# plt.figure(figsize=(20, 20))
# pos = nx.spring_layout(G)
pos = nx.kamada_kawai_layout(G)
nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="indigo")
edges = nx.draw_networkx_edges(
    G,
    pos,
    node_size=node_sizes,
    # arrowstyle="->",
    arrowsize=10,
    edge_color=edge_colors,
    edge_cmap=cmap,
    width=2,
    arrows=True,
)
# print(len(edges),M, len(node_sizes))
# exit()
# nx.draw_networkx_labels(G, pos, font_size=10, font_color='b')

# set alpha value for each edge
for i in range(M):
    edges[i].set_alpha(edge_alphas[i])

pc = mpl.collections.PatchCollection(edges, cmap=cmap)
pc.set_array(edge_colors)
ax = plt.gca()
ax.set_axis_off()
plt.colorbar(pc, ax=ax)
plt.show()
