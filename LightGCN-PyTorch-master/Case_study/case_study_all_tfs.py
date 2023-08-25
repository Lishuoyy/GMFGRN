import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# import torch as th
import igraph as ig
import networkx as nx

# import nxviz as nv
# Read the data
network_path = 'hesc2/result_all_tfs.txt'
network = pd.read_csv(network_path, sep=',', header=None)
network.columns = ['tf', 'target', 'score']
network = network.iloc[:100, :]
# network['direction'] = ['undirected' for i in range(len(network))]

# table = network.iloc[:200,:]
# table.to_excel('hesc2/result_all_tfs_200.xlsx', index=False)
# print(table)
# print(table['weight'].max(), table['weight'].min())
# exit()


# Create a undirected graph
g = ig.Graph.TupleList(network.itertuples(index=False), directed=False, weights=True)
g.vs["label"] = g.vs["name"]
fig, ax = plt.subplots(figsize=(20, 20), dpi=300)

ig.plot(
    g,
    ax,
    layout=g.layout("kk"),
    vertex_size=65,
    vertex_label_size=20,
    vertex_label=g.vs["name"],
    autocurve=True,
    # scale=100,
    #     vertex_label_size=10,
    #     edge_arrow_size=0.5,
    #     edge_width=0.5,
    #     edge_curved=0.2,
    #     edge_color="gray",
)
plt.show()
