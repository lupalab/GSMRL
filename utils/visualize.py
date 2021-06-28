import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import networkx as nx
import numpy as np
from collections import Counter

def show_graph(graph, mask, fname):
    G = nx.DiGraph()
    edges = list(zip(*reversed(np.where(graph))))
    G.add_edges_from(edges)
    nodes = [(i,{'step':s}) for i,s in enumerate(mask)]
    G.add_nodes_from(nodes)

    fig = plt.figure()
    pos_nodes = nx.spring_layout(G)
    nx.draw(G, pos_nodes, with_labels=True)

    pos_attrs = {}
    for node, coords in pos_nodes.items():
        pos_attrs[node] = (coords[0], coords[1] + 0.08)
    node_attrs = nx.get_node_attributes(G, 'step')
    custom_node_attrs = {}
    for node, attr in node_attrs.items():
        custom_node_attrs[node] = f"step:{attr}"
    nx.draw_networkx_labels(G, pos_attrs, labels=custom_node_attrs)

    plt.savefig(fname)
    plt.close('all')

def show_mask(mask, fname):
    '''
    mask: [B, d]
    '''
    B, d = mask.shape
    res = np.zeros([d, d])
    for step in range(d):
        inds = np.where(mask == step)[1]
        count = Counter(inds)
        for k, v in count.items():
            res[k, step] = v / B
    
    fig, ax = plt.subplots()
    ax.matshow(res, cmap=plt.cm.hot)
    for (i, j), z in np.ndenumerate(res):
        ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')

    plt.xlabel('steps')
    plt.ylabel('selected')
    plt.savefig(fname)
    plt.close('all')

def plot_dict(filename, value_dict, xlabel='step'):
    # Plot step vs the mean(last 50 episodes' rewards)
    fig = plt.figure(figsize=(12, 4 * len(value_dict)))

    for i, (key, values) in enumerate(value_dict.items()):
        ax = fig.add_subplot(len(value_dict), 1, i + 1)
        ax.plot(range(len(values)), values)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(key)
        ax.grid('k--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(filename)

if __name__ == '__main__':
    import os
    import argparse
    import gzip
    import pickle
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', type=str)
    args = parser.parse_args()

    with gzip.open(args.fname, 'rb') as f:
        res = pickle.load(f)
    
    dir_name = os.path.split(args.fname)[0]
    save_path = os.path.join(dir_name, 'mask.png')
    show_mask(res['masks'], save_path)




    



