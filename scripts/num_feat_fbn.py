import os
import sys
p = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(p)
import argparse
import logging
import tensorflow as tf
import numpy as np
from pprint import pformat, pprint
import gzip
import pickle
import time
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

from datasets import get_dataset
from models import get_model
from afa_modules.BN import BN
from utils.hparams import HParams

parser = argparse.ArgumentParser()
parser.add_argument('--cfg_file', type=str)
parser.add_argument('--gfile', type=str, required=True)
parser.add_argument('--mfile', type=str, required=True)
args = parser.parse_args()
params = HParams(args.cfg_file)
setattr(params, 'gfile', args.gfile)
setattr(params, 'mfile', args.mfile)
pprint(params.dict)

save_dir = os.path.split(params.mfile)[0]

###############################################################

# graph
with open(params.gfile, 'rb') as f:
    res = pickle.load(f)
graph = res['graph']
n_node = len(graph)
n_feat = n_node - params.n_target
y = [n_node-1-i for i in range(params.n_target)]
dag = BN()
for n in range(n_node):
    dag.add_node(str(n))
for n in range(n_node):
    for p in range(n_node):
        if graph[n, p]:
            dag.add_edge((str(p), str(n)))

def is_indep(i, y, o):
    o = [str(oi) for oi in o]
    for yi in y:
        if not dag.is_dsep(str(i), str(yi), o):
            return False
    
    return True

# mask
with gzip.open(params.mfile, 'rb') as f:
    res = pickle.load(f)
masks = res['masks']
num_feats = np.ones_like(masks) * -1
for n in range(len(masks)):
    mask = masks[n]
    for step in range(n_feat):
        o = list(np.where(np.logical_and(mask>=0, mask<step))[0])
        u = list(set(range(n_feat)) - set(o))
        num = 0
        for i in u:
            if not is_indep(i, y, o):
                num += 1
        num_feats[n, step] = num

with gzip.open(f'{save_dir}/num_feat.pgz', 'wb') as f:
    pickle.dump(num_feats, f)

