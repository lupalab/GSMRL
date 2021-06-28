import os
import sys
p = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(p)
import argparse
import logging
import tensorflow as tf
import numpy as np
import pickle
from pprint import pformat, pprint
import matplotlib.pyplot as plt

from dag_modules.dag_env import Env
from utils.hparams import HParams
from dag_modules.metrics import compute_metrics

parser = argparse.ArgumentParser()
parser.add_argument('--cfg_file', type=str)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--env_device', type=int, default=0)
args = parser.parse_args()
params = HParams(args.cfg_file)

################################################################
logging.basicConfig(filename=params.exp_dir + '/debug.log',
                    filemode='w',
                    level=logging.INFO,
                    format='%(asctime)s %(message)s')
logging.info(pformat(params.dict))

np.random.seed(args.seed)
tf.set_random_seed(args.seed)

# train
with tf.device(f"/gpu:{args.env_device}"):
    env = Env(params)

# gt dag
A_gt = env.trainset.adjacency
ll_gt = env._joint_ll(A_gt, 'train')[0]
logging.info(f'll_gt: {ll_gt}')

# random graph
def gen_random_graph(d, degree):
    prob = float(degree) / (d - 1)
    B = np.tril((np.random.rand(d, d) < prob).astype(float), k=-1)
    P = np.random.permutation(np.eye(d, d))  # permutes first axis only
    B_perm = P.T.dot(B).dot(P)

    return B_perm

d = len(A_gt)
degree = np.sum(A_gt) / d

x = [0.]
y = [ll_gt]
for i in range(100):
    A_gen = gen_random_graph(d, degree)
    metrics = compute_metrics(A_gen, A_gt)
    ll_gen = env._joint_ll(A_gen, 'train')[0]
    logging.info(f'll_gen: {ll_gen}')
    x.append(metrics['shd'])
    y.append(ll_gen)

fig = plt.figure()
plt.plot(x, y)
plt.savefig('debug.png')
plt.close('all')

