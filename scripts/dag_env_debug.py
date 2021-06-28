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

from dag_modules.ppo import PPOPolicy
from dag_modules.acflow_env import Env as ACFlowEnv
from dag_modules.scikit_env import Env as SciKitEnv
from utils.hparams import HParams

parser = argparse.ArgumentParser()
parser.add_argument('--cfg_file', type=str)
parser.add_argument('--env', type=str)
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--hard', action='store_true')
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--env_device', type=int, default=0)
parser.add_argument('--agent_device', type=int, default=1)
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()
params = HParams(args.cfg_file)

np.random.seed(args.seed)
tf.set_random_seed(args.seed)


env_dict = {
    'acflow': ACFlowEnv,
    'scikit': SciKitEnv
}
Env = env_dict[args.env]

# train
with tf.device(f"/gpu:{args.env_device}"):
    env = Env(params)

with tf.device(f"/gpu:{args.agent_device}"):
    agent = PPOPolicy(params, env)

os.makedirs(f'{params.exp_dir}/debug', exist_ok=True)

x = env.data
d = x.shape[1]
for _ in range(10):
    i, j = np.random.choice(d, size=(2), replace=False)
    b = np.zeros_like(x)
    m = np.zeros_like(x)
    m[:,i] = 1.0
    m[:,j] = 1.0
    sam = env.model.run(env.model.sam, {env.model.x: x, env.model.b:b, env.model.m:m})
    sam = (sam[:,i], sam[:,j])
    org = (x[:,i], x[:,j])
    fig = plt.figure()
    plt.scatter(sam[0], sam[1], label='sam')
    plt.scatter(org[0], org[1], label='org')
    plt.legend()
    plt.savefig(f'{params.exp_dir}/debug/{i}-{j}.png')