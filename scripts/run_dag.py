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

################################################################
logging.basicConfig(filename=params.exp_dir + f'/{args.mode}.log',
                    filemode='w',
                    level=logging.DEBUG if args.debug else logging.INFO,
                    format='%(asctime)s %(message)s')
logging.info(pformat(params.dict))

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

if args.mode == 'resume':
    agent.load()
    agent.train()
if args.mode == 'train':
    agent.train()
eval_dict = agent.evaluate(hard=args.hard)

# save
os.makedirs(f'{params.exp_dir}/evaluate', exist_ok=True)
with open(f'{params.exp_dir}/evaluate/eval.pkl', 'wb') as f:
    pickle.dump(eval_dict, f)