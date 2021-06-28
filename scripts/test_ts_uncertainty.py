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
from utils.hparams import HParams

parser = argparse.ArgumentParser()
parser.add_argument('--cfg_file', type=str)
parser.add_argument('--split', type=str)
args = parser.parse_args()
params = HParams(args.cfg_file)
pprint(params.dict)

os.environ['CUDA_VISIBLE_DEVICES'] = params.gpu
np.random.seed(params.seed)
tf.set_random_seed(params.seed)

save_dir = os.path.join(params.exp_dir, f'{args.split}_ts_uncertainty_cal')
os.makedirs(save_dir, exist_ok=True)
logging.basicConfig(filename=save_dir + f'/{args.split}.log',
                    filemode='w',
                    level=logging.INFO,
                    format='%(message)s')
logging.info(pformat(params.dict))

config = tf.ConfigProto()
config.log_device_placement = True
config.allow_soft_placement = True
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

###############################################################
# data
testset = get_dataset(args.split, params)

# model
model = get_model(sess, params)
saver = tf.train.Saver(tf.global_variables())
restore_from = os.path.join(params.exp_dir, 'weights/params.ckpt')
saver.restore(sess, restore_from)

###############################################################
probs = []
label = []

B = params.batch_size
T = params.time_steps
C = params.n_classes
d = params.dimension
# construct mask
b = np.zeros([B,T,T,d//T], dtype=np.float32)
for t in range(T):
    b[:,t,:t+1] = 1.

testset.initialize(sess)
num_batches = testset.num_batches
for n in range(num_batches):
    x, y = sess.run([testset.x, testset.y])
    B = x.shape[0]
    x = np.repeat(np.expand_dims(x, axis=1), T, axis=1)
    xx = x.reshape([B*T,d])
    bb = b[:B].reshape([B*T,d]) 
    prob = sess.run(model.prob,
        {model.x:xx, model.b:bb, model.m:bb})
    prob = prob.reshape([B,T,C])
    probs.append(prob)
    label.append(y)

probs = np.concatenate(probs, axis=0)
label = np.concatenate(label, axis=0)

res = {
    'probs': probs,
    'label': label
}
with gzip.open(f'{save_dir}/results.pgz', 'wb') as f:
    pickle.dump(res, f)

