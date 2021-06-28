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
from afa_modules.fbn_afa import AFA
from utils.hparams import HParams
from utils.visualize import show_mask

parser = argparse.ArgumentParser()
parser.add_argument('--cfg_file', type=str)
parser.add_argument('--pred_dir', type=str)
args = parser.parse_args()
params = HParams(args.cfg_file)

os.environ['CUDA_VISIBLE_DEVICES'] = params.gpu
np.random.seed(params.seed)
tf.set_random_seed(params.seed)

save_dir = os.path.join(params.exp_dir, args.pred_dir)

config = tf.ConfigProto()
config.log_device_placement = True
config.allow_soft_placement = True
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

###############################################################
# data
testset = get_dataset('test', params)

# model
model = get_model(sess, params)
saver = tf.train.Saver(tf.global_variables())
restore_from = os.path.join(params.exp_dir, 'weights/params.ckpt')
saver.restore(sess, restore_from)

###############################################################

with gzip.open(f'{save_dir}/results.pgz', 'rb') as f:
    res = pickle.load(f)
inds = res['inds']
masks = res['masks']
masks = masks[np.argsort(inds)]

def extend_mask(mask):
    d = mask.shape[1]
    mask = mask.copy()
    for m in mask:
        size = np.sum(m==-1)
        m[m==-1] = np.arange(d-1, d-size-1, -1)
    
    return mask

def predict(x, y, mask):
    # x: [B,d]
    # mask: [B,d]
    B,d = mask.shape
    mask = extend_mask(mask)
    t = np.arange(d).reshape([1,d,1])
    b = (np.expand_dims(mask, axis=1) <= t).astype(np.float32)
    xx = np.repeat(np.expand_dims(x,axis=1),d,axis=1)
    xx = np.reshape(xx,[B*d,d])
    yy = np.repeat(np.expand_dims(y,axis=1),d,axis=1)
    yy = np.reshape(yy,[B*d,params.n_target])
    bb = np.reshape(b,[B*d,d])
    pred = sess.run(model.mean_y, 
        {model.x:xx,model.y:yy,model.b:bb,model.m:bb})
    pred = pred.reshape([B,d,params.n_target])

    return pred


mse = []
testset.initialize(sess)
num_batches = testset.num_batches
for n in range(num_batches):
    i, x, y = sess.run([testset.i, testset.x, testset.y])
    mask = masks[i]
    pred = predict(x, y, mask)
    mse.append(np.sum((pred-y[:,None,:])**2, axis=-1))
mse = np.concatenate(mse, axis=0)
with gzip.open(f'{save_dir}/pred_fbn.pgz', 'wb') as f:
    pickle.dump(mse, f)

x = range(1, params.dimension+1)
y = np.sqrt(np.mean(mse, axis=0))
fig = plt.figure()
plt.plot(x, y, marker='x')
plt.xticks(x)
plt.xlabel('num feature')
plt.ylabel('RMSE')
plt.savefig(f'{save_dir}/rmse.png')
plt.close('all')