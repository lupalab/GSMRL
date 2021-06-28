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
parser.add_argument('--mfile', type=str)
args = parser.parse_args()
params = HParams(args.cfg_file)
pprint(params.dict)

os.environ['CUDA_VISIBLE_DEVICES'] = params.gpu
np.random.seed(params.seed)
tf.set_random_seed(params.seed)

save_dir = os.path.join(params.exp_dir, 'test_reg2')
os.makedirs(save_dir, exist_ok=True)
logging.basicConfig(filename=save_dir + '/test.log',
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
testset = get_dataset('test', params)

# model
model = get_model(sess, params)
saver = tf.train.Saver(tf.global_variables())
restore_from = os.path.join(params.exp_dir, 'weights/params.ckpt')
saver.restore(sess, restore_from)

##############################################################
with gzip.open(args.mfile, 'rb') as f:
    res = pickle.load(f)
inds = res['inds']
masks = res['masks']
masks = masks[np.argsort(inds)]

##############################################################
preds = []
mse = []

testset.initialize(sess)
num_batches = testset.num_batches
for n in range(num_batches):
    i, x, y = sess.run([testset.i, testset.x, testset.y])
    B, d = x.shape
    B, Nt = y.shape
    mask = np.expand_dims(masks[i], axis=1)
    xx = np.repeat(np.expand_dims(x, axis=1), d, axis=1).reshape([B*d, d])
    yy = np.repeat(np.expand_dims(y, axis=1), d, axis=1).reshape([B*d, Nt])
    t1 = np.zeros([d, 1])
    t2 = np.expand_dims(np.arange(d), axis=1)
    bb = np.logical_and(mask >= t1, mask <= t2-1).astype(np.float32)
    bb = bb.reshape([B*d, d])
    mm = np.logical_and(mask >= t1, mask <= t2).astype(np.float32)
    mm = mm.reshape([B*d, d])
    pred = sess.run(model.mean_y, {model.x:xx, model.y:yy, model.b:bb, model.m:mm})
    pred = pred.reshape([B, d, Nt])
    preds.append(pred)
    mse.append(np.sum((pred - y[:,None,:])**2, axis=-1))
preds = np.concatenate(preds, axis=0)
mse = np.concatenate(mse, axis=0)

res = {
    'masks': masks,
    'preds': preds,
    'mse': mse
}
with gzip.open(f'{save_dir}/results.pgz', 'wb') as f:
    pickle.dump(res, f)

x = range(1, params.dimension+1)
y = np.sqrt(np.mean(mse, axis=0))
fig = plt.figure()
plt.plot(x, y, marker='x')
plt.xticks(x)
plt.xlabel('num feature')
plt.ylabel('RMSE')
plt.savefig(f'{save_dir}/rmse.png')
plt.close('all')
