import os
import sys
p = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(p)
import argparse
import logging
import tensorflow as tf
import numpy as np
from sklearn.metrics import auc
from pprint import pformat, pprint
import gzip
import pickle
import time
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

from datasets import get_dataset
from models import get_model
from afa_modules.ts_afa import AFA
from utils.hparams import HParams
from utils.visualize import show_mask

parser = argparse.ArgumentParser()
parser.add_argument('--cfg_file', type=str)
parser.add_argument('--alpha', type=float, default=10.0)
parser.add_argument('--num_batches', type=int, default=-1)
parser.add_argument('--num_steps', type=int, default=-1)
args = parser.parse_args()
params = HParams(args.cfg_file)
setattr(params, 'alpha', args.alpha)
setattr(params, 'num_batches', args.num_batches)
setattr(params, 'num_steps', args.num_steps)
pprint(params.dict)

os.environ['CUDA_VISIBLE_DEVICES'] = params.gpu
np.random.seed(params.seed)
tf.set_random_seed(params.seed)

save_dir = os.path.join(params.exp_dir, 'test_ts')
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

###############################################################
# # cross validate alpha
# testset.initialize(sess)
# x, y = sess.run([testset.x, testset.y])
# x, y = x[:100], y[:100]
# alpha_list = np.arange(10., 100., 10.)
# logging.info(f'searching alpha over {list(alpha_list)}')
# best_area = 0
# best_alpha = 0
# for alpha in alpha_list:
#     params.alpha = alpha
#     afa = AFA(params, model)
#     mask, pred = afa(x)
#     acc = (pred == np.expand_dims(y,axis=1)).astype(np.float32)
#     acc = np.mean(acc, axis=0)
#     T = np.arange(1, params.time_steps+1)
#     area = auc(T, acc)
#     if area > best_area:
#         best_alpha = alpha
# logging.info(f'set alpha to {best_alpha}')
# params.alpha = best_alpha

# AFA
afa = AFA(params, model)

###############################################################
inds = []
masks = []
preds = []
acc = []

testset.initialize(sess)
num_batches = testset.num_batches
if params.num_batches > 0:
    num_batches = min(num_batches, params.num_batches)
for n in range(num_batches):
    i, x, y = sess.run([testset.i, testset.x, testset.y])
    mask, pred = afa(x)
    inds.append(i)
    masks.append(mask)
    preds.append(pred)
    acc.append((pred == np.expand_dims(y,axis=1)).astype(np.float32))
inds = np.concatenate(inds, axis=0)
masks = np.concatenate(masks, axis=0)
preds = np.concatenate(preds, axis=0)
acc = np.concatenate(acc, axis=0)

res = {
    'inds': inds,
    'masks': masks,
    'preds': preds,
    'acc': acc
}
with gzip.open(f'{save_dir}/results.pgz', 'wb') as f:
    pickle.dump(res, f)

x = range(1, params.time_steps+1)
y = np.mean(acc, axis=0)
fig = plt.figure()
plt.plot(x, y, marker='x')
plt.xticks(x)
plt.xlabel('time step')
plt.ylabel('accuracy')
plt.savefig(f'{save_dir}/acc.png')
plt.close('all')

B = masks.shape[0]
masks = np.reshape(masks, [B,params.time_steps,-1])
show_mask(masks[:,:,0], f'{save_dir}/mask.png')