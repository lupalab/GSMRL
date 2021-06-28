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

from datasets import get_dataset
from models import get_model
from afa_modules.fbn_markov_blanket import MB
from afa_modules.fbn_learn import GS
from utils.hparams import HParams

parser = argparse.ArgumentParser()
parser.add_argument('--cfg_file', type=str)
parser.add_argument('--normalize', action='store_true')
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--num_batches', type=int, default=-1)
parser.add_argument('--cmi_thresh', type=float, default=0.0)
args = parser.parse_args()
params = HParams(args.cfg_file)
setattr(params, 'normalize', args.normalize)
setattr(params, 'batch_size', args.batch_size)
setattr(params, 'num_batches', args.num_batches)
setattr(params, 'cmi_thresh', args.cmi_thresh)
pprint(params.dict)

os.environ['CUDA_VISIBLE_DEVICES'] = params.gpu
np.random.seed(params.seed)
tf.set_random_seed(params.seed)

save_dir = os.path.join(params.exp_dir, 'fbn_estimate')
os.makedirs(save_dir, exist_ok=True)
logging.basicConfig(filename=save_dir + '/fbn_estimate.log',
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
validset = get_dataset('valid', params)

# model
model = get_model(sess, params)
saver = tf.train.Saver(tf.global_variables())
restore_from = os.path.join(params.exp_dir, 'weights/params.ckpt')
saver.restore(sess, restore_from)

# MB
markB = MB(params, model)

# GS
gs = GS(params, model)

###############################################################

data = []
label = []
validset.initialize(sess)
num_batches = validset.size // params.batch_size
if params.num_batches > 0:
    num_batches = min(num_batches, params.num_batches)
for n in range(num_batches):
    x, y = sess.run([validset.x, validset.y])
    data.append(x)
    label.append(y)
data = np.concatenate(data, axis=0)
label = np.concatenate(label, axis=0)

# estimate markov blanket
cmi, mb = markB(data, label)
logging.info('markov blanket:')
logging.info(mb)
with open(f'{save_dir}/results.pkl', 'wb') as f:
    pickle.dump({'cmi':cmi, 'mb':mb}, f)

# get DAG
graph = gs(data, label, mb)
logging.info('graph:')
logging.info(graph)
with open(f'{save_dir}/results.pkl', 'wb') as f:
    pickle.dump({'cmi':cmi, 'mb':mb, 'graph':graph}, f)

