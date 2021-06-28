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
from utils.visualize import show_mask

parser = argparse.ArgumentParser()
parser.add_argument('--cfg_file', type=str)
args = parser.parse_args()
params = HParams(args.cfg_file)
pprint(params.dict)

os.environ['CUDA_VISIBLE_DEVICES'] = params.gpu
np.random.seed(params.seed)
tf.set_random_seed(params.seed)

save_dir = os.path.join(params.exp_dir, 'debug')
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
total_acc = []
total_xent = []

testset.initialize(sess)
num_batches = testset.num_batches
for n in range(num_batches):
    x, y = sess.run([testset.x, testset.y])
    b = np.ones_like(x)
    m = b
    acc, xent = sess.run([model.acc, model.xent], {model.x:x, model.y:y, model.b:b, model.m:m})
    total_acc.append(acc)
    total_xent.append(xent)
total_acc = np.concatenate(total_acc, axis=0)
total_xent = np.concatenate(total_xent, axis=0)

logging.info(f'acc: {np.mean(total_acc)}')
logging.info(f'xent: {np.mean(total_xent)}')