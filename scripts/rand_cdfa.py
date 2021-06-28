import os
import sys
p = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(p)
import numpy as np
import tensorflow as tf
import logging
import argparse
import time
import pickle
import gzip
from pprint import pformat, pprint
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()

from utils.hparams import HParams
from models import get_model
from cdfa_modules.gp_env import Dataset as GPDataset
from cdfa_modules.img_env import Dataset as ImgDataset

parser = argparse.ArgumentParser()
parser.add_argument('--cfg_file', type=str)
args = parser.parse_args()
params = HParams(args.cfg_file)
pprint(params.dict)

os.environ['CUDA_VISIBLE_DEVICES'] = params.gpu
np.random.seed(params.seed)
tf.set_random_seed(params.seed)

save_dir = os.path.join(params.exp_dir, 'test_rand')
os.makedirs(save_dir, exist_ok=True)

config = tf.ConfigProto()
config.log_device_placement = True
config.allow_soft_placement = True
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

################################################################
# data
Dataset = {
    'gp': GPDataset,
    'img': ImgDataset,
}[params.dataset]

testset = Dataset(params.x_size, params.y_size, params.batch_size, 'test')

# model
model = get_model(sess, params)
saver = tf.train.Saver(tf.global_variables())
restore_from = os.path.join(params.exp_dir, 'weights/params.ckpt')
saver.restore(sess, restore_from)


def test(step):
    total_acc = []
    for i in range(100):
        batch = testset.sample()
        num_points = batch['xt'].shape[1]
        idx = np.random.choice(num_points, size=step, replace=False)
        x = batch['xt'][:,idx]
        y = batch['yt'][:,idx]
        m = batch['mt'][:,idx]
        feed_dict = {model.xc: x,
                     model.yc: y,
                     model.mc: m,
                     model.lab: batch['lab']}
        acc = sess.run([model.acc], feed_dict)
        total_acc.append(acc)

    return np.concatenate(total_acc).mean()

# rand policy
res = []
steps = range(2,10)
for step in steps:
    acc = test(step)
    res.append(acc)

pickle.dump((steps, res), open(f'{save_dir}/res.pkl', 'wb'))

# plot
fig, axs = plt.subplots()
axs.plot(steps, res)
plt.savefig(f'{save_dir}/acc.png')
plt.close('all')

print(res)

