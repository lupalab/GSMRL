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

save_dir = os.path.join(params.exp_dir, 'test_np')
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

#################################################################

batch = testset.sample()
feed_dict = {model.xc: batch['xt'],
             model.yc: batch['yt'],
             model.mc: batch['mt'],
             model.xt: batch['xt'],
             model.yt: batch['yt'],
             model.mt: batch['mt'],
             model.lab: batch['lab']}
acc = sess.run([model.acc], feed_dict)

logging.info(f'acc (xt): {np.mean(acc)}')


feed_dict = {model.xc: batch['xc'],
             model.yc: batch['yc'],
             model.mc: batch['mc'],
             model.xt: batch['xt'],
             model.yt: batch['yt'],
             model.mt: batch['mt'],
             model.lab: batch['lab']}
acc, mean, scale = sess.run([model.acc, model.pre_yt_loc, model.pre_yt_scale], feed_dict)
batch['acc'] = acc
batch['mean'] = mean
batch['scale'] = scale

with gzip.open(f'{save_dir}/results.pgz', 'wb') as f:
    pickle.dump(batch, f)

logging.info(f'acc (xc): {np.mean(acc)}')


def plot_functions(target_x, target_y, context_x, context_y, pred_y, std, fname):
    """Plots the predicted mean and variance and the context points.

    Args: 
      target_x: An array of shape [num_targets,1] that contains the
          x values of the target points.
      target_y: An array of shape [num_targets,1] that contains the
          y values of the target points.
      context_x: An array of shape [num_contexts,1] that contains 
          the x values of the context points.
      context_y: An array of shape [num_contexts,1] that contains 
          the y values of the context points.
      pred_y: An array of shape [num_targets,1] that contains the
          predicted means of the y values at the target points in target_x.
      std: An array of shape [num_targets,1] that contains the
          predicted std dev of the y values at the target points in target_x.
    """
    # sort
    idx = np.argsort(target_x[:,0])
    target_x = target_x[idx]
    target_y = target_y[idx]
    pred_y = pred_y[idx]
    std = std[idx]
    # Plot everything
    plt.plot(target_x, pred_y, 'b', linewidth=2)
    plt.plot(target_x, target_y, 'k:', linewidth=2)
    plt.plot(context_x, context_y, 'ko', markersize=10)
    plt.fill_between(
        target_x[:, 0],
        pred_y[:, 0] - std[:, 0],
        pred_y[:, 0] + std[:, 0],
        alpha=0.2,
        facecolor='#65c9f7',
        interpolate=True)
    # Make the plot pretty
    # plt.yticks([-2, 0, 2], fontsize=16)
    # plt.xticks([-2, 0, 2], fontsize=16)
    # plt.ylim([-2, 2])
    plt.grid('off')
    ax = plt.gca()
    plt.savefig(fname)
    plt.close('all')


def plot_image(xt, yt, xc, yc, ym, ys, fname):
    im = ym.reshape([28,28])
    fig, axs = plt.subplots()
    axs.imshow(im)
    plt.savefig(fname)
    plt.close('all')


for i in range(acc.shape[0]):
    xc = batch['xc'][i]
    yc = batch['yc'][i]
    xt = batch['xt'][i]
    yt = batch['yt'][i]
    ym = batch['mean'][i]
    ys = batch['scale'][i]
    if params.dataset == 'gp':
        plot_functions(xt, yt, xc, yc, ym, ys, f'{save_dir}/{i}.png')
    elif params.dataset == 'img':
        plot_image(xt, yt, xc, yc, ym, ys, f'{save_dir}/{i}.png')
    else:
        raise NotImplementedError()
