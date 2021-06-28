import os
import sys
p = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(p)
import numpy as np
from scipy.stats import entropy
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
from cdfa_modules.gp_env import Env as GPEnv
from cdfa_modules.img_env import Dataset as ImgDataset
from cdfa_modules.img_env import Env as ImgEnv

parser = argparse.ArgumentParser()
parser.add_argument('--cfg_file', type=str)
parser.add_argument('--grid', type=int, default=4)
args = parser.parse_args()
params = HParams(args.cfg_file)
params.batch_size=100
pprint(params.dict)

os.environ['CUDA_VISIBLE_DEVICES'] = params.gpu
np.random.seed(params.seed)
tf.set_random_seed(params.seed)

save_dir = os.path.join(params.exp_dir, f'test_greedy_grid{args.grid}')
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

###################################################################

def init():
    X = np.linspace(-2., 2., args.grid)
    X = np.repeat(np.expand_dims(X, axis=0), params.batch_size, axis=0)
    X = np.expand_dims(X, axis=-1)
    X0 = np.zeros([params.batch_size,1,1])
    X = np.concatenate([X0,X], axis=1)

    Y0 = testset.gp0.sample(X)
    Lab0 = np.zeros([params.batch_size], dtype=np.int64)
    Y1 = testset.gp1.sample(X)
    Lab1 = np.ones([params.batch_size], dtype=np.int64)

    xt = np.concatenate([X[:,1:],X[:,1:]])
    yt = np.concatenate([Y0[:,1:],Y1[:,1:]])
    lab =  np.concatenate([Lab0,Lab1])
    
    x0 = np.concatenate([X[:,:1],X[:,:1]], axis=0)
    y0 = np.concatenate([Y0[:,:1],Y1[:,:1]], axis=0)
    m0 = np.ones([params.batch_size*2,1,1], dtype=np.float32)

    return x0,y0,m0,xt,yt,lab

def run_step(xc, yc, mc, xt, yt, mt):
    xc = np.expand_dims(xc, axis=0)
    yc = np.expand_dims(yc, axis=0)
    mc = np.expand_dims(mc, axis=0)
    xt = np.expand_dims(xt, axis=0)
    yt = np.expand_dims(yt, axis=0)
    mt = np.expand_dims(mt, axis=0)

    pre_prob = sess.run(model.prob,
                    feed_dict={model.xc: xc,
                               model.yc: yc,
                               model.mc: mc})
    ym, ys = sess.run([model.pre_yt_loc, model.pre_yt_scale],
                    feed_dict={model.xc: xc,
                               model.yc: yc,
                               model.mc: mc,
                               model.xt: xt})
    candidates = np.where(mt[0,:,0]==0)[0]
    rewards = []
    for idx in candidates:
        cmi = []
        for _ in range(10):
            r = np.random.randn(1,1,ym.shape[-1])
            x = xt[:,idx,None]
            y = ym[:,idx,None]+ys[:,idx,None]*r
            m = np.ones([1,1,1], dtype=np.float32)
            x = np.concatenate([xc,x], axis=1)
            y = np.concatenate([yc,y], axis=1)
            m = np.concatenate([mc,m], axis=1)
            post_prob = sess.run(model.prob,
                        feed_dict={model.xc: x,
                                model.yc: y,
                                model.mc: m})
            kl = entropy(post_prob.T, pre_prob.T)
            cmi.append(kl)
        cmi = np.mean(cmi)
        rewards.append(cmi)
    idx = candidates[np.argmax(rewards)]
    xc = np.concatenate([xc, xt[:,idx,None]], axis=1)
    yc = np.concatenate([yc, yt[:,idx,None]], axis=1)
    mc = m.copy()
    mt = mt.copy()
    mt[:,idx,0] = xc.shape[1]

    return xc[0], yc[0], mc[0], mt[0]

def predict(xc,yc,mc,lab):
    xc = np.expand_dims(xc, axis=0)
    yc = np.expand_dims(yc, axis=0)
    mc = np.expand_dims(mc, axis=0)
    lab = np.expand_dims(lab, axis=0)
    # print(xc.shape, xc[0,:,0])
    # print(yc.shape, yc[0,:,0])

    acc = sess.run(model.acc,
            feed_dict={model.xc: xc,
                       model.yc: yc,
                       model.mc: mc,
                       model.lab: lab})
    # print(acc.shape)
    # print(acc)
    
    return acc[0]


def test(fname):
    xc, yc, mc, xt, yt, lab = init()
    mt = np.zeros([xt.shape[0],xt.shape[1],1], dtype=np.float32)
    preds = []

    acc = sess.run(model.acc,
            feed_dict={model.xc: xt,
                       model.yc: yt,
                       model.mc: np.ones([xt.shape[0],xt.shape[1],1],dtype=np.float32),
                       model.lab: lab})
    # print('xt:',xt[0,:,0])
    # print('yt:',yt[0,:,0])
    # print('acc:', acc[0])

    for i in range(xt.shape[0]):
        labi = lab[i]
        xti, yti, mti = xt[i], yt[i], mt[i]
        xci, yci, mci = xc[i], yc[i], mc[i]
        acc = []
        for step in range(args.grid):
            xci, yci, mci, mti = run_step(xci, yci, mci, xti, yti, mti)
            # print(f'step:{step}')
            # print(f'xc: {xci[:,0]}')
            # print(f'yc: {yci[:,0]}')
            # print(f'mc: {mci[:,0]}')
            # print(f'mt: {mti[:,0]}')
            mt[i] = mti.copy()
            p = predict(xci, yci, mci, labi)
            # print(f'acc: {p}')
            acc.append(p)
        preds.append(acc)
    preds = np.array(preds)

    with open(fname, 'wb') as f:
        pickle.dump({
            'xt': xt,
            'yt': yt,
            'mt': mt,
            'preds': preds
        }, f)

    return preds

res = test(f'{save_dir}/results.pkl')
x = range(2, args.grid+2)
y = np.mean(res, axis=0)
fig = plt.figure()
plt.plot(x, y, marker='x')
plt.savefig(f'{save_dir}/acc.png')
plt.close('all')