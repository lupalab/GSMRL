import os
import sys
import logging
import numpy as np
import tensorflow as tf
from scipy.stats import entropy
from scipy.special import softmax
from easydict import EasyDict as edict

from utils.hparams import HParams
from datasets.vec import Dataset

acflow_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
sys.path.append(acflow_path)
from ACFlow_DFA.models.flow import Model

logger = logging.getLogger()

class Env(object):
    def __init__(self, hps, split):
        self.hps = hps
        self.act_size = self.hps.act_size
        self.terminal_act = self.hps.act_size - 1
        self.n_future = 2

        g = tf.Graph()
        with g.as_default():
            # open a session
            config = tf.ConfigProto()
            config.log_device_placement = True
            config.allow_soft_placement = True
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config, graph=g)
            # build ACFlow model
            model_hps = HParams(f'{hps.model_dir}/params.json')
            self.model = Model(model_hps)
            self.x_ph = tf.placeholder(tf.float32, [None]+model_hps.image_shape)
            self.b_ph = tf.placeholder(tf.float32, [None]+model_hps.image_shape)
            self.m_ph = tf.placeholder(tf.float32, [None]+model_hps.image_shape)
            self.y_ph = tf.placeholder(tf.float32, [None])
            x_ph = tf.cast(self.x_ph, tf.uint8)
            b_ph = tf.cast(self.b_ph, tf.uint8)
            m_ph = tf.cast(self.m_ph, tf.uint8)
            y_ph = tf.cast(self.y_ph, tf.int64)
            dummyset = edict()
            dummyset.x, dummyset.b, dummyset.m, dummyset.y = x_ph, b_ph, m_ph, y_ph
            self.model.build(dummyset, dummyset, dummyset)
            # restore weights
            self.saver = tf.train.Saver()
            weights_dir = f'{hps.model_dir}/weights/'
            restore_from = tf.train.latest_checkpoint(weights_dir)
            logger.info(f'restore from {restore_from}')
            self.saver.restore(self.sess, restore_from)
            # build dataset
            self.dataset = Dataset(hps.dfile, split, hps.episode_workers)
            self.dataset.initialize(self.sess)
            if hasattr(self.dataset, 'cost'):
                self.cost = self.dataset.cost
            else:
                self.cost = np.array([self.hps.acquisition_cost] * self.hps.dimension, dtype=np.float32)
    
    def reset(self, loop=True, init=False):
        '''
        return state and mask
        '''
        if init:
            self.dataset.initialize(self.sess)
        try:
            self.x, self.y = self.sess.run([self.dataset.x, self.dataset.y])
            self.m = np.zeros_like(self.x)
            return self.x * self.m, self.m.copy()
        except:
            if loop:
                self.dataset.initialize(self.sess)
                self.x, self.y = self.sess.run([self.dataset.x, self.dataset.y])
                self.m = np.zeros_like(self.x)
                return self.x * self.m, self.m.copy()
            else:
                return None, None

    def _rec_reward(self, x, m):
        xs = np.reshape(x*255, [x.shape[0]]+self.hps.image_shape)
        ms = np.reshape(m, [m.shape[0]]+self.hps.image_shape)
        mse = self.sess.run(self.model.mse,
                    feed_dict={self.x_ph: xs,
                               self.b_ph: ms,
                               self.m_ph: ms})
        
        return -mse

    def _bpd_gain(self, x, old_m, m):
        xx = np.concatenate([x, x], axis=0)
        bb = np.concatenate([m, old_m], axis=0)
        xxs = np.reshape(xx*255, [xx.shape[0]]+self.hps.image_shape)
        bbs = np.reshape(bb, [bb.shape[0]]+self.hps.image_shape)
        llu = self.sess.run(self.model.test_llu,
                feed_dict={self.x_ph: xxs,
                           self.b_ph: bbs,
                           self.m_ph: bbs})
        num = np.sum(1.-bb, axis=1)
        bpd = llu / (num + 1e-8)
        post_bpd, pre_bpd = np.split(bpd, 2, axis=0)

        gain = pre_bpd - post_bpd

        return gain
    
    def step(self, action):
        empty = action == -1
        terminal = action == self.terminal_act
        normal = np.logical_and(~empty, ~terminal)
        reward = np.zeros([action.shape[0]], dtype=np.float32)
        done = np.zeros([action.shape[0]], dtype=np.bool)
        if np.any(empty):
            done[empty] = True
            reward[empty] = 0.
        if np.any(terminal):
            done[terminal] = True
            x = self.x[terminal]
            m = self.m[terminal]
            reward[terminal] = self._rec_reward(x, m)
        if np.any(normal):
            x = self.x[normal]
            a = action[normal]
            m = self.m[normal]
            old_m = m.copy()
            assert np.all(old_m[np.arange(len(a)), a] == 0)
            m[np.arange(len(a)), a] = 1.
            self.m[normal] = m.copy() # explicitly update m
            acquisition_cost = self.cost[a]
            bpd_gain = self._bpd_gain(x, old_m, m)
            reward[normal] = bpd_gain - acquisition_cost

        return self.x * self.m, self.m.copy(), reward, done

    def peek(self, state, mask):
        xs = np.reshape(state*255, [state.shape[0]]+self.hps.image_shape)
        ms = np.reshape(mask, [mask.shape[0]]+self.hps.image_shape)
        sam = self.sess.run(self.model.sam,
                feed_dict={self.x_ph: xs,
                           self.b_ph: ms,
                           self.m_ph: np.ones_like(ms)})
        sam = sam.astype(np.float32) / 255
        sam_mean = np.reshape(np.mean(sam, axis=1), [sam.shape[0],-1])
        sam_std = np.reshape(np.std(sam, axis=1), [sam.shape[0],-1])

        future = np.concatenate([sam_mean, sam_std], axis=-1)

        return future

    def evaluate(self, state, mask):
        xs = np.reshape(self.x*255, [state.shape[0]]+self.hps.image_shape)
        ms = np.reshape(mask, [mask.shape[0]]+self.hps.image_shape)
        mse = self.sess.run(self.model.mse,
                    feed_dict={self.x_ph: xs,
                               self.b_ph: ms,
                               self.m_ph: ms})

        return {'mse': mse}

    def finetune(self, batch):
        raise NotImplementedError()