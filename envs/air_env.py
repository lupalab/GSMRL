import logging
import numpy as np
import tensorflow as tf
from scipy.stats import entropy
from scipy.special import softmax

from utils.hparams import HParams
from models import get_model
from datasets.vec import Dataset

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
            self.model = get_model(self.sess, model_hps)
            # restore weights
            self.saver = tf.train.Saver()
            restore_from = f'{hps.model_dir}/weights/params.ckpt'
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
        # mse = self.model.run(self.model.mse,
        #             feed_dict={self.model.x: x,
        #                        self.model.b: m,
        #                        self.model.m: m})
        mse = self.model.mse(x, m)

        return -mse

    def _bpd_gain(self, x, old_m, m):
        '''
        bpd gain by acquiring new feaure
        '''
        xx = np.concatenate([x, x], axis=0)
        bb = np.concatenate([m, old_m], axis=0)
        # bpd = self.model.run(self.model.bpd,
        #         feed_dict={self.model.x: xx,
        #                    self.model.b: bb,
        #                    self.model.m: bb})
        bpd = self.model.bpd(xx, bb)
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
        # sam = self.model.run(self.model.sam,
        #         feed_dict={self.model.x: state,
        #                    self.model.b: mask,
        #                    self.model.m: np.ones_like(mask)})
        sam = self.model.sample(state, mask)
        sam_mean = np.mean(sam, axis=1)
        sam_std = np.std(sam, axis=1)

        future = np.concatenate([sam_mean, sam_std], axis=-1)

        return future

    def evaluate(self, state, mask):
        # mse = self.model.run(self.model.mse,
        #             feed_dict={self.model.x: self.x,
        #                        self.model.b: mask,
        #                        self.model.m: mask})
        mse = self.model.mse(self.x, mask)

        return {'mse': mse}

    def finetune(self, batch):
        _ = self.model.run(self.model.train_op,
                feed_dict={self.model.x: batch['x'],
                           self.model.b: batch['m'],
                           self.model.m: batch['m_next']})
