import logging
import numpy as np
import tensorflow.compat.v1 as tf
import os
from scipy.stats import entropy
from scipy.special import softmax

from utils.hparams import HParams
from datasets.uci import Dataset
from models.ray_model import RayACEModel
import ray

logger = logging.getLogger()


class Env(object):
    def __init__(self, hps, split):
        self.hps = hps
        self.act_size = self.hps.act_size
        self.terminal_act = self.hps.act_size - 1
        self.n_future = 2
        # build ACE model
        model_hps = HParams(f'{hps.model_dir}/params.json')
        self.model = RayACEModel.options(name="ACEModel").remote(model_hps)
        ray.get(self.model.load_weights.remote(f'{hps.model_dir}/weights.h5'))
        # build dataset
        self.dataset = Dataset(hps.dfile, split, hps.batch_size, 'remaining', True)
        self.dataset.initialize()
        if hasattr(self.dataset, 'cost'):
            self.cost = self.dataset.cost
        else:
            self.cost = np.array([self.hps.acquisition_cost] * self.hps.dimension, dtype=np.float32)

    def reset(self, loop=True, init=False):
        '''
        return state and mask
        '''
        if init:
            self.dataset.initialize()
        try:
            self.x = self.dataset.next_batch()['x']
            self.m = np.zeros_like(self.x)
            return self.x * self.m, self.m.copy()
        except:
            if loop:
                self.dataset.initialize()
                self.x = self.dataset.next_batch()['x']
                self.m = np.zeros_like(self.x)
                return self.x * self.m, self.m.copy()
            else:
                return None, None

    def _rec_reward(self, x, m):
        mse = ray.get(self.model.mse.remote(x, m))

        return -mse

    def _bpd_gain(self, x, old_m, m):
        '''
        bpd gain by acquiring new feaure
        '''
        xx = np.concatenate([x, x], axis=0)
        bb = np.concatenate([m, old_m], axis=0)
        bpd = ray.get(self.model.bpd.remote(xx, bb))
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
        sam = ray.get(self.model.sample.remote(state, mask))
        sam_mean = np.mean(sam, axis=1)
        sam_std = np.std(sam, axis=1)

        future = np.concatenate([sam_mean, sam_std], axis=-1)

        return future

    def evaluate(self, state, mask):
        mse = ray.get(self.model.mse.remote(self.x, mask))

        return {'mse': mse}

    def finetune(self, batch):
        _ = self.model.run(self.model.train_op,
                feed_dict={self.model.x: batch['x'],
                           self.model.b: batch['m'],
                           self.model.m: batch['m_next']})
