import logging
import numpy as np
import tensorflow as tf

from utils.hparams import HParams
from models import get_model
from datasets.vec import Dataset

logger = logging.getLogger()


class Env(object):
    def __init__(self, hps, split):
        self.hps = hps
        self.act_size = self.hps.act_size
        self.terminal_act = self.hps.act_size - 1

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


    def _reg_reward(self, x, m, y):
        '''
        calculate the MSE as reward
        '''
        mse = self.model.run(self.model.mse, 
                    feed_dict={self.model.x: x,
                               self.model.b: m,
                               self.model.m: m,
                               self.model.y: y})
        
        return -mse

    
    def _info_gain_sd(self, x, old_m, m, y):
        '''
        information gain by acquiring new feaure
        entropy reduction
        '''
        B, d = x.shape
        N = 21
        bin_width = 1. / (N-1)
        xx = np.concatenate([x, x], axis=0)
        xx = np.repeat(np.expand_dims(xx, axis=1), N, axis=1)
        xx = xx.reshape([2*B*N, d])
        bb = np.concatenate([m, old_m], axis=0)
        bb = np.repeat(np.expand_dims(bb, axis=1), N, axis=1)
        bb = bb.reshape([2*B*N, d])
        yy = np.linspace(0., 1., N)
        yy = np.repeat(np.expand_dims(yy, axis=0), 2*B, axis=0)
        yy = yy.reshape([2*B*N, 1])
        hist = self.model.run(self.model.logpy,
                    feed_dict={self.model.x: xx,
                               self.model.b: bb,
                               self.model.m: bb,
                               self.model.y: yy})
        hist = hist.reshape([2*B, N])
        post_hist, pre_hist = np.split(hist, 2, axis=0)
        
        post_ent = - bin_width * np.sum(np.exp(post_hist)*post_hist, axis=1)
        pre_ent = - bin_width * np.sum(np.exp(pre_hist)*pre_hist, axis=1)
        ig = pre_ent - post_ent

        return ig

    
    def _info_gain(self, x, old_m, m, y):
        '''
        information gain by acquiring new feaure
        entropy reduction
        '''
        xx = np.concatenate([x, x], axis=0)
        bb = np.concatenate([m, old_m], axis=0)
        yy = np.concatenate([y, y], axis=0)
        sam_y = self.model.run(self.model.sam_y,
                    feed_dict={self.model.x: xx,
                               self.model.b: bb,
                               self.model.m: bb,
                               self.model.y: yy})
        post_y, pre_y = np.split(sam_y, 2, axis=0)
        post_var = np.var(post_y, axis=1)
        pre_var = np.var(pre_y, axis=1)
        post_ent = np.sum(0.5*np.log(2.*np.pi*np.e*post_var), axis=1)
        pre_ent = np.sum(0.5*np.log(2.*np.pi*np.e*pre_var), axis=1)
        ig = pre_ent - post_ent

        return ig

    
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
            y = self.y[terminal]
            m = self.m[terminal]
            reward[terminal] = self._reg_reward(x, m, y)
        if np.any(normal):
            x = self.x[normal]
            y = self.y[normal]
            a = action[normal]
            m = self.m[normal]
            old_m = m.copy()
            assert np.all(old_m[np.arange(len(a)), a] == 0)
            m[np.arange(len(a)), a] = 1.
            self.m[normal] = m.copy() # explicitly update m
            acquisition_cost = self.cost[a]
            info_gain = self._info_gain(x, old_m, m, y)
            reward[normal] = info_gain - acquisition_cost

        return self.x * self.m, self.m.copy(), reward, done
        

    def peek(self, state, mask):
        future = self.model.run(self.model.sam,
                     feed_dict={self.model.x: state,
                                self.model.b: mask,
                                self.model.m: np.ones_like(mask),
                                self.model.y: self.y})
        future = np.mean(future, axis=1)

        return future


    def evaluate(self, state, mask):
        mse = self.model.run(self.model.mse,
                    feed_dict={self.model.x: state,
                               self.model.b: mask,
                               self.model.m: mask,
                               self.model.y: self.y})

        return {'mse': mse}

    
    def finetune(self, batch):
        _ = self.model.run(self.model.train_op,
                feed_dict={self.model.x: batch['x'],
                           self.model.y: batch['y'],
                           self.model.b: batch['m'],
                           self.model.m: batch['m_next']})