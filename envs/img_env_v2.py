import os
import sys
import importlib
import logging
import numpy as np
import tensorflow as tf
from scipy.stats import entropy
from scipy.special import softmax
from easydict import EasyDict as edict

from utils.hparams import HParams
from datasets.vec import Dataset

# acflow_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
# sys.path.append(acflow_path)
# from ACFlow_AFA.models.flow_classifier import Model

p = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(p)
from models.acflow_classifier import Model

logger = logging.getLogger()

class Env(object):
    def __init__(self, hps, split):
        self.hps = hps
        self.act_size = self.hps.act_size
        self.max_acquisition = self.hps.max_acquisition
        self.n_future = 5
        self.task = 'cls'

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

    def _cls_reward(self, x, m, y, p):
        '''
        calculate the cross entropy loss as reward
        '''
        xs = np.reshape(x*255, [x.shape[0]]+self.hps.image_shape)
        ms = np.reshape(m, [m.shape[0]]+self.hps.image_shape)
        xent_acflow = self.sess.run(self.model.xent, 
                    feed_dict={self.x_ph: xs,
                               self.b_ph: ms,
                               self.m_ph: ms,
                               self.y_ph: y})

        xent_policy = -np.log(softmax(p)[np.arange(len(p)), y.astype(np.int64)])

        xent = np.minimum(xent_acflow, xent_policy)
        
        return -xent

    def _info_gain(self, x, old_m, m, y):
        '''
        information gain by acquiring new feaure
        entropy reduction
        '''
        xx = np.concatenate([x, x], axis=0)
        bb = np.concatenate([m, old_m], axis=0)
        xxs = np.reshape(xx*255, [xx.shape[0]]+self.hps.image_shape)
        bbs = np.reshape(bb, [bb.shape[0]]+self.hps.image_shape)
        prob = self.sess.run(self.model.prob,
                   feed_dict={self.x_ph: xxs,
                              self.b_ph: bbs,
                              self.m_ph: bbs})
        post_prob, pre_prob = np.split(prob, 2, axis=0)
        ig = entropy(pre_prob.T) - 0.99 * entropy(post_prob.T)

        return ig

    def step(self, action, prediction):
        empty = action == -1
        terminal = np.sum(self.m, axis=1) == self.max_acquisition
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
            p = prediction[terminal]
            reward[terminal] = self._cls_reward(x, m, y, p)
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
        xs = np.reshape(state*255, [state.shape[0]]+self.hps.image_shape)
        ms = np.reshape(mask, [mask.shape[0]]+self.hps.image_shape)
        logits, sam, pred_sam = self.sess.run(
                [self.model.logpo, self.model.sam, self.model.pred_sam],
                feed_dict={self.x_ph: xs,
                           self.b_ph: ms,
                           self.m_ph: np.ones_like(ms)})
        sam = sam.astype(np.float32) / 255
        pred_sam = pred_sam.astype(np.float32) / 255
        sam_mean = np.reshape(np.mean(sam, axis=1), [sam.shape[0],-1])
        sam_std = np.reshape(np.std(sam, axis=1), [sam.shape[0],-1])
        pred_sam_mean = np.reshape(np.mean(pred_sam, axis=1), [pred_sam.shape[0],-1])
        pred_sam_std = np.reshape(np.std(pred_sam, axis=1), [pred_sam.shape[0],-1])

        prob = softmax(logits, axis=-1)
        prob = np.max(prob, axis=-1, keepdims=True)
        prob = np.ones_like(state) * prob

        future = np.concatenate([prob, sam_mean, sam_std, pred_sam_mean, pred_sam_std], axis=-1)

        return future

    def evaluate(self, state, mask, prediction):
        xs = np.reshape(state*255, [state.shape[0]]+self.hps.image_shape)
        ms = np.reshape(mask, [mask.shape[0]]+self.hps.image_shape)
        acc_acflow = self.sess.run(self.model.acc,
                    feed_dict={self.x_ph: xs,
                               self.b_ph: ms,
                               self.m_ph: ms,
                               self.y_ph: self.y})

        pred = np.argmax(prediction, axis=1)
        acc_policy = (pred == self.y).astype(np.float32)

        # final reward
        cost = np.mean(mask, axis=1)
        reward_acflow = acc_acflow - cost
        reward_policy = acc_policy - cost

        return {'acc_acflow': acc_acflow, 
                'acc_policy': acc_policy,
                'reward_acflow': reward_acflow, 
                'reward_policy': reward_policy}

    def finetune(self, batch):
        raise NotImplementedError()