import tensorflow as tf
import numpy as np
import copy

from .ACTAN import Flow

class Model(object):
    def __init__(self, sess, hps):
        self.sess = sess
        self.hps = hps

        params = copy.deepcopy(hps)
        self.flow = Flow(params, 'flow')
        params = copy.deepcopy(hps)
        params.dimension += hps.n_target
        self.reg = Flow(params, 'reg')
        
        self.build()

    def sample(self, x, b, m):
        B = tf.shape(x)[0]
        d = self.hps.dimension
        N = self.hps.num_samples

        x = tf.tile(tf.expand_dims(x,axis=1), [1,N,1])
        x = tf.reshape(x, [B*N,d])
        b = tf.tile(tf.expand_dims(b,axis=1), [1,N,1])
        b = tf.reshape(b, [B*N,d])
        m = tf.tile(tf.expand_dims(m,axis=1), [1,N,1])
        m = tf.reshape(m, [B*N,d])

        sam = self.flow.inverse(x, b, m)

        return sam

    def build(self):
        self.x = tf.placeholder(tf.float32, [None, self.hps.dimension])
        self.b = tf.placeholder(tf.float32, [None, self.hps.dimension])
        self.m = tf.placeholder(tf.float32, [None, self.hps.dimension])
        self.y = tf.placeholder(tf.float32, [None, self.hps.n_target])

        xy = tf.concat([self.x, self.y], axis=1)
        B = tf.shape(xy)[0]
        d = self.hps.dimension
        N = self.hps.num_samples
        Nt = self.hps.n_target

        # log p(y | x_u, x_o)
        by = tf.concat([self.m, tf.zeros((B,Nt),dtype=tf.float32)], axis=1)
        my = tf.concat([self.m, tf.ones((B,Nt),dtype=tf.float32)], axis=1)
        self.logpy = self.reg.forward(xy, by, my)
        # mean of p(y | x_u, x_o)
        mean_y = self.reg.mean(xy, by, my)
        self.mean_y = mean_y[:,-Nt:]
        # log p(y | x_o)
        bo = tf.concat([self.b, tf.zeros((B,Nt), dtype=tf.float32)], axis=1)
        mo = tf.concat([self.b, tf.ones((B,Nt), dtype=tf.float32)], axis=1)
        self.logpo = self.reg.forward(xy, bo, mo)
        # log ratio
        self.log_ratio = self.logpy - self.logpo

        # log p(x_u | x_o)
        self.logpj = self.flow.forward(self.x, self.b, self.m)
        # sample p(x_u | x_o)
        sam = self.sample(self.x, self.b, self.m)
        # sample p(y | x_u, x_o)
        y = tf.tile(tf.expand_dims(self.y, axis=1), [1,N,1])
        y = tf.reshape(y, [B*N,Nt])
        sam_xy = tf.concat([sam, y], axis=1)
        buy = tf.tile(tf.expand_dims(by,axis=1), [1,N,1])
        buy = tf.reshape(buy, [B*N,d+Nt])
        muy = tf.tile(tf.expand_dims(my,axis=1), [1,N,1])
        muy = tf.reshape(muy, [B*N,d+Nt])
        sam_j = self.reg.mean(sam_xy, buy, muy)
        self.sam_j = tf.reshape(sam_j, [B,N,d+Nt])

        # loss
        loss = -tf.reduce_mean(self.logpy) - tf.reduce_mean(self.logpj) - tf.reduce_mean(self.logpo)
        if self.hps.lambda_mse > 0:
            mse = tf.reduce_mean(tf.reduce_sum(tf.square(self.mean_y - self.y), axis=1))
            loss += self.hps.lambda_mse * mse
        tf.summary.scalar('loss', loss)

        # metric
        self.metric = self.logpy + self.logpj + self.logpo

        # train
        self.global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.train.inverse_time_decay(
            self.hps.lr, self.global_step,
            self.hps.decay_steps, self.hps.decay_rate,
            staircase=True)
        tf.summary.scalar('lr', learning_rate)
        if self.hps.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate)
        elif self.hps.optimizer == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(
                learning_rate=learning_rate)
        else:
            optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=learning_rate)
        grads_and_vars = optimizer.compute_gradients(
            loss, tf.trainable_variables())
        grads, vars_ = zip(*grads_and_vars)
        if self.hps.clip_gradient > 0:
            grads, gradient_norm = tf.clip_by_global_norm(
                grads, clip_norm=self.hps.clip_gradient)
            gradient_norm = tf.check_numerics(
                gradient_norm, "Gradient norm is NaN or Inf.")
            tf.summary.scalar('gradient_norm', gradient_norm)
        capped_grads_and_vars = zip(grads, vars_)
        self.train_op = optimizer.apply_gradients(
            capped_grads_and_vars, global_step=self.global_step)

        # summary
        self.summ_op = tf.summary.merge_all()

    def run(self, cmd, feed_dict):
        out = self.sess.run(cmd, feed_dict)

        return out










