import tensorflow as tf
import numpy as np

from .ACNP import CNP

class Model(object):
    def __init__(self, sess, hps):
        self.sess = sess
        self.hps = hps

        self.cnp = CNP(hps)
        self.build()

    def build(self):
        self.xc = tf.placeholder(tf.float32, [None, None, self.hps.x_size])
        self.yc = tf.placeholder(tf.float32, [None, None, self.hps.y_size])
        self.mc = tf.placeholder(tf.float32, [None, None, 1])
        self.xt = tf.placeholder(tf.float32, [None, None, self.hps.x_size])
        self.yt = tf.placeholder(tf.float32, [None, None, self.hps.y_size])
        self.mt = tf.placeholder(tf.float32, [None, None, 1])
        self.lab = tf.placeholder(tf.int64, [None])

        self.cnp.build(self.xc, self.yc, self.mc, self.xt, self.yt, self.mt, self.lab)
        self.xent = self.cnp.xent
        self.log_likel = self.cnp.log_likel
        self.prob = self.cnp.prob
        self.pred = self.cnp.pred
        self.acc = tf.cast(tf.equal(self.pred, self.lab), tf.float32)
        self.pre_yt_loc = self.cnp.pre_yt_loc
        self.pre_yt_scale = self.cnp.pre_yt_scale
        self.post_yt_loc = self.cnp.post_yt_loc
        self.post_yt_scale = self.cnp.post_yt_scale

        # loss
        loss = -tf.reduce_mean(self.log_likel)
        tf.summary.scalar('loss', loss)

        # metric
        self.metric = self.log_likel

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

