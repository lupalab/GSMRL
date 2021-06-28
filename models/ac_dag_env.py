import tensorflow as tf
import numpy as np

from .ACTAN import Flow

class Model(object):
    def __init__(self, sess, hps):
        self.sess = sess
        self.hps = hps

        self.flow = Flow(hps)
        self.build()

    def build(self):
        self.x = tf.placeholder(tf.float32, [None, self.hps.dimension])
        self.b = tf.placeholder(tf.float32, [None, self.hps.dimension])
        self.m = tf.placeholder(tf.float32, [None, self.hps.dimension])
        self.y = tf.placeholder(tf.float32, [None, self.hps.dimension])

        # log p (x_u | x_o)
        self.logp = self.flow.forward(self.x, self.b, self.m)

        # sample p(x_u | x_o)
        self.sam = self.flow.inverse(self.x, self.b, self.m)

        # mean of p(x_u | x_o)
        self.mean = self.flow.mean(self.x, self.b, self.m)

        # loss
        nll_loss = tf.reduce_mean(-self.logp)
        mse_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.mean-self.x)*self.m*(1-self.b), axis=1))
        loss = nll_loss * self.hps.lambda_nll + mse_loss * self.hps.lambda_mse
        tf.summary.scalar('loss', loss)

        # metric
        self.metric = self.logp

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