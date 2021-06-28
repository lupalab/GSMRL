import tensorflow as tf
import numpy as np

from .ACTAN import Flow

class Model(object):
    def __init__(self, sess, hps):
        self.sess = sess
        self.hps = hps

        self.flow = Flow(hps)
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
        sam = tf.reshape(sam, [B,N,d])

        return sam

    def build(self):
        self.x = tf.placeholder(tf.float32, [None, self.hps.dimension])
        self.b = tf.placeholder(tf.float32, [None, self.hps.dimension])
        self.m = tf.placeholder(tf.float32, [None, self.hps.dimension])

        B = tf.shape(self.x)[0]
        ones = tf.ones([B, self.hps.dimension], dtype=tf.float32)

        self.logpo = self.flow.forward(self.x, self.b, ones)
        self.logpu = self.flow.forward(self.x, self.m, ones)
        q = self.m * (1-self.b)
        self.logpi = self.flow.forward(self.x, self.b, 1-q)

        # bpd p(x_u | x_o)
        self.bpd = -self.logpo / (tf.reduce_sum(1-self.b, axis=1) + 1e-8)

        # sample p(x_u | x_o)
        self.sam = self.sample(self.x, self.b, ones)
        self.mean = self.flow.mean(self.x, self.b, ones)
        self.mse = tf.reduce_sum(tf.square(self.mean - self.x), axis=1)

        # log ratio
        self.log_ratio = self.logpu - self.logpi

        # loss
        loss = -tf.reduce_mean(self.logpo) - tf.reduce_mean(self.logpu) - tf.reduce_mean(self.logpi)
        if self.hps.lambda_mse > 0:
            mse = tf.reduce_mean(self.mse)
            loss += self.hps.lambda_mse * mse
        tf.summary.scalar('loss', loss)

        # metric
        self.metric = self.logpo + self.logpu + self.logpi

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

    