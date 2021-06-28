import tensorflow as tf
import numpy as np

from .ACTAN import Flow

class Model(object):
    def __init__(self, sess, hps):
        self.sess = sess
        self.hps = hps

        self.flow = Flow(hps)
        self.build()

    def classify(self, x, b, m):
        B = tf.shape(x)[0]
        d = self.hps.dimension
        N = self.hps.n_classes

        x = tf.tile(tf.expand_dims(x,axis=1), [1,N,1])
        x = tf.reshape(x, [B*N,d])
        b = tf.tile(tf.expand_dims(b,axis=1), [1,N,1])
        b = tf.reshape(b, [B*N,d])
        m = tf.tile(tf.expand_dims(m,axis=1), [1,N,1])
        m = tf.reshape(m, [B*N,d])
        y = tf.tile(tf.expand_dims(tf.range(N),axis=0), [B,1])
        y = tf.reshape(y, [B*N])

        # log p(x_u | x_o, y)
        logp = self.flow.cond_forward(x, y, b, m)
        # logits
        logits = tf.reshape(logp, [B,N])

        return logits

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
        y = tf.random_uniform([B*N], dtype=tf.int64, minval=0, maxval=self.hps.n_classes)

        sam = self.flow.cond_inverse(x, y, b, m)
        sam = tf.reshape(sam, [B,N,d])

        return sam

    def cond_sample(self, x, y, b, m):
        B = tf.shape(x)[0]
        d = self.hps.dimension
        N = self.hps.num_samples

        x = tf.tile(tf.expand_dims(x,axis=1), [1,N,1])
        x = tf.reshape(x, [B*N,d])
        b = tf.tile(tf.expand_dims(b,axis=1), [1,N,1])
        b = tf.reshape(b, [B*N,d])
        m = tf.tile(tf.expand_dims(m,axis=1), [1,N,1])
        m = tf.reshape(m, [B*N,d])
        y = tf.tile(tf.expand_dims(y,axis=1), [1,N])
        y = tf.reshape(y, [B*N])

        sam = self.flow.cond_inverse(x, y, b, m)
        sam = tf.reshape(sam, [B,N,d])

        return sam

    def build(self):
        self.x = tf.placeholder(tf.float32, [None, self.hps.dimension])
        self.b = tf.placeholder(tf.float32, [None, self.hps.dimension])
        self.m = tf.placeholder(tf.float32, [None, self.hps.dimension])
        self.y = tf.placeholder(tf.float32, [None])
        y = tf.cast(self.y, tf.int64)

        ones = tf.ones(tf.shape(self.x), dtype=tf.float32)
        zeros = tf.zeros(tf.shape(self.x), dtype=tf.float32)

        # class weights
        class_weights = np.array(self.hps.class_weights, dtype=np.float32)
        class_weights /= np.sum(class_weights)
        class_weights = np.log(class_weights)

        # log p(x_u | x_o, y)
        self.logpu = self.classify(self.x, self.b, ones)
        # log p(x_o | y)
        self.logpo = self.classify(self.x, zeros, self.b)
        # logits
        self.logits = self.logpo + class_weights

        # p(y | x_u, x_o)
        self.prob = tf.nn.softmax(self.logits)
        self.pred = tf.argmax(self.prob, axis=1)
        self.acc = tf.cast(tf.equal(self.pred, y), tf.float32)

        # log p(x_u | x_o)
        self.log_likel = (tf.reduce_logsumexp(self.logpu + self.logpo + class_weights, axis=1) - 
                          tf.reduce_logsumexp(self.logpo + class_weights, axis=1))
        # sample p(x_u | x_o)
        self.sam = self.sample(self.x, self.b, ones)

        # log p(x_u | x_o, y) for real y
        self.cond_logpu = self.flow.cond_forward(self.x, y, self.b, ones)
        # sample p(x_u | x_o, y)
        self.cond_sam = self.cond_sample(self.x, y, self.b, ones)
        # sample p(x_u | x_o, y) based on predicted y
        self.pred_sam = self.cond_sample(self.x, self.pred, self.b, ones)

        # loss
        self.xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=y)
        xent = tf.reduce_mean(self.xent)
        tf.summary.scalar('xent', xent)
        nll = tf.reduce_mean(-self.log_likel)
        # nll = tf.reduce_mean(-self.cond_logpu)
        tf.summary.scalar('nll', nll)
        loss = xent * self.hps.lambda_xent + self.hps.lambda_nll * nll
        tf.summary.scalar('loss', loss)

        # metric
        self.metric = self.acc

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