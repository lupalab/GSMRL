#!/usr/bin/env python3

import ray

@ray.remote(num_gpus=1)
class RayACEModel():
    def __init__(self, hps):
        import tensorflow as tf
        from models.ace import ACEModel
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = hps.gpu
        self.hps = hps
        self.model = ACEModel(hps.dimension)
        self._sample = tf.function(self.model.sample)
        self._bpd = tf.function(self.model.bpd, input_signature=[tf.TensorSpec(shape=(None, hps.dimension)),
                                                                 tf.TensorSpec(shape=(None, hps.dimension))])
        self._mse = tf.function(self.model.mse, input_signature=[tf.TensorSpec(shape=(None, hps.dimension)),
                                                                 tf.TensorSpec(shape=(None, hps.dimension))])

    def load_weights(self, model_path):
        self.model.load_weights(model_path)

    def sample(self, x_o, mask):
        return self._sample(x_o, mask, num_samples=self.hps.num_samples, use_proposal=True).numpy()

    def bpd(self, x, mask):
        return self._bpd(x, mask).numpy()

    def mse(self, x, mask):
        return self._mse(x, mask).numpy()
