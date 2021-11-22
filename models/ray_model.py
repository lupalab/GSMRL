#!/usr/bin/env python3

import ray

@ray.remote
class RayACEModel():
    def __init__(self, hps):
        import tensorflow as tf
        from models.ace import ACEModel
        self.hps = hps
        self.model = ACEModel(hps.dimension)
        self.model.load_weights(hps.exp_dir + '/weights.h5')

    def sample(self, x_o, mask):
        return self.model.sample(x_o, mask, num_samples=self.hps.num_samples).numpy()

    def bpd(self, x, mask):
        return self.model.bpd(x, mask).numpy()

    def mse(self, x, mask):
        return self.model.mse(x, mask).numpy()
