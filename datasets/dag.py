import os
import tensorflow as tf
import numpy as np
import math
import pickle

def _parse(x, d):
    b = np.zeros([d], dtype=np.float32)
    no = np.random.choice(d)
    o = np.random.choice(d, [no], replace=False)
    b[o] = 1.
    m = b.copy()
    w = list(np.where(b == 0)[0])
    w = np.random.choice(w)
    m[w] = 1.

    return x, b, m

class Dataset(object):
    def __init__(self, file_path, exp_id, split, batch_size):
        random = np.random.RandomState(42)

        # Load the graph
        self.adjacency = np.load(os.path.join(file_path, "DAG{}.npy".format(exp_id)))
        
        # Load data
        data_path = os.path.join(file_path, "data{}.npy".format(exp_id))
        data = np.load(data_path).astype(np.float32)

        # Determine train/test partitioning
        train_samples = int(data.shape[0] * 0.8)
        test_samples = data.shape[0] - train_samples
        
        # Shuffle and filter examples
        # shuffle_idx = np.arange(data.shape[0])
        # random.shuffle(shuffle_idx)
        # data = data[shuffle_idx]

        # Train/Test split
        if split == 'train':
            self.data = data[:train_samples]
        else:
            self.data = data[train_samples:]

        self.size = self.data.shape[0]
        self.d = self.data.shape[1]
        self.num_batches = math.ceil(self.size / batch_size)

        dst = tf.data.Dataset.from_tensor_slices(self.data)
        if split == 'train':
            dst = dst.shuffle(self.size)
        dst = dst.map(lambda x: tuple(
            tf.py_func(_parse, [x, self.d], 
            [tf.float32, tf.float32, tf.float32])),
            num_parallel_calls=16)
        dst = dst.batch(batch_size)
        dst = dst.prefetch(1)
        dst_it = dst.make_initializable_iterator()
        self.x, self.b, self.m = dst_it.get_next()
        self.y = tf.zeros_like(self.b)
        self.initializer = dst_it.initializer

    def initialize(self, sess):
        sess.run(self.initializer)