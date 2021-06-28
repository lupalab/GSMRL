import tensorflow as tf
import numpy as np
import math
import pickle
import random

def _parse(i, x, y, d):
    true_miss = np.array([])
    for val in x:
        if np.isnan(val):
            true_miss = np.append(true_miss, 0.0)
        else:
            true_miss = np.append(true_miss, 1.0)

    true_miss = true_miss.astype(np.float32)
    b = np.zeros([d], dtype=np.float32)
    not_miss = []
    for idx, val in enumerate(true_miss):
        if val == 1:
            not_miss.append(idx)
    not_miss = np.array(not_miss)
    no = np.random.choice(len(not_miss))
    o = np.random.choice(len(not_miss), [no], replace=False)
    ele = random.choice(not_miss)
    while ele in not_miss[o]:
        ele = random.choice(not_miss)
    b[not_miss[o]] = 1.0
    m = b.copy()
    a = not_miss[o]
    a = np.append(a, ele)
    m[a] = 1.0
    for idx,val in enumerate(x):
        if np.isnan(val):
            x[idx] = 0.0
    x = x.astype(np.float32)
    return i, x, y, b, m

class Dataset(object):
    def __init__(self, dfile, split, batch_size):
        super().__init__()

        with open(dfile, 'rb') as f:
            data_dict = pickle.load(f)
        data, label = data_dict[split]
        self.size = data.shape[0]
        self.d = data.shape[1]
        self.num_batches = math.ceil(self.size / batch_size)

        ind = tf.range(self.size, dtype=tf.int64)
        dst = tf.data.Dataset.from_tensor_slices((ind, data, label))
        if split == 'train':
            dst = dst.shuffle(self.size)
        dst = dst.map(lambda i, x, y: tuple(
            tf.py_func(_parse, [i, x, y, self.d], 
            [tf.int64, tf.float32, tf.float32, tf.float32, tf.float32])),
            num_parallel_calls=16)
        dst = dst.batch(batch_size)
        dst = dst.prefetch(1)
        dst_it = dst.make_initializable_iterator()
        self.i, self.x, self.y, self.b, self.m = dst_it.get_next()
        self.initializer = dst_it.initializer

    def initialize(self, sess):
        sess.run(self.initializer)