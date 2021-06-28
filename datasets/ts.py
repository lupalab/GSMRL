import tensorflow as tf
import numpy as np
import math
import pickle

def _parse0(i, x, y, d, t):
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    b = np.zeros_like(x)
    no = np.random.choice(t+1)
    o = np.random.choice(t, [no], replace=False)
    b[o] = 1.
    m = b.copy()
    if no == 0 or np.max(o) < t-1:
        rmin = 0 if no == 0 else np.max(o)+1
        w = np.random.choice(range(rmin, t))
        m[w] = 1.
    x = x.reshape([-1])
    b = b.reshape([-1])
    m = m.reshape([-1])

    return i, x, y, b, m

def _parse(i, x, y, d, t):
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    b = np.zeros_like(x)
    no = np.random.choice(t+1)
    o = np.random.choice(t, [no], replace=False)
    b[o] = 1.
    m = b.copy()
    if no < t:
        u = list(set(range(t)) - set(o))
        w = np.random.choice(u)
        m[w] = 1.
    x = x.reshape([-1])
    b = b.reshape([-1])
    m = m.reshape([-1])

    return i, x, y, b, m

class Dataset(object):
    def __init__(self, dfile, split, batch_size, time_steps):
        super().__init__()

        with open(dfile, 'rb') as f:
            data_dict = pickle.load(f)
        data, label = data_dict[split]
        self.size = data.shape[0]
        assert data.shape[1] == time_steps
        self.d = data.shape[1] * data.shape[2]
        self.num_batches = math.ceil(self.size / batch_size)

        ind = tf.range(self.size, dtype=tf.int64)
        dst = tf.data.Dataset.from_tensor_slices((ind, data, label))
        if split == 'train':
            dst = dst.shuffle(self.size)
        dst = dst.map(lambda i, x, y: tuple(
            tf.py_func(_parse, [i, x, y, self.d, time_steps], 
            [tf.int64, tf.float32, tf.float32, tf.float32, tf.float32])),
            num_parallel_calls=16)
        dst = dst.batch(batch_size)
        dst = dst.prefetch(1)
        dst_it = dst.make_initializable_iterator()
        self.i, self.x, self.y, self.b, self.m = dst_it.get_next()
        self.initializer = dst_it.initializer

    def initialize(self, sess):
        sess.run(self.initializer)

if __name__ == '__main__':
    dataset = Dataset('../data/time_series/digits.pkl', 'train', 32, 8)
    sess = tf.Session()
    dataset.initialize(sess)
    i,x,y,b,m = sess.run([dataset.i,dataset.x,dataset.y,dataset.b,dataset.m])
    print(i)
    print(x.shape)
    print(y.shape)
    print(b.shape)
    print(m.shape)
    print(x[0])
    print(b[0])
    print(m[0])