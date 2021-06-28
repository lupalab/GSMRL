import tensorflow as tf
import numpy as np
import math
import gzip
import pickle
import logging

def _parse_rnd(i, x, y, d):
    b = np.zeros([d], dtype=np.float32)
    no = np.random.choice(d)
    o = np.random.choice(d, [no], replace=False)
    b[o] = 1.
    m = b.copy()
    w = list(np.where(b == 0)[0])
    w = np.random.choice(w)
    m[w] = 1.

    return i, x, y, b, m

def _parse_bn(i, x, y, g):
    node_id = np.random.choice(len(g))
    b = g[node_id]
    m = b.copy()
    m[node_id] = 1.

    return i, x, y, b, m

def _parse(i, x, y, g, rate):
    if np.random.rand() < rate:
        return _parse_rnd(i, x, y, len(g))
    else:
        return _parse_bn(i, x, y, g)


class Dataset(object):
    def __init__(self, dfile, gfile, split, batch_size, rate):
        super().__init__()

        with open(dfile, 'rb') as f:
            data_dict = pickle.load(f)

        data, label = data_dict[split]
        self.size = data.shape[0]
        self.d = data.shape[1]
        self.num_batches = math.ceil(self.size / batch_size)

        if gfile:
            logging.info('using graph within gfile')
            with gzip.open(gfile, 'rb') as f:
                res = pickle.load(f)
            self.graph = res['graph'].astype(np.float32)
        elif 'graph' in data_dict:
            logging.info('using graph within the dataset')
            self.graph = data_dict['graph']
        else:
            logging.info('using naive bayes graph')
            self.graph = np.zeros([self.d, self.d], dtype=np.float32)

        ind = tf.range(self.size, dtype=tf.int64)
        dst = tf.data.Dataset.from_tensor_slices((ind, data, label))
        if split == 'train':
            dst = dst.shuffle(self.size)
        dst = dst.map(lambda i,x,y:tuple(
            tf.py_func(_parse, [i,x,y,self.graph,rate],
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
    sess = tf.Session()
    trainset = Dataset('../data/synthetic_bn/syn_bn.pkl', 'train', 1)
    print(trainset.size)
    print(trainset.d)
    print(trainset.num_batches)
    trainset.initialize(sess)
    x,y,b,m = sess.run([trainset.x, trainset.y, trainset.b, trainset.m])
    print(x.shape)
    print(y.shape)
    print(b.shape)
    print(m.shape)
    print(trainset.graph)
    print(b)
    print(m)