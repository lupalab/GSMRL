import os
import sys
p = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(p)
import argparse
import logging
import tensorflow as tf
import numpy as np
from pprint import pformat, pprint

from cdfa_modules.gp_env import Dataset as GPDataset
from cdfa_modules.img_env import Dataset as ImgDataset
from models import get_model
from utils.hparams import HParams

parser = argparse.ArgumentParser()
parser.add_argument('--cfg_file', type=str)
args = parser.parse_args()
params = HParams(args.cfg_file)
pprint(params.dict)

################################################################
logging.basicConfig(filename=params.exp_dir + '/train.log',
                    filemode='w',
                    level=logging.INFO,
                    format='%(message)s')
logging.info(pformat(params.dict))

os.environ['CUDA_VISIBLE_DEVICES'] = params.gpu
np.random.seed(params.seed)
tf.set_random_seed(params.seed)

config = tf.ConfigProto()
config.log_device_placement = True
config.allow_soft_placement = True
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

################################################################
Dataset = {
    'gp': GPDataset,
    'img': ImgDataset
}[params.dataset]

# data
trainset = Dataset(params.x_size, params.y_size, params.batch_size, 'train')
validset = Dataset(params.x_size, params.y_size, params.batch_size, 'valid')
testset = Dataset(params.x_size, params.y_size, params.batch_size, 'test')

# model
model = get_model(sess, params)
initializer = tf.global_variables_initializer()
sess.run(initializer)
saver = tf.train.Saver(tf.global_variables())
writer = tf.summary.FileWriter(params.exp_dir + '/summaries')

total_params = 0
trainable_variables = tf.trainable_variables()
logging.info('=' * 20)
logging.info("Variables:")
logging.info(pformat(trainable_variables))
for k, v in enumerate(trainable_variables):
    num_params = np.prod(v.get_shape().as_list())
    total_params += num_params

logging.info("TOTAL TENSORS: %d TOTAL PARAMS: %f[M]" % (
    k + 1, total_params / 1e6))
logging.info('=' * 20)

###############################################################

def train():
    train_metrics = []
    train_acc = []
    for i in range(100):
        batch = trainset.sample()
        feed_dict={model.xc: batch['xc'],
                   model.yc: batch['yc'],
                   model.mc: batch['mc'],
                   model.xt: batch['xt'],
                   model.yt: batch['yt'],
                   model.mt: batch['mt'],
                   model.lab: batch['lab']}
        metric, acc, summ, step, _ = model.run(
            [model.metric, model.acc, model.summ_op, model.global_step, model.train_op], 
            feed_dict)
        if (params.summ_freq > 0) and (i % params.summ_freq == 0):
            writer.add_summary(summ, step)
        train_metrics.append(metric)
        train_acc.append(acc)
    train_metrics = np.concatenate(train_metrics, axis=0)
    train_acc = np.concatenate(train_acc, axis=0)
    logging.info(f'acc: {np.mean(train_acc)}')

    return np.mean(train_metrics)

def valid():
    valid_metrics = []
    for i in range(20):
        batch = validset.sample()
        feed_dict={model.xc: batch['xc'],
                   model.yc: batch['yc'],
                   model.mc: batch['mc'],
                   model.xt: batch['xt'],
                   model.yt: batch['yt'],
                   model.mt: batch['mt'],
                   model.lab: batch['lab']}
        metric = model.run(model.metric, feed_dict)
        valid_metrics.append(metric)
    valid_metrics = np.concatenate(valid_metrics, axis=0)

    return np.mean(valid_metrics)

def test():
    test_metrics = []
    for i in range(20):
        batch = testset.sample()
        feed_dict={model.xc: batch['xc'],
                   model.yc: batch['yc'],
                   model.mc: batch['mc'],
                   model.xt: batch['xt'],
                   model.yt: batch['yt'],
                   model.mt: batch['mt'],
                   model.lab: batch['lab']}
        metric = model.run(model.metric, feed_dict)
        test_metrics.append(metric)
    test_metrics = np.concatenate(test_metrics, axis=0)

    return np.mean(test_metrics)

###############################################################

logging.info('starting training')
best_train_metric = -np.inf
best_valid_metric = -np.inf
best_test_metric = -np.inf
for epoch in range(params.epochs):
    train_metric = train()
    valid_metric = valid()
    test_metric = test()
    # save
    if train_metric > best_train_metric:
        best_train_metric = train_metric
    if valid_metric > best_valid_metric:
        best_valid_metric = valid_metric
    if test_metric > best_test_metric:
        best_test_metric = test_metric
    save_path = os.path.join(params.exp_dir, 'weights/params.ckpt')
    saver.save(sess, save_path)

    logging.info("Epoch %d, train: %.4f/%.4f, valid: %.4f/%.4f test: %.4f/%.4f" %
                 (epoch, train_metric, best_train_metric, 
                 valid_metric, best_valid_metric,
                 test_metric, best_test_metric))
    sys.stdout.flush()