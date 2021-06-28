import os
import sys
p = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(p)
import argparse
import logging
import tensorflow as tf
import numpy as np
from pprint import pformat, pprint
import gzip
import pickle
import time
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import calibration

from utils.hparams import HParams

parser = argparse.ArgumentParser()
parser.add_argument('--cfg_file', type=str)
args = parser.parse_args()
params = HParams(args.cfg_file)
pprint(params.dict)

save_dir = os.path.join(params.exp_dir, 'cal_ts_uncertainty')
os.makedirs(save_dir, exist_ok=True)
logging.basicConfig(filename=save_dir + '/test.log',
                    filemode='w',
                    level=logging.INFO,
                    format='%(message)s')
logging.info(pformat(params.dict))

valid_file = f'{params.exp_dir}/valid_ts_uncertainty_cal/results.pgz'
test_file = f'{params.exp_dir}/test_ts_uncertainty_cal/results.pgz'

with gzip.open(valid_file, 'rb') as f:
    valid_dict = pickle.load(f)
valid_probs = valid_dict['probs']
valid_label = valid_dict['label']

with gzip.open(test_file, 'rb') as f:
    test_dict = pickle.load(f)
test_probs = test_dict['probs']
test_label = test_dict['label']

# calibration
# calibrator = calibration.PlattBinnerMarginalCalibrator(len(valid_label), num_bins=10)
# calibrator.train_calibration(valid_probs[:,-1], valid_label.astype(np.int32))

# test_probs_rep = test_probs.reshape([-1, test_probs.shape[-1]])
# calibrated_zs = calibrator.calibrate(test_probs_rep)
# test_probs_cal = calibrated_zs.reshape(test_probs.shape)

T = test_probs.shape[1]
test_probs_cal = []
for t in range(T):
    calibrator = calibration.PlattBinnerMarginalCalibrator(len(valid_label), num_bins=5)
    calibrator.train_calibration(valid_probs[:,t], valid_label.astype(np.int32))
    test_probs_cal.append(calibrator.calibrate(test_probs[:,t]))
test_probs_cal = np.stack(test_probs_cal, axis=1)


preds = np.argmax(test_probs_cal, axis=-1)
probs = np.max(test_probs_cal, axis=-1)
acc = (preds == np.expand_dims(test_label, axis=1)).astype(np.float32)
res = {
    'probs': probs,
    'preds': preds,
    'acc': acc
}
with gzip.open(f'{save_dir}/results.pgz', 'wb') as f:
    pickle.dump(res, f)

# plot
T = test_probs.shape[1]
x = range(1, T+1)
y = np.mean(acc, axis=0)
p = np.mean(probs, axis=0)
fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(1,2,1)
ax.plot(x, y, marker='x')
ax.set_xticks(x)
ax.set_xlabel('time step')
ax.set_ylabel('accuracy')

ax = fig.add_subplot(1,2,2)
ax.plot(x, p, marker='x')
ax.set_xticks(x)
ax.set_xlabel('time step')
ax.set_ylabel('probability')

thresh = 0.9
length = []
for prob in probs:
    inds = np.where(prob >= thresh)[0]
    ind = np.min(inds) if len(inds) > 0 else T-1
    length.append(ind+1)
length = np.mean(length)

plt.title(f'threshold:0.9 => time:{length}')

plt.savefig(f'{save_dir}/acc.png')
plt.close('all')
