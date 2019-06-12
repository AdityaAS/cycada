import subprocess
import os
from os.path import join
from glob import glob
import pdb
import scipy.io as sio
from joblib import Parallel, delayed
import multiprocessing
import numpy as np
import pickle
import sys
root_dir = '/efs/data/HMR_ADULT/SURREAL/surreal/download/SURREAL/data/cmu/train/'

inputs = []
interval = 10000
index = int(sys.argv[1])

# videos = sorted(glob(join(root_dir, "*_segm.mat")))
def bodypart2fgbg(bps):
    segm = sio.loadmat(bps)
    fgbg = []
    count = len(list(segm.keys())) - 3
    for j in range(count):
        seg = segm['segm' + '_' + str(j+1)]
        seg = seg > 0
        fgbg.append(seg)

    fgbg = np.array(fgbg)
    fgbgfile = bps.split('/')[-1].split('.')[0][:-5] + '_fgbg.npy'
    fgbgdir = '/'.join(bps.split('/')[:-1])

    np.save(join(fgbgdir, fgbgfile), fgbg)

# inputs = []
# for i in range(0, 3):
#     root = join(root_dir, 'run{}'.format(i))
#     dirs = sorted(glob(join(root, '*')))
#     for path in dirs:
#         bp_segmentation = sorted(glob(join(path, '*_segm.mat')))
#         inputs = inputs + bp_segmentation

# import pdb; pdb.set_trace()

inputs = pickle.load(open('/efs/data/users/aditya/surrealsegm.pkl', 'rb'))

# bodypart2fgbg(inputs[0])
inputs = inputs[interval*index:interval*(index+1)]
# COMMENTING TO PREVENT FUCKUPS
results = Parallel(n_jobs=16)(delayed(bodypart2fgbg)(i) for i in inputs)