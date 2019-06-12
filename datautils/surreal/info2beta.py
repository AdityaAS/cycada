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

interval = 10000
index = int(sys.argv[1])

# videos = sorted(glob(join(root_dir, "*_segm.mat")))
def info2beta(infofile):
    info = sio.loadmat(infofile)
    betas = info['shape']
    betafile = infofile.split('/')[-1].split('.')[0][:-5] + '_beta.npy'
    betadir = '/'.join(infofile.split('/')[:-1])
    if betas.T.all():
        np.save(join(betadir, betafile), betas.T)
    else:
        print("Betas error!")
        import pdb; pdb.set_trace()
    
# inputs = []
# for i in range(0, 3):
#     root = join(root_dir, 'run{}'.format(i))
#     dirs = sorted(glob(join(root, '*')))
#     for path in dirs:
#         info = sorted(glob(join(path, '*_info.mat')))
#         inputs = inputs + info

inputs = pickle.load(open('/efs/data/users/aditya/surrealinfo.pkl', 'rb'))
inputs = inputs[interval*index:interval*(index+1)]
results = Parallel(n_jobs=16)(delayed(info2beta)(i) for i in inputs)