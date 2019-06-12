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
import cv2

root_dir = '/scratch/users/aditya/adult/SURREAL/surreal/download/SURREAL/data/cmu'

inputs = []
interval = 10000
index = int(sys.argv[1])

def bodypart2fgbg(bps):
    segm = sio.loadmat(bps)

    subfolder = '/'.join(bps.split('/')[:-1])
    subject = bps.split('/')[-1].split('.')[0][:-5]

    segmaskfolder = join(subfolder, 'segmasks', subject)

    if not(os.path.exists(segmaskfolder)):
        os.makedirs(segmaskfolder)

    fgbg = []
    count = len(list(segm.keys())) - 3

    for j in range(count):
        seg = segm['segm' + '_' + str(j+1)]
        seg = seg > 0        
        name = '%06d.png'%(j+1)
        cv2.imwrite(join(segmaskfolder, name), seg.astype(float))
        
    # fgbg = np.array(fgbg)
    # fgbgfile = bps.split('/')[-1].split('.')[0][:-5] + '_fgbg.npy'
    # fgbgdir = '/'.join(bps.split('/')[:-1])
    # np.save(join(fgbgdir, fgbgfile), fgbg)

inputs = pickle.load(open('/scratch/users/aditya/adult/SURREAL/surreal/download/SURREAL/data/cmu/allinfos.pkl', 'rb'))
inputs2 = [input.replace('_info', '_segm') for input in inputs]

inputs2 = inputs2[interval*index:interval*(index+1)]
print(len(inputs2))

# COMMENTING TO PREVENT FUCKUPS
# results = Parallel(n_jobs=16)(delayed(bodypart2fgbg)(i) for i in inputs2)