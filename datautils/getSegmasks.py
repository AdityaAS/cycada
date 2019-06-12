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




mode = sys.argv[1]#'train'

def info2beta(infofile, video):
    info = sio.loadmat(infofile)

    betafile = infofile.split('/')[-1].split('.')[0][:-5]

    cmd = '</dev/null ffmpeg -i \"' + video + '\" -qscale:v 1 \"' + join(mode + '_img' , video.split('/')[-1].split('.')[0] + '_%d.png\"' )
    os.system(cmd)
    
    for k,v in info.items():
        if k == '__header__' or k == '__version__' or k == "__globals__":
            continue
        if v.max() == 0:
           continue



        v = np.minimum(v*1000, 255*np.ones(v.shape))
        im = np.zeros((v.shape[0], v.shape[1], 3))
        im[:,:,0] = v
        im[:,:,1] = v
        im[:,:,2] = v
        cv2.imwrite(mode + '_seg/' + video.split('/')[-1].split('.')[0] +  '_' + k.split('_')[-1] + '.png', im)
        print(k)

os.makedirs(mode + '_seg', exist_ok=True)
os.makedirs(mode + '_img', exist_ok=True)

inputs = pickle.load(open('/scratch/users/aditya/adult/SURREAL/surreal/download/SURREAL/data/cmu/seg_' + mode + '.pkl', 'rb'))
inputsRGB = pickle.load(open('/scratch/users/aditya/adult/SURREAL/surreal/download/SURREAL/data/cmu/allMP4_' + mode + '.pkl', 'rb'))


inputs = inputs[0:40000]
inputsRGB = inputsRGB[0:40000]

#info2beta(inputs[0], inputsRGB[0])
results = Parallel(n_jobs=80)(delayed(info2beta)(i, j) for i, j in zip(inputs, inputsRGB))
