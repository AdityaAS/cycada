import subprocess
import os
from os.path import join
from glob import glob
import pdb
import sys
from joblib import Parallel, delayed
import multiprocessing
import pickle

root_dir = '/scratch/users/aditya/adult/SURREAL/surreal/download/SURREAL/data/cmu'

interval = 10000
index = int(sys.argv[1])

def convertVideoToImageSequence(videofile):
    videono = videofile.split('/')[-1].split('.')[0]
    image_directory = join('/'.join(videofile.split('/')[:-1]), 'images', videono)
    
    if not os.path.exists(image_directory):
        os.makedirs(image_directory)

    cmd = '</dev/null ffmpeg -i \"' + videofile + '\" -qscale:v 1 \"' + join(image_directory, '%06d.png\"' )
    os.system(cmd)

inputs = pickle.load(open(join(root_dir, 'allmp4s.pkl'), 'rb'))

# convertVideoToImageSequence(allmp4s[0])

inputs = inputs[interval*index:interval*index + interval]

# convertVideoToImageSequence(inputs[0])

# COMMENTING TO PREVENT FUCKUPS
# results = Parallel(n_jobs=16)(delayed(convertVideoToImageSequence)(inputt) \
# 								for inputt in inputs)


