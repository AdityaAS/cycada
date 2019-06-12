import subprocess
import os
from os.path import join
from glob import glob
import pdb
import sys
from joblib import Parallel, delayed
import multiprocessing
import pickle

interval = 10000
index = int(sys.argv[1])

mode = 'test'
def convertVideoToImageSequence(video, image_directory):
    if not os.path.exists(image_directory):
        os.makedirs(image_directory)

    cmd = '</dev/null ffmpeg -i \"' + video + '\" -qscale:v 1 \"' + join(image_directory, video.split('/')[-1].split('.')[0] + '_%06d.png\"' )
    os.system(cmd)

# inputs = []

# videos = sorted(glob(join(root_dir, "*.mp4")))

# for i in range(0, 3):
#     root = join(root_dir, 'run{}'.format(i))
#     dirs = sorted(glob(join(root, '*')))
#     for path in dirs:
#         videos = sorted(glob(join(path, '*.mp4')))
#         for video in videos:
#             videono = video.split('/')[-1].split('.')[0]
#             image_directory = join('/'.join(video.split('/')[:-1]), 'images', videono)
#             inputs.append([video, image_directory])

# import pdb; pdb.set_trace()

inputs = pickle.load(open('/home/users/aditya/data/adult/SURREAL/surreal/download/SURREAL/data/cmu/allMP4_' + mode + '.pkl', 'rb'))
inputs = inputs[interval*index:interval*index + interval]
results = Parallel(n_jobs=80)(delayed(convertVideoToImageSequence)(inputt, mode + '_out') for inputt in inputs)

# for inputt in inputs:
#     convertVideoToImageSequence(inputt, 'train_out')