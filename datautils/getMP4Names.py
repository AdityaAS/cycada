import glob
import pickle
import sys

mode = sys.argv[1] #'train'
folders = glob.glob("/scratch/users/aditya/adult/SURREAL/surreal/download/SURREAL/data/cmu/" + mode +"/run0/*")

files = []
wf = open("allMP4_" + mode + ".pkl", 'wb')
for f in folders:
    files += glob.glob(f + '/*.mp4') #03_01_c0005.mp4

import pdb;pdb.set_trace()
pickle.dump(files, wf)
