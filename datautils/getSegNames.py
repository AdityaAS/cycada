import glob
import pickle
import sys

mode = sys.argv[1] #'train'
folders = glob.glob("/scratch/users/aditya/adult/SURREAL/surreal/download/SURREAL/data/cmu/" + mode +"/run0/*")

files = []
wf = open("seg_" + mode + ".pkl", 'wb')
for f in folders:
    files += glob.glob(f + '/*_segm.mat')

pickle.dump(files, wf)
