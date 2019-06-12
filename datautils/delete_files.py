import glob
import os.path
from os import path

files = glob.glob("/scratch/users/aditya/surreal/train/paired/images/*.png")
for f in files:
	f = f.split('/')
	f = f[-1]

	if path.exists("segmasks/" + f):
		continue
	else:
		print(f)
		os.remove("images/" + f)


files = glob.glob("/scratch/users/aditya/surreal/train/paired/segmasks/*.png")
for f in files:
        f = f.split('/')
        f = f[-1]
       
        if path.exists("images/" + f):
                continue
        else:
                print(f)
                os.remove("segmasks/" + f)
