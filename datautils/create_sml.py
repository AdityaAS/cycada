import glob
import os.path
from os import path

files = glob.glob("/scratch/users/aditya/surreal/train/paired/images/*.png")

import random 
from shutil import copyfile

files = random.sample(files, 40000)

for f in files:
	f = f.split("/")[-1]
	copyfile("images/" + f, "../../../surreal_sml/train/paired/images/" + f) # ../../../surreal_sml/train/paired/	
	copyfile("segmasks/" + f, "../../../surreal_sml/train/paired/segmasks/" + f)
