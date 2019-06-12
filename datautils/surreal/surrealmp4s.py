import pickle
from glob import glob
from os.path import join, exists

root_path = '/scratch/users/aditya/adult/SURREAL/surreal/download/SURREAL/data/cmu/'
splits = ['train', 'test', 'val']
runs = ['run0', 'run1', 'run2']

surrealmp4s = []
surrealinfos = []

for split in splits:
	splitdir = join(root_path, split)
	
	for run in runs:
		
		runpath = join(splitdir, run)
		subjects = sorted(glob(join(runpath, '*')))

		for subject in subjects:
			mp4s = glob(join(subject, '*.mp4'))
			surrealmp4s = surrealmp4s + mp4s
			
			infos = glob(join(subject, '*_info.mat'))
			surrealinfos = surrealinfos + infos


# COMMENTING TO PREVENT FUCKUPS
# pickle.dump(surrealmp4s, open(join(root_path, 'allmp4s.pkl'), 'wb'))
# pickle.dump(surrealinfos, open(join(root_path, 'allinfos.pkl'), 'wb'))

