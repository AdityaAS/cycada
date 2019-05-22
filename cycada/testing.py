from data.blender import Blender
from data.opendr import OpenDR

data1 = Blender('/home/ubuntu/anthro-efs/anthro-backup-virginia/data/HMR_baby/datasets/singleview_blender_100k_visibility', remap_labels=False)
print(len(data1))
print(data1[0][0])
print(data1[0][1])

data2 = OpenDR('/home/ubuntu/anthro-efs/anthro-backup-virginia/data/HMR_baby/datasets/singleview_opendr_color_100k_copy', remap_labels=False)
print(len(data2))
print(data2[0][0])
print(data2[0][1])