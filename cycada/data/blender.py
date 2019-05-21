import os.path
import numpy as np
import scipy.io
import torch
import torch.utils.data as data
from glob import glob
from PIL import Image

from .data_loader import register_data_params, register_dataset_obj
from .data_loader import DatasetParams

@register_dataset_params('blender')
class BlenderParams(DatasetParams):
	num_channels = 3
	image_size = 224
	mean = 0.5
	num_cls = 2
	target_transform = None

@register_dataset_obj('blender')
class Blender(data.Dataset):

    def __init__(self, root, num_cls=2, split='train', remap_labels=True, 
            transform=None, target_transform=None):
        self.root = root
        self.split = split
        self.remap_labels = remap_labels
        self.ids = self.collect_ids()
        self.transform = transform
        self.target_transform = target_transform
        self.img_path = os.path.join(self.root, self.split, 'paired', 'images')
        self.seg_path = os.path.join(self.root, self.split, 'paired', 'segmasks')
        self.num_cls = num_cls

    
    def collect_ids(self):
        pathname = os.path.join(self.img_path, '*')
        list_file = glob(pathname)
        id_len = len(list_file)
        ids = ['{:05d}'.format(i) for i in range(id_len)]
        return ids

    def img_path(self, id):
        filename = '{:05d}.png'.format(id)
        return os.path.join(self.img_path, filename)

    def label_path(self, id):
        filename = '{:05d}.png'.format(id)
        return os.path.join(self.seg_path, filename)

    def __getitem__(self, index):
        id = self.ids[index]
        img_path = self.img_path(id)
        label_path = self.label_path(id)
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        target = Image.open(label_path)
        if self.remap_labels:
            target = np.asarray(target)
            target = remap_labels_to_train_ids(target)
            target = Image.fromarray(target, 'L')
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.ids)
