import os.path
import numpy as np
import scipy.io
import torch
import torch.utils.data as data
from glob import glob
from PIL import Image

from data.data_loader import register_data_params, register_dataset_obj
from data.data_loader import DatasetParams

@register_data_params('opendr')
class OpenDRParams(DatasetParams):
    num_channels = 3
    image_size = 480
    mean = 0.5
    num_cls = 2
    target_transform = None

@register_dataset_obj('opendr')
class OpenDR(data.Dataset):

    def __init__(self, root, num_cls=2, split='train', remap_labels=True, 
            transform=None, target_transform=None):
        self.root = root
        self.split = split
        self.remap_labels = remap_labels
        self.ids = self.collect_ids()
        self.transform = transform
        self.target_transform = target_transform
        self.im_path = os.path.join(self.root, self.split, 'paired', 'images')
        self.l_path = os.path.join(self.root, self.split, 'paired', 'segmasks')
        self.num_cls = num_cls

    
    def collect_ids(self):
        pathname = os.path.join(self.root, self.split, 'paired', 'images/*')
        list_file = glob(pathname)
        id_len = len(list_file)
        ids = ['{:05d}'.format(i) for i in range(id_len)]
        return ids

    def img_path(self, id):
        filename = id + ".jpg"
        return os.path.join(self.im_path, filename)

    def label_path(self, id):
        filename = id + ".png"
        return os.path.join(self.l_path, filename)

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