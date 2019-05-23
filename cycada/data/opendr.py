import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset
from glob import glob
from PIL import Image
from os.path import join, exists

from cycada.data.data_loader import register_data_params, register_dataset_obj
from cycada.data.data_loader import DatasetParams

@register_data_params('singleview_opendr_solid')
class OpenDRParams(DatasetParams):
    num_channels = 3
    image_size = 480
    mean = 0.5
    num_cls = 2
    target_transform = None

@register_dataset_obj('singleview_opendr_solid')
class OpenDR(Dataset):

    def __init__(self, root, num_cls=2, split='train', remap_labels=True, 
            transform=None, target_transform=None):
        self.root = root
        self.split = split
        self.remap_labels = remap_labels
        self.name = "singleview_opendr_solid"
        self.transform = transform
        self.images = []
        self.segmasks = []
        self.target_transform = target_transform
        self.im_path = join(self.root, self.split, 'paired', 'images')
        self.seg_path = join(self.root, self.split, 'paired', 'segmasks')
        self.collect_ids()
        self.num_cls = num_cls

    def collect_ids(self):
        self.images = sorted(glob(join(self.im_path, '*')))
        self.segmasks = sorted(glob(join(self.seg_path, '*')))

    def img_path(self, index):
        return self.images[index]

    def label_path(self, index):
        return self.segmasks[index]

    def __iter__(self):
        return self

    def __getitem__(self, index):
        img_path = self.img_path(index)
        label_path = self.label_path(index)

        img = Image.open(img_path).convert('RGB')
                    
        target = Image.open(label_path)        
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.images)