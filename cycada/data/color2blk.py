import os.path
import numpy as np
import scipy.io
import torch
import torch.utils.data as data
from glob import glob
from PIL import Image

from .data_loader import register_data_params, register_dataset_obj
from .data_loader import DatasetParams


@register_data_params('color2blk')
class color2blkParams(DatasetParams):
    num_channels = 3
    image_size = 256
    mean = 0.5
    num_cls = 2
    target_transform = None

@register_dataset_obj('color2blk')
class color2blk(data.Dataset):

    def __init__(self, root, num_cls=2, split='train', remap_labels=True, 
            transform=None, target_transform=None):
        self.root = root
        self.split = split
        self.remap_labels = remap_labels
        self.ids, self.label_ids = self.collect_ids()
        self.transform = transform
        self.target_transform = target_transform
        self.im_path = os.path.join(self.root, self.split, 'paired', 'images')
        self.l_path = os.path.join(self.root, self.split, 'paired', 'segmasks')
        self.num_cls = num_cls

    
    def collect_ids(self):
        pathname = os.path.join(self.root, self.split, 'paired', 'images/*')
        list_file = glob(pathname)
        Y = [ int(x.split('_')[0].split('/')[-1]) for x in list_file ]
        list_file = [x for _,x in sorted(zip(Y,list_file))]
        id_len = len(list_file)
        list_file_lbl = ['{:05d}'.format(i) for i in range(id_len)]
        return list_file, list_file_lbl

    def img_path(self, id):
        filename = id# + ".jpg"
        return os.path.join(self.im_path, filename)

    def label_path(self, id):
        filename = id + ".png"
        return os.path.join(self.l_path, filename)

    def __getitem__(self, index):
        img_path = self.img_path(self.ids[index])
        label_path = self.label_path(self.label_ids[index])
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        target = Image.open(label_path)

        if self.target_transform is not None:
            target = self.target_transform(target)
        import pdb;pdb.set_trace()
        return img, target

    def __len__(self):
        return len(self.ids)