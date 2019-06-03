import os.path
import numpy as np
import scipy.io
import torch
import torch.utils.data as data
from glob import glob
from PIL import Image

from .data_loader import register_data_params, register_dataset_obj
from .data_loader import DatasetParams

from .cityscapes import remap_labels_to_train_ids

@register_data_params('blk')
class blkParams(DatasetParams):
    num_channels = 3
    image_size = 256
    mean = 0.5
    num_cls = 2
    target_transform = None

@register_dataset_obj('blk')
class blk(data.Dataset):

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
        # import pdb;pdb.set_trace()
        # for x in list_file:
            # try:
        # Y = [ int(x.split('_')[0]) for x in list_file ]
            # except:
            #     import pdb;pdb.set_trace()
        # list_file = [x for _,x in sorted(zip(Y,list_file))]
        id_len = len(list_file)
        ids = ['{:05d}'.format(i) for i in range(id_len)]
        return ids

    def img_path(self, id):
        filename = id + ".png"
        return os.path.join(self.im_path, filename)

    def label_path(self, id):
        filename = id + ".png"
        return os.path.join(self.l_path, filename)

    def __getitem__(self, index):
        id = self.ids[index]
        img_path = self.img_path(id)
        label_path = self.label_path(id)
        img = Image.open(img_path).convert('L')
        # img_f = np.zeros((img.shape[0], img.shape[1], 3))
        # img_f[:,:,0] = img
        # img_f[:,:,1] = img
        # img_f[:,:,2] = img
        # img = Image.fromarray(np.uint8(img_f))
        if self.transform is not None:
            img = self.transform(img)
        img = img.repeat(3,1,1)
        target = Image.open(label_path)
        # if self.remap_labels:
        #     target = np.asarray(target)/255
        #     target = Image.fromarray(target, 'L')
        if self.target_transform is not None:
            target = self.target_transform(target)/255.0
        return img, target

    def __len__(self):
        return len(self.ids)