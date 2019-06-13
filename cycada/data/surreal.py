import numpy as np
import scipy.io
import torch
import os
from torch.utils.data import Dataset
from glob import glob
from os.path import join, exists
import json
from cycada.data.data_loader import register_data_params, register_dataset_obj
from cycada.data.data_loader import DatasetParams
import cv2
from cycada.data.util import convert_image_by_pixformat_normalize


@register_data_params('surreal')
class SurrealParams(DatasetParams):
    num_channels = 3
    image_size = 256
    mean = 0.5
    num_cls = 2
    fraction = 1.0
    target_transform = None
    black = False

    def __init__(self, name):
        config = None
        print("PARAM: {}".format(os.getcwd()))
        with open(join("dataset_configs", name+".json"), 'r') as f:
            config = json.load(f)
        self.num_channels = config["num_channels"]
        self.image_size = config["image_size"]
        self.mean = config["mean"]
        self.num_cls = config["num_cls"]
        self.fraction = config["fraction"]
        self.target_transform = config["target_transform"]
        self.black = config["black"]

@register_dataset_obj('surreal')
class Surreal(Dataset):

    def __init__(self, name, root, params, num_cls=2, split='train', remap_labels=True, 
            transform=None, target_transform=None):
        self.root = root
        self.split = split
        self.remap_labels = remap_labels
        self.name = name
        self.transform = transform
        self.images = []
        self.segmasks = []
        self.target_transform = target_transform
        self.im_path = join(self.root, self.split, 'paired', 'images')
        self.seg_path = join(self.root, self.split, 'paired', 'segmasks')
        self.num_cls = num_cls
        self.size = (params.image_size, params.image_size)
        self.bw_flag = params.black
        self.seed = 255
        self.fraction = params.fraction if (self.split == 'train') else 1.0
        self.collect_ids()
        

    def collect_ids(self):
        np.random.seed(255)
        self.images = np.asarray(glob(join(self.im_path, '*')))
        self.segmasks = np.asarray(glob(join(self.seg_path, '*')))
        self.sampled_indices = np.arange(int(self.fraction * len(self.images)))
        #np.random.seed(255)
        self.images = self.images[self.sampled_indices]
        #np.random.seed(255)
        self.segmasks = self.segmasks[self.sampled_indices]

    def img_path(self, index):
        return self.images[index]

    def label_path(self, index):
        return self.segmasks[index]

    def __iter__(self):
        return self

    '''
    Input: Index of image to return
    Output:
        Image in the format NCHW - normalized
        Segmask in the format NHW (channels = 1 is understood) - not normalized because they are class labels
    '''
    def __getitem__(self, index):
        img_path = self.img_path(index)
        label_path = self.label_path(index)

        img = None
        if self.bw_flag:
            img = cv2.imread(img_path, 0)
            img_temp = np.expand_dims(img, axis = 2)
            img = np.concatenate((img_temp, img_temp, img_temp), axis=2)
        else:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        target = cv2.imread(label_path)
        img = cv2.resize(img, self.size)
        target = cv2.resize(target, self.size)
        # Convert to NCHW format and normalize to -1 to 1
        # WARNING: Original code did mean normalization, we did min max normalization. Change if necessary to old one.
        img = torch.Tensor(convert_image_by_pixformat_normalize(img))

        #WARNING: target must be made up of 0s and 1s only!
        target = torch.Tensor(target.transpose(2, 0, 1)).mean(dim=0) / 255

        return img, target

    def __len__(self):
        return len(self.images)
