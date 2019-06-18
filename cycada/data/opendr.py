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

def Sqr(img):

    mx = max(img.shape[0], img.shape[1])
    img2 = np.zeros((mx, mx, 3))
    try:
        img2[int((mx - img.shape[0])/2):mx-int((mx - img.shape[0])/2), int((mx - img.shape[1])/2):mx - int((mx - img.shape[1])/2) ,:] = img
    except:
        try:
            img2[int((mx - img.shape[0])/2):mx-int((mx - img.shape[0])/2), int((mx - img.shape[1])/2):-1 -int((mx - img.shape[1])/2) ,:] = img
        except:
            img2[int((mx - img.shape[0])/2):mx - 1-int((mx - img.shape[0])/2), int((mx - img.shape[1])/2):mx - int((mx - img.shape[1])/2) ,:] = img

    return img2

def Crp(img):

    mn = min(img.shape[0], img.shape[1])
    y, x = np.random.randint(img.shape[0] - mn + 1), np.random.randint(img.shape[1] - mn + 1)

    crop_img = img[y:y+mn, x:x+mn]
    return crop_img


@register_data_params('opendr')
class OpenDRParams(DatasetParams):
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

@register_dataset_obj('opendr')
class OpenDR(Dataset):

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
        self.images = sorted(glob(join(self.im_path, '*')))
        self.segmasks = sorted(glob(join(self.seg_path, '*')))
        # np.random.seed(self.seed)
        # self.images = sorted(np.random.choice(self.images, int(self.fraction * len(self.images))))
        # np.random.seed(self.seed)
        # self.segmasks = sorted(np.random.choice(self.segmasks, int(self.fraction * len(self.segmasks))))

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

        target = cv2.imread(label_path)

        if img.shape[0] != img.shape[1]:
            img = Sqr(img)
        if target.shape[0] != target.shape[1]:
            target = Sqr(target)

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
