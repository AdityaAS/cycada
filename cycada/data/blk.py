import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset
from glob import glob
from os.path import join, exists

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

@register_data_params('blk')
# @register_data_params('singleview_opendr_color_100k_copy')
class BlkParams(DatasetParams):
    num_channels = 3
    image_size = 256
    mean = 0.5
    num_cls = 2
    target_transform = None

# @register_dataset_obj('singleview_opendr_color_100k_copy')
@register_dataset_obj('blk')
class Blk(Dataset):

    def __init__(self, root, num_cls=2, split='train', remap_labels=True, 
            transform=None, target_transform=None, size=(256, 256), blk = True):
        self.blk = blk
        self.root = root
        self.split = split
        self.remap_labels = remap_labels
        self.name = "blk"
        self.transform = transform
        self.images = []
        self.segmasks = []
        self.target_transform = target_transform
        self.im_path = join(self.root, self.split, 'paired', 'images')
        self.seg_path = join(self.root, self.split, 'paired', 'segmasks')
        self.collect_ids()
        self.num_cls = num_cls
        self.size = size

    def collect_ids(self):
        self.images = sorted(glob(join(self.im_path, '*')))
        print(len(self.images))
        self.segmasks = sorted(glob(join(self.seg_path, '*')))

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
        #for loading bw images
        if self.blk:
           img = cv2.imread(img_path, 0)
           img_temp = np.expand_dims(img, axis = 2)
           img = np.concatenate((img_temp, img_temp, img_temp), axis=2)
        else:
           img = cv2.imread(img_path)
        
        target = cv2.imread(label_path)
        # import pdb;pdb.set_trace()

        if img.shape[0] != img.shape[1]:
            img = Sqr(img)
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
