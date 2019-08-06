import os.path
import sys 

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import glob
from os.path import join
from cycada.data.data_loader import register_data_params, register_dataset_obj
from cycada.data.data_loader import DatasetParams
import cv2

ignore_label = 255
id2label = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
            3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
            7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
            14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
            18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
            28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}
palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
            220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
            0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
classes = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign',
        'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
        'bicycle']

def crp(img, target):

    mn = 768#min(img.shape[0], img.shape[1])
    y, x = np.random.randint(img.shape[0] - mn + 1), np.random.randint(img.shape[1] - mn + 1)

    crop_img = img[y:y+mn, x:x+mn]
    crop_tgt = target[y:y+mn, x:x+mn]
    return crop_img, crop_tgt

def remap_labels_to_train_ids(arr):
    out = ignore_label * np.ones(arr.shape, dtype=np.uint8)
    for id, label in id2label.items():
        out[arr == id] = int(label)
    return out


def remap_labels_to_train_ids(arr):
    out = ignore_label * np.ones(arr.shape, dtype=np.uint8)
    for id, label in id2label.items():
        out[arr == id] = int(label)
    return out

@register_data_params('gta2cityscapes')
class gta2cityscapesParams(DatasetParams):
    num_channels = 3
    image_size   = 768
    mean         = 0.5
    std          = 0.5
    num_cls      = 19
    target_transform = None


@register_dataset_obj('gta2cityscapes')
class Gta2cityscapes(data.Dataset):

    def __init__(self, name, root, params, num_cls=19, split='train', remap_labels=True, transform=None,
                 target_transform=None):
        self.root = root
        sys.path.append(root)
        self.split = split
        self.remap_labels = remap_labels
        self.transform = transform
        self.target_transform = target_transform
        self.num_cls = 19
        self.im_path = join(self.root, self.split, 'paired', 'images')
        self.seg_path = join(self.root, self.split, 'paired', 'segmasks')
        self.ids = self.collect_ids()
        
        self.id2label = id2label
        self.classes = classes

    def collect_ids(self):
        self.images = sorted(glob.glob(join(self.im_path, '*')))
        print(len(self.images))
        self.segmasks = sorted(glob.glob(join(self.seg_path, '*')))

    def img_path(self, id):
        return self.images[id]

    def label_path(self, id):
        return self.segmasks[id]

    def __getitem__(self, index):
 
        while (1):
            img = cv2.imread(self.img_path(index))
            target = cv2.imread(self.label_path(index), cv2.IMREAD_GRAYSCALE)

            img = cv2.resize(img, (1024, int((1024/img.shape[0])*img.shape[1]) ))
            target = cv2.resize(target, (img.shape[0],img.shape[1]) )

            img, target = crp(img, target)

            
            img, target = Image.fromarray(img.astype('uint8'), 'RGB'), Image.fromarray(target.astype('uint8'), 'L')

            if self.transform is not None:
                img = self.transform(img)
            
            if self.remap_labels:
                target = np.asarray(target)
                try:
                    target = remap_labels_to_train_ids(target)
                    break
                except:
                    print(target)
                    index = (index + np.random.randint(len(self.images))) % len(self.images) 


        target = Image.fromarray(np.uint8(target), 'L')
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    cs = Cityscapes('/x/CityScapes')
