import os.path

from .cityscapes import remap_labels_to_train_ids
import numpy as np
import scipy.io
import torch
import torch.utils.data as data
from PIL import Image

from .data_loader import register_data_params, register_dataset_obj
from .data_loader import DatasetParams
from .cityscapes import id2label as LABEL2TRAIN
import cv2

def crp(img, target):

    mn = 768#min(img.shape[0], img.shape[1])
    y, x = np.random.randint(img.shape[0] - mn + 1), np.random.randint(img.shape[1] - mn + 1)

    crop_img = img[y:y+mn, x:x+mn]
    crop_tgt = target[y:y+mn, x:x+mn]
    return crop_img, crop_tgt


@register_data_params('gta5')
class GTA5Params(DatasetParams):
    num_channels = 3
    image_size = 768
    mean = 0.5
    std = 0.5
    num_cls = 19
    target_transform = None

@register_dataset_obj('gta5')
class GTA5(data.Dataset):

    def __init__(self, name, root, params, num_cls=19, split='train', remap_labels=True, transform=None,
                 target_transform=None):
        self.root = root
        self.split = split
        self.remap_labels = remap_labels
        self.ids = self.collect_ids()
        self.transform = transform
        self.target_transform = target_transform
        m = scipy.io.loadmat(os.path.join(self.root, 'mapping.mat'))
        full_classes = [x[0] for x in m['classes'][0]]
        self.classes = []
        for old_id, new_id in LABEL2TRAIN.items():
            if not new_id == 255 and old_id > 0:
                self.classes.append(full_classes[old_id])
        self.num_cls = 19

    
    def collect_ids(self):
        splits = scipy.io.loadmat(os.path.join(self.root, 'split.mat'))
        ids = splits['{}Ids'.format(self.split)].squeeze()

        return ids

    def img_path(self, id):
        filename = '{:05d}.png'.format(id)
        return os.path.join(self.root, 'images', filename)

    def label_path(self, id):
        filename = '{:05d}.png'.format(id)
        return os.path.join(self.root, 'segmasks', filename)

    def __getitem__(self, index):

        while (1):
            try:
                id = self.ids[index]
                img_path = self.img_path(id)
                label_path = self.label_path(id)

                img = cv2.imread(self.img_path(id))
                target = cv2.imread(self.label_path(id), cv2.IMREAD_GRAYSCALE)
                img, target = Image.fromarray(img.astype('uint8'), 'RGB'), Image.fromarray(target.astype('uint8'), 'L')
                break
            except:
                print("{} is missing ".format(index))
                index = (index + np.random.randint(0, len(self.ids)))%len(self.ids)



        if self.transform is not None:
            img = self.transform(img)

        if self.remap_labels:
            target = np.asarray(target)
            target = remap_labels_to_train_ids(target)
            target = Image.fromarray(target, 'L')
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.ids)

