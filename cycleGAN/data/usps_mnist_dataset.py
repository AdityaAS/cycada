import random
import os.path
import torch
import gzip
import torchvision.transforms as transforms
from torchvision.datasets.mnist import MNIST
from data.base_dataset import BaseDataset
from urllib.parse import urljoin
import scipy.io
import numpy as np
import sys
sys.path.append('../../')
sys.path.append('../')
from cycada.data import util
from PIL import Image
from PIL.ImageOps import invert


class UspsMnistDataset(BaseDataset):
    def name(self):
        return 'UspsMnistDataset'

    def initialize(self, opt):
        self.opt = opt
        #print(opt)
        self.usps_root = opt.dataroot_A
        self.mnist_root = opt.dataroot_B
        self.usps_transforms = transforms.Compose([transforms.Pad(6),transforms.ToTensor()])
        self.mnist_transforms = transforms.ToTensor()
        self.target_transforms = torch.from_numpy

        self.mnist = MNIST(os.path.join(opt.dataroot, 'mnist'),
                           train=opt.isTrain, download=True, transform=self.mnist_transforms)
        self.mnist_indices = list(range(len(self.mnist)))
        
        
        self.usps_base_url = 'http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/'

        self.usps_data_files = {
            'train': 'zip.train.gz',
            'test': 'zip.test.gz'
        }
                               
        datapath = os.path.join(self.usps_root, self.usps_data_files['train'])
        print(datapath)
        self.download()
        #datapath = os.path.join(self.root, self.data_files['test'])

        self.usps_images, self.usps_targets = self.read_data(datapath)

        


        #self.shuffle_indices()
    def get_path(self, filename):
        return os.path.join(self.usps_root, filename)

    def read_data(self, path):
        images = []
        targets = []
        with gzip.GzipFile(path, 'r') as f:
            for line in f:
                split = line.strip().split()
                label = np.array(int(float(split[0])))
                pixels = np.array([(float(x) + 1) / 2 for x in split[1:]]) * 255
                num_pix = 16
                pixels = pixels.reshape(num_pix, num_pix).astype('uint8')
                img = Image.fromarray(pixels, mode='L')
                images.append(img)
                targets.append(label)
        return images, targets

    def download(self):
        data_dir = self.usps_root
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        for filename in self.usps_data_files.values():
            path = self.get_path(filename)
            if not os.path.exists(path):
                url = urljoin(self.usps_base_url, filename)
                util.maybe_download(url, path)

    def shuffle_indices(self):
        self.mnist_indices = list(range(len(self.mnist)))
        self.svhn_indices = list(range(self.svhn.shape[0]))
        print('num mnist', len(self.mnist_indices), 'num svhn', len(self.svhn_indices))
        if not self.opt.serial_batches:
            random.shuffle(self.mnist_indices)
            random.shuffle(self.svhn_indices)

    def __getitem__(self, index):
        #import pdb;pdb.set_trace()

        A_img = self.usps_images[index]
        A_label = self.usps_targets[index]
        A_img = self.usps_transforms(A_img)
        A_label = self.target_transforms(A_label)

        B_img, B_label = self.mnist[self.mnist_indices[index % len(self.mnist)]]
        B_label = torch.from_numpy(np.asarray(B_label))


        A_path = '%01d_%05d.png' % (A_label, index)
        B_path = '%01d_%05d.png' % (B_label, index)
   
        #A_img, B_img = B_img, A_img
        #A_path, B_path = B_path, A_path
        #A_label, B_label = B_label, A_label

        item = {}
        item.update({'A': A_img,
                     'A_paths': A_path,
                     'A_label': A_label
                 })
        
        item.update({'B': B_img,
                     'B_paths': B_path,
                     'B_label': B_label
                 })
        return item
        
    def __len__(self):
        #if self.opt.which_direction == 'AtoB':
        #    return len(self.mnist)
        #else:            
        #    return self.svhn.shape[0]

        return min(len(self.usps_images),len(self.mnist)) #min(len(self.mnist), self.svhn.shape[0])
        
