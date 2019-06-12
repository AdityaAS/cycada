from os.path import join
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import cv2
import numpy as np

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

class UnalignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        # self.root = opt.dataroot
        self.dir_A = join(opt.dataroot_A, opt.phase, "paired/images")
        self.dir_B = join(opt.dataroot_B, opt.phase, "paired/images")

        # self.dir_A = join(opt.dataroot, opt.phase + 'A')
        # self.dir_B = join(opt.dataroot, opt.phase + 'B')


        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        # A_img = Image.open(A_path).convert('RGB')
        # B_img = Image.open(B_path).convert('RGB')

        A_img = cv2.imread(A_path)
        B_img = cv2.imread(B_path)

        if A_img.shape[0] != A_img.shape[1]:
            A_img = Sqr(A_img)
            B_img = Sqr(B_img)

        A_img = Image.fromarray(np.uint8(A_img))
        B_img = Image.fromarray(np.uint8(B_img))


        A = self.transform(A_img)
        B = self.transform(B_img)

        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)
        
        return {'A': A, 'B': B,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDataset'
