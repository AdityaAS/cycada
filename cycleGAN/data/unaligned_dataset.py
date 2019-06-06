import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import torch
import random
import cv2


class UnalignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'A')

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
        # print('(A, B) = (%d, %d)' % (index_A, index_B))
        #A_img = Image.open(A_path).convert('RGB')
        #B_img = Image.open(B_path).convert('RGB')
        try:
            A_img = Image.fromarray(cv2.imread(A_path))
            B_img = Image.fromarray(cv2.imread(B_path))
        except:
            print("Error in loading{}".format(B_img))
            import pdb; pdb.set_trace()
        A = self.transform(A_img)
        B = self.transform(B_img)
        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[2, ...] * 0.299 + A[1, ...] * 0.587 + A[0, ...] * 0.114 #BGR_to_gray
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



class UnalignedALabeledDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_Alabel = os.path.join(opt.dataroot, opt.phase + 'A_label')

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)
        self.A_label_paths = make_dataset(self.dir_Alabel)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.A_label_paths = sorted(self.A_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        A_label_paths = self.A_label_paths[index % self.A_size]

        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        # print('(A, B) = (%d, %d)' % (index_A, index_B))
        #A_img = Image.open(A_path).convert('RGB')
        #B_img = Image.open(B_path).convert('RGB')
        try:
            A_img = Image.fromarray(cv2.imread(A_path))
            B_img = Image.fromarray(cv2.imread(B_path))
            A_label = Image.fromarray(cv2.imread(A_label_paths))
        except:
            print("Error in loading{}".format(B_img))
        A = self.transform(A_img)
        B = self.transform(B_img)
        #A_label
        #A_label = torch.Tensor(A_label.transpose(2, 0, 1)).mean(dim=0) / 255
        A_label = self.transform(A_label).mean(dim=0) / 255
        #print(A_label.size())
        #import pdb; pdb.set_trace()

        #print(A_label.size())
        #import pdb; pdb.set_trace();
        #A_label = A_label.transpose(2, 0, 1).mean(dim=0) / 255

        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[2, ...] * 0.299 + A[1, ...] * 0.587 + A[0, ...] * 0.114 #BGR_to_gray
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)

        
        return {'A': A, 'B': B, 'A_label': A_label,
                'A_paths': A_path, 'B_paths': B_path, 'A_label_paths': A_label_paths}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'Unaligned_A_LabeledDataset'

