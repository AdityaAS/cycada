import os
import json
from os.path import join
from torchvision import datasets, transforms
from .data_loader import DatasetParams
from .data_loader import register_dataset_obj, register_data_params

@register_data_params('mnist')
class MNISTParams(DatasetParams):
    
    num_channels = 1
    image_size   = 28
    mean         = 0.1307
    std          = 0.3081
    num_cls      = 10
    transform = transforms.ToTensor()
    target_transform = transforms.ToTensor()

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
        self.transform = transforms.Compose([transforms.Resize((self.image_size, self.image_size)),transforms.ToTensor()])

@register_dataset_obj('mnist')
class MNIST(datasets.MNIST):
    def __init__(self, name, root, params, num_cls=2, split='train',
            transform=None, target_transform=None, download=True):
        train = True
        if not (split == 'train'):
            train = False
        super(MNIST, self).__init__(root, train=train, transform=params.transform,
                target_transform=params.target_transform, download=download)
