import logging
import logging.config
import os.path
from collections import OrderedDict

import numpy as np
import torch
import yaml
from torch.nn.parameter import Parameter
from tqdm import tqdm

class TqdmHandler(logging.StreamHandler):

    def __init__(self):
        logging.StreamHandler.__init__(self)

    def emit(self, record):
        msg = self.format(record)
        tqdm.write(msg)


def config_logging(logfile=None):
    path = os.path.join(os.path.dirname(__file__), 'logging.yml')
    with open(path, 'r') as f:
        config = yaml.load(f.read())
    if logfile is None:
        del config['handlers']['file_handler']
        del config['root']['handlers'][-1]
    else:
        config['handlers']['file_handler']['filename'] = logfile
   # logging.config.dictConfig(config)


# How is this different from torchvision.transforms.ToTensor()
def to_tensor_raw(im):
    return torch.from_numpy(np.array(im, np.int64, copy=False))


def safe_load_state_dict(net, state_dict):
    """Copies parameters and buffers from :attr:`state_dict` into
    this module and its descendants. Any params in :attr:`state_dict`
    that do not match the keys returned by :attr:`net`'s :func:`state_dict()`
    method or have differing sizes are skipped.

    Arguments:
        state_dict (dict): A dict containing parameters and
            persistent buffers.
    """
    own_state = net.state_dict()
    skipped = []
    for name, param in state_dict.items():
        if name not in own_state:
            skipped.append(name)
            continue
        if isinstance(param, Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        if own_state[name].size() != param.size():
            skipped.append(name)
            continue
        own_state[name].copy_(param)

    if skipped:
        logging.info('Skipped loading some parameters: {}'.format(skipped))

def step_lr(optimizer, mult):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        param_group['lr'] = lr * mult

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n,n)

def check_label(label, num_cls):
    "Check that no labels are out of range"
    print(label.size())
    label_classes = np.unique(label.numpy().flatten())
    print(label_classes)
    label_classes = label_classes[label_classes < 255]
    if len(label_classes) == 0:
        print('All ignore labels')
        return False
    class_too_large = label_classes.max() > num_cls
    if class_too_large or label_classes.min() < 0:
        print('Labels out of bound')
        print(label_classes)
        return False
    return True

def roundrobin_infinite(*loaders):
    if not loaders:
        return
    iters = [iter(loader) for loader in loaders]
    while True:
        for i in range(len(iters)):
            it = iters[i]
            try:
                yield next(it)
            except StopIteration:
                iters[i] = iter(loaders[i])
                yield next(iters[i])

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def get_pred(pred):
    return torch.max(pred, dim=1)[1].long()

def preprocess_viz(image, pred, labels):
    p_maps = get_pred(pred)
    p_maps = p_maps.cpu().numpy()
    labels = labels.cpu().numpy()
    images = image.cpu().numpy()
    im1 = rgb2gray(images[0].T)
    im2 = rgb2gray(images[1].T)
    final = np.stack((im1,p_maps[0],labels[0],im2,p_maps[1],labels[1]))
    final = np.expand_dims(final, axis=1)
    return final



