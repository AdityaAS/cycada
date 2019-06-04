import os
from tqdm import *

import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../cycada')
import click
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.autograd import Variable
from collections import deque
from cycada.data.data_loader import dataset_obj
from cycada.data.data_loader import get_fcn_dataset
from cycada.models.models import get_model
from cycada.tools.util import make_variable
from cycada.models.models import models
from cycada.util import to_tensor_raw
from metrics import IoU, recall

from cycada.data.color2blk import color2blk
from cycada.data.blk import blk

def fmt_array(arr, fmt=','):
    strs = ['{:.3f}'.format(x) for x in arr]
    return fmt.join(strs)

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def result_stats(hist):
    acc_overall = np.diag(hist).sum() / hist.sum() * 100
    acc_percls = np.diag(hist) / (hist.sum(1) + 1e-8) * 100
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-8) * 100
    freq = hist.sum(1) / hist.sum()
    fwIU = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc_overall, acc_percls, iu, fwIU

def norm(tensor):
    r = tensor.max() - tensor.min()
    tensor = (tensor - tensor.min())/r
    return tensor

def mxAxis(tensor):
    _, indices = torch.max(tensor, 0)
    import pdb;pdb.set_trace()
    return indices

@click.command()
@click.option('--path', type=click.Path(exists=True))
@click.option('--dataset', default='cityscapes')
@click.option('--datadir', default='',

        type=click.Path(exists=True))
@click.option('--model', default='fcn8s', type=click.Choice(models.keys()))
@click.option('--gpu', default='0')
@click.option('--num_cls', default=19)
def main(path, dataset, datadir, model, gpu, num_cls):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    net = get_model(model, num_cls=num_cls)
    net.load_state_dict(torch.load(path))
    net.eval()
    ds = get_fcn_dataset(dataset, os.path.join(datadir, dataset), split='test')
    classes = ds.num_cls
    collate_fn = torch.utils.data.dataloader.default_collate
    loader = torch.utils.data.DataLoader(ds,  num_workers=8, batch_size=16, shuffle=False, pin_memory=True, collate_fn=collate_fn)

    intersections = np.zeros(num_cls)
    unions = np.zeros(num_cls)
    
    ious = deque(maxlen=625)
    recalls = deque(maxlen=625)

    errs = []
    hist = np.zeros((num_cls, num_cls))

    if len(loader) == 0:
        print('Empty data loader')
        return
    iterations = tqdm(iter(loader))
    for im, label in iterations:
        im = make_variable(im, requires_grad=False)
        label = make_variable(label, requires_grad=False)
        p = net(im)
        score = p

        #import pdb; pdb.set_trace()
        #import pdb;pdb.set_trace()
        #val = im[0].permute(1, 2, 0).cpu().numpy()
        #im = Image.fromarray(val)
        #label = Image.fromarray(label[0].cpu().numpy())
        #score = Image.fromarray(mxAxis(score[0]).cpu().numpy())

        #im.save("img.png")
        #label.save("img_lbl.png")
        #score.save("img_sc.png")
        
        #import pdb;pdb.set_trace();
        iou = IoU(p, label)
        rc = recall(p, label)

        ious.append(iou.item())
        recalls.append(rc.item())
        #iterations.set_postfix({'miou : {}, mrecall : {}'.format(np.mean(ious), np.mean(recalls))})
        print("iou: ",np.mean(ious))
        print("recalls: ",np.mean(recalls))
        #print(','.join(num_cls))
    #print(fmt_array(iu))
    #print(np.nanmean(iu), fwIU, acc_overall, np.nanmean(acc_percls))
  

if __name__ == '__main__':
    main()
