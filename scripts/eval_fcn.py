import os
from tqdm import *

import click
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.autograd import Variable
import sys
sys.path.append('.')

from cycada.data.data_loader import dataset_obj
from cycada.data.data_loader import get_fcn_dataset
from cycada.models.models import get_model
from cycada.tools.util import make_variable
from cycada.models.models import models
from cycada.util import to_tensor_raw

# dataloaders
from cycada.data.adda_datasets import AddaDataLoader
from cycada.data.cyclegta5 import CycleGTA5
from cycada.data.usps import USPS
from cycada.data.color2blk import Color2Blk
from cycada.data.blk import Blk

from cycada.metrics import IoU, recall, sklearnScores

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
    return indices

@click.command()
@click.option('--path', type=click.Path(exists=True))
@click.option('--dataset', default='blk')
@click.option('--datadir', default='/home/users/aditya/data/', type=click.Path(exists=True))
@click.option('--model', default='fcn8s', type=click.Choice(models.keys()))
@click.option('--num_cls', default=2)

def main(path, dataset, datadir, model, num_cls):
    net = get_model(model, num_cls=num_cls)
    net.load_state_dict(torch.load(path))
    net.eval()
    ds = get_fcn_dataset(dataset, os.path.join(datadir, dataset), split='test')
    classes = ds.num_cls
    collate_fn = torch.utils.data.dataloader.default_collate
    loader = torch.utils.data.DataLoader(ds,  num_workers=8, batch_size=16, shuffle=False, pin_memory=True, collate_fn=collate_fn)

    intersections = np.zeros(num_cls)
    unions = np.zeros(num_cls)
    
    ious = list()
    recalls = list()
    precisions = list()
    fscores = list()

    errs = []
    hist = np.zeros((num_cls, num_cls))

    if len(loader) == 0:
        print('Empty data loader')
        return
    iterations = tqdm(iter(loader))
    for i, (im, label) in enumerate(iterations):
        if i != 31 and i != 379:
           continue
        im = make_variable(im, requires_grad=False)
        label = make_variable(label, requires_grad=False)
        p = net(im)
        score = p

        iou = IoU(p, label)
        rc = recall(p, label)
        pr, rc, fs, _ = sklearnScores(p, label)

        if i == 31 or i == 379: #% int(len(iterations)/15) == 0:

            im = Image.fromarray(np.uint8(norm(im[0]).permute(1, 2, 0).cpu().data.numpy()*255))
            label = Image.fromarray(np.uint8(label[0].cpu().data.numpy()*255))
            score = Image.fromarray(np.uint8(mxAxis(score[0]).cpu().data.numpy()*255))
            
            im.save("img_" + str(i) + ".png")
            label.save("img_lbl_" + str(i) + ".png")
            score.save("img_sc_" + str(i) + ".png")
        

        ious.append(iou.item())

        recalls.append(rc)
        precisions.append(pr)
        fscores.append(fs)

        print("iou: ",np.mean(ious))
        print("recalls: ",np.mean(recalls))
        print("precision: ",np.mean(precisions))
        print("f1: ",np.mean(fscores))

        #print(','.join(num_cls))
    #print(fmt_array(iu))
    #print(np.nanmean(iu), fwIU, acc_overall, np.nanmean(acc_percls))
    print(np.argmax(ious), np.argmin(ious))  

if __name__ == '__main__':
    main()
