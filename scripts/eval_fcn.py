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

def saveImg(im ,label, score, name):
    im = np.uint8(norm(im[0]).permute(1, 2, 0).cpu().data.numpy()*255)
    label = np.uint8(label[0].cpu().data.numpy()*255)
    label3 = np.zeros((label.shape[0], label.shape[1], 3))
    for i in range(3):
        label3[:,:,i] = label
    score = np.uint8(mxAxis(score[0]).cpu().data.numpy()*255)
    score3 = np.zeros((label.shape[0], label.shape[1], 3))
    for i in range(3):
        score3[:,:,i] = score
    
    out = Image.fromarray(np.uint8(np.concatenate([im, score3, label3], axis=1)))
    
    out.save(name)

@click.command()
@click.option('--path', type=click.Path(exists=True))
@click.option('--dataset', default='blk')
@click.option('--data_type', default='opendr')
@click.option('--datadir', default='/home/users/aditya/data/', type=click.Path(exists=True))
@click.option('--model', default='fcn8s', type=click.Choice(models.keys()))
@click.option('--num_cls', default=2)
@click.option('--mode', default='test')


def main(path, dataset, data_type, datadir, model, num_cls, mode):
    net = get_model(model, num_cls=num_cls)
    net.load_state_dict(torch.load(path))
    net.eval()
    ds = get_fcn_dataset(dataset, data_type, os.path.join(datadir, dataset), split=mode)
    classes = ds.num_cls
    collate_fn = torch.utils.data.dataloader.default_collate
    
    loader = torch.utils.data.DataLoader(ds,  num_workers=0, batch_size=16, shuffle=False, pin_memory=True, collate_fn=collate_fn)

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

    folderPath = '/'.join(path.split('/')[:-1]) + '/' + path.split('/')[-1].split('.')[0]

    os.makedirs(folderPath + '_worst_10', exist_ok=True)
    os.makedirs(folderPath + '_best_10', exist_ok=True)

    for i, (im, label) in enumerate(iterations):

        im = make_variable(im, requires_grad=False)
        label = make_variable(label, requires_grad=False)
        p = net(im)
        score = p

        iou = IoU(p, label)
        rc = recall(p, label)
        pr, rc, fs, _ = sklearnScores(p, label)

        ious.append(iou.item())

        recalls.append(rc)
        precisions.append(pr)
        fscores.append(fs)

        print("iou: ",np.mean(ious))
        print("recalls: ",np.mean(recalls))
        print("precision: ",np.mean(precisions))
        print("f1: ",np.mean(fscores))

    # Max, Min 10
    mx = list(np.argsort(ious)[-10:])
    mn = list(np.argsort(ious)[:10])

    iterations = tqdm(iter(loader))
    for i, (im, label) in enumerate(iterations):

        if i in mx:

            im = make_variable(im, requires_grad=False)
            label = make_variable(label, requires_grad=False)
            p = net(im)
            score = p

            saveImg(im, label, score, folderPath + '_best_10' + "/img_" + str(i) + ".png")

        if i in mn:

            im = make_variable(im, requires_grad=False)
            label = make_variable(label, requires_grad=False)
            p = net(im)
            score = p

            saveImg(im, label, score, folderPath + '_worst_10' + "/img_" + str(i) + ".png")

    print("="*100 + "\niou: ",np.mean(ious))
if __name__ == '__main__':
    main()
