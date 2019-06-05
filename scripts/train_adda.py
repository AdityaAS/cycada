import os
from os.path import join

import sys
sys.path.append('.')

from cycada.data.adda_datasets import AddaDataLoader
from cycada.data.cyclegta5 import CycleGTA5
from cycada.data import *
from cycada.data.usps import USPS
from cycada.data.mnist import MNIST
from cycada.data.svhn import SVHN
from cycada.data.cyclegan import Svhn2MNIST, Usps2Mnist, Mnist2Usps
from cycada.tools.train_task_net import train as train_source
from cycada.tools.test_task_net import load_and_test_net
from cycada.tools.train_adda_net import train_adda
import torch
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--s', dest='src', type=str, default='usps2mnist')
parser.add_argument('--t', dest='tgt', type=str, default='mnist')
parser.add_argument('--b', dest='batchSize', type=int, default=128)
parser.add_argument('--wd', dest='weight_decay', type=int, default=0)
parser.add_argument('--dd', dest='datadir', type=str, default='/home/ubuntu/anthro-efs/anthro-backup-virginia/data')
parser.add_argument('--mn', dest='modelName', type=str, default='model_A')
parser.add_argument('--m', dest='model', type=str, default='LeNet')
parser.add_argument('--nc', dest='numClasses', type=int, default=10)
parser.add_argument('--pe', dest='pixLevEpochs', type=int, default=100)
parser.add_argument('--fe', dest='featLevEpochs', type=int, default=200)
parser.add_argument('--plr', dest='pixLR', type=float, default=1e-4)
parser.add_argument('--flr', dest='featLR', type=float, default=1e-5)
parser.add_argument('--iter', dest='iter', type=int, default=None)
parser.add_argument('--ns', dest='numSave', type=int, default=50)

args = parser.parse_args()

# set random seed to 4325 
# to reproduce the exact numbers
np.random.seed(4325)

###################################
# Set to your preferred data path #
###################################
datadir = args.datadir
###################################

# Problem Params
src = args.src
tgt = args.tgt

base_src = src.split('2')[0]

model = args.model
num_cls = args.numClasses

# Output directory
outdir = 'results/{}_to_{}/{}'.format(src, tgt, args.modelName)

# Optimization Params
betas = (0.9, 0.999) # Adam default
weight_decay = args.weight_decay # Adam default
batch = args.batchSize

src_lr = args.pixLR
adda_lr = args.featLR
src_num_epoch = args.pixLevEpochs
adda_num_epoch = args.featLevEpochs

src_datadir = join(datadir, src)
args.src_net_file = join(outdir, '{}_net_{}'.format(model, src))          
args.adda_net_file = join(outdir, 'adda_{:s}_net_{:s}_{:s}'.format(model, src, tgt))

src_net_file = args.src_net_file + '_final.pth'
adda_net_file = args.adda_net_file + '_final.pth'

#######################
# 1. Train Source Net #
#######################

if os.path.exists(src_net_file):
    print('Skipping source net training, exists:', src_net_file)
else:
    
    train_source(src, src_datadir, model, num_cls, args, 
            outdir=outdir, num_epoch=src_num_epoch, batch=batch, 
            lr=src_lr, betas=betas, weight_decay=weight_decay)


#####################
# 2. Train Adda Net #
#####################

if os.path.exists(adda_net_file):
    print('Skipping adda training, exists:', adda_net_file)
else:
    train_adda(src, tgt, model, num_cls, args, num_epoch=adda_num_epoch, 
            batch=batch, datadir=datadir,
            outdir=outdir, src_weights=src_net_file, 
            lr=adda_lr, betas=betas, weight_decay=weight_decay)

##############################
# 3. Evalute source and adda #
##############################
tgt_datadir = join(datadir, tgt)
print()
if src == base_src:
    print('----------------')
    print('Test set:', src)
    print('----------------')
    print('Evaluating {} source model: {}'.format(src, src_net_file))
    load_and_test_net(src, src_datadir, src_net_file, model, num_cls, 
            dset='test', base_model=None)


print('----------------')
print('Test set:', tgt)
print('----------------')
print('Evaluating {} source model: {}'.format(src, src_net_file))
cm = load_and_test_net(tgt, tgt_datadir, src_net_file, model, num_cls, 
        dset='test', base_model=None)

print(cm)

print('Evaluating {}->{} adda model: {}'.format(src, tgt, adda_net_file))
cm = load_and_test_net(tgt, tgt_datadir, adda_net_file, 'AddaNet', num_cls, 
        dset='test', base_model=model)
print(cm)
