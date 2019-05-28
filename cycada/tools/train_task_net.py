from __future__ import print_function

import os
from os.path import join
import numpy as np
import argparse

# Import from torch
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# Import from Cycada Package 
from ..models.models import get_model
from ..data.data_loader import load_data
from .test_task_net import test
from .util import make_variable, save_model, save_opt, load_opt, load_model

def train_epoch(loader, net, opt_net, epoch):
    log_interval = 100 # specifies how often to display
    net.train()

    for batch_idx, (data, target) in enumerate(loader):

        # make data variables
        data = make_variable(data, requires_grad=False)
        target = make_variable(target, requires_grad=False)
        #import pdb;pdb.set_trace()
        
        # zero out gradients
        opt_net.zero_grad()
       
        # forward pass
        score = net(data)
        loss = net.criterion(score, target)
        
        # backward pass
        loss.backward()
        
        # optimize classifier and representation
        opt_net.step()
       
        # Logging
        if batch_idx % log_interval == 0:
            print('[Train] Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(loader.dataset),
                100. * batch_idx / len(loader), loss.item()), end="")
            pred = score.data.max(1)[1]
            correct = pred.eq(target.data).cpu().sum()
            acc = correct.item() / len(pred) * 100.0
            print('  Acc: {:.2f}'.format(acc))


def train(data, datadir, model, num_cls, args, outdir='', 
        num_epoch=100, batch=128, 
        lr=1e-4, betas=(0.9, 0.999), weight_decay=0):
    """Train a classification net and evaluate on test set."""

    # Setup GPU Usage
    if torch.cuda.is_available():  
        kwargs = {'num_workers': batch, 'pin_memory': True}
    else:
        kwargs = {}

    ############
    # Load Net #
    ############
    net = get_model(model, num_cls=num_cls)
    print('-------Training net--------')
    print(net)

    ############################
    # Load train and test data # 
    ############################
    train_data = load_data(data, 'train', batch=batch, 
        rootdir=datadir, num_channels=net.num_channels, 
        image_size=net.image_size, download=True, kwargs=kwargs)
    
    test_data = load_data(data, 'test', batch=batch, 
        rootdir=datadir, num_channels=net.num_channels, 
        image_size=net.image_size, download=True, kwargs=kwargs)
   
    ###################
    # Setup Optimizer #
    ###################
    opt_net = optim.Adam(net.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    itr = 0
    if args.iter is not None:
       net = load_model(net, args.src_net_file + '_' + str(args.iter) + '.pth')
       opt_net = load_opt(opt_net, args.src_net_file + '_opt_net_' + str(args.iter) + '.pth')
       itr = args.iter
   
    #########
    # Train #
    #########
    print('Training {} model for {}'.format(model, data))
    for epoch in range(itr, num_epoch):
        train_epoch(train_data, net, opt_net, epoch)
    
        if epoch % args.numSave == 0:
            save_model(net, args.src_net_file + '_' + str(epoch) + '.pth')
            save_opt(opt_net, args.src_net_file + '_opt_net_' + str(epoch) + '.pth') 
    ########
    # Test #
    ########
    if test_data is not None:
        print('Evaluating {}-{} model on {} test set'.format(model, data, data))
        test(test_data, net)

    ############
    # Save net #
    ############
    os.makedirs(outdir, exist_ok=True)
    outfile = args.src_net_file + '_final.pth'#join(outdir, 'src_{:s}_net_{:s}_{:s}.pth'.format(model, src, tgt))

    print('Saving to', outfile)
    net.save(outfile)

    return net
