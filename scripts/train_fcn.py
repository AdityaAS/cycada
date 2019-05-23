import sys
sys.path.append('./')
import logging
import os.path
from collections import deque

import click
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tensorboardX import SummaryWriter
from PIL import Image
from torch.autograd import Variable
print(os.getcwd())
from cycada.data.data_loader import get_fcn_dataset as get_dataset
from cycada.models import get_model
from cycada.models.models import models
from cycada.transforms import augment_collate
from cycada.util import config_logging
from cycada.util import to_tensor_raw
from cycada.util import roundrobin_infinite
from cycada.tools.util import make_variable
from cycada.loss_fns import supervised_loss
from cycada.metrics import IoU, recall

@click.command()
@click.argument('output')
@click.option('--phase', default='train')
@click.option('--dataset', required=True, multiple=True)
@click.option('--datadir', default="", type=click.Path(exists=True))
@click.option('--batch_size', '-b', default=1)
@click.option('--lr', '-l', default=0.001)
@click.option('--step', type=int)
@click.option('--iterations', '-i', default=100000)
@click.option('--momentum', '-m', default=0.9)
@click.option('--snapshot', '-s', default=5000)
@click.option('--downscale', type=int)
@click.option('--augmentation/--no-augmentation', default=False)
@click.option('--fyu/--torch', default=False)
@click.option('--crop_size', default=120)
@click.option('--weights', type=click.Path(exists=True))
@click.option('--model', default='fcn8s', type=click.Choice(models.keys()))
@click.option('--num_cls', default=2, type=int)
@click.option('--gpu', default='0')

def main(output, phase, dataset, datadir, batch_size, lr, step, iterations, 
        momentum, snapshot, downscale, augmentation, fyu, crop_size, 
        weights, model, gpu, num_cls):
    if weights is not None:
        raise RuntimeError("weights don't work because eric is bad at coding")
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    config_logging()
    
    logdir = 'runs/{:s}/{:s}'.format(model, '-'.join(dataset))
    writer = SummaryWriter(log_dir=logdir)
    net = get_model(model, num_cls=num_cls)
    #print("Model Parameter{}".format(net.parameters))

    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    #print("NUM params {}".format(params))
    net.cuda()
    transform = []
    target_transform = []
    if downscale is not None:
        transform.append(torchvision.transforms.Resize(480 // downscale))
        target_transform.append(
            torchvision.transforms.Resize(480 // downscale,
                                         interpolation=Image.NEAREST))
    transform.extend([
        torchvision.transforms.Resize(480),
        net.transform
        ])
    target_transform.extend([
        torchvision.transforms.Resize(480, interpolation=Image.NEAREST),
        to_tensor_raw
        ])
    transform = torchvision.transforms.Compose(transform)
    target_transform = torchvision.transforms.Compose(target_transform)
    dataset = dataset[0]

    datasets_train = get_dataset(dataset, os.path.join(datadir, dataset), split='train',transform=transform,
                        target_transform=target_transform)
                        #for name in dataset]

    datasets_val = get_dataset(dataset, os.path.join(datadir, dataset), split='val',transform=transform,
                        target_transform=target_transform)
                        #for name in dataset]

    datasets_test = get_dataset(dataset, os.path.join(datadir, dataset), split='test',transform=transform,
                        target_transform=target_transform)
                        #

    if weights is not None:
        weights = np.loadtxt(weights)
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum,
                          weight_decay=0.0005)

    if augmentation:
        collate_fn = lambda batch: augment_collate(batch, crop=crop_size, flip=True)
    else:
        collate_fn = torch.utils.data.dataloader.default_collate

    train_loader = torch.utils.data.DataLoader(datasets_train, batch_size=batch_size,
                                            shuffle=True, num_workers=2,
                                            collate_fn=collate_fn,
                                            pin_memory=True)

    val_loader = torch.utils.data.DataLoader(datasets_val, batch_size=batch_size,
                                            shuffle=True, num_workers=2,
                                            collate_fn=collate_fn,
                                            pin_memory=True)

    test_loader = torch.utils.data.DataLoader(datasets_test, batch_size=batch_size,
                                            shuffle=True, num_workers=2,
                                            collate_fn=collate_fn,
                                            pin_memory=True)
    iteration = 0
    losses = deque(maxlen=10)

    max_epochs = math.ceil(iterations/batch_size)
    for epochs in range(max_epochs):
        if phase == 'train':
            net.train()
            for im, label in train_loader:
                # Clear out gradients
                opt.zero_grad()
                # load data/label
                im = make_variable(im, requires_grad=False)
                label = make_variable(label, requires_grad=False)
        
                # forward pass and compute loss
                preds = net(im)
                loss = supervised_loss(preds, label)

                IoU(preds, label)
                recall(preds, label)
                print(epochs, loss)
        
                # backward pass
                loss.backward()
                losses.append(loss.item())
        
                # step gradients
                opt.step()

        if epochs%5 == 0:
            net.eval()
            for im, label in val_loader:
                # load data/label
                im = make_variable(im, requires_grad=False)
                label = make_variable(label, requires_grad=False)
        
                # forward pass and compute loss
                preds = net(im)
                loss = supervised_loss(preds, label)
                
        if epochs%50 == 0:
            net.eval()
            for im, label in val_loader:
                # load data/label
                im = make_variable(im, requires_grad=False)
                label = make_variable(label, requires_grad=False)
        
                # forward pass and compute loss
                preds = net(im)
                loss = supervised_loss(preds, label)


        # log results
        if epochs%1 == 0:
            logging.info('Iteration {}:\t{}'
                            .format(iteration, np.mean(losses)))
            writer.add_scalar('loss', np.mean(losses), iteration)
        iteration += 1
        if step is not None and epochs % step == 0:
            logging.info('Decreasing learning rate by 0.1.')
            step_lr(optimizer, 0.1)
        if epochs % snapshot == 0:
            torch.save(net.state_dict(),
                        '{}-iter{}.pth'.format(output, iteration))
        if epochs % 10 == 0:
            continue
        if epochs == max_epochs - 1:
            logging.info('Optimization complete.')
            break           

if __name__ == '__main__':
    main()
