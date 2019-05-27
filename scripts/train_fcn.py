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
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from PIL import Image
from copy import copy
from torch.autograd import Variable
print(os.getcwd())
from cycada.data.data_loader import get_fcn_dataset as get_dataset
from cycada.models import get_model
from cycada.models.models import models
from cycada.transforms import augment_collate
from cycada.util import config_logging
from cycada.util import to_tensor_raw
from cycada.util import roundrobin_infinite
from cycada.util import preprocess_viz
from cycada.tools.util import make_variable
from cycada.loss_fns import supervised_loss
from cycada.metrics import IoU, recall
from tqdm import tqdm


# TODO (Design Choic: Passing dataloders directly to the train function v/s passing dataset path and then building loaders inside the train function?
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
# Maybe rewrite this to have epoch_train, epoch_val, epoch_test etc.
def main(output, phase, dataset, datadir, batch_size, lr, step, iterations, 
        momentum, snapshot, downscale, augmentation, fyu, crop_size, 
        weights, model, gpu, num_cls):

    if weights is not None:
        raise RuntimeError("weights don't work because eric is bad at coding")

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    config_logging()
    
    # Initialize SummaryWriter - For tensorboard visualizations
    logdir = 'runs/{:s}/{:s}'.format(model, '-'.join(dataset))
    writer = SummaryWriter(logdir=logdir)

    # Get appropriate model based on cmd line architecture
    net = get_model(model, num_cls=num_cls)

    # Get appropriate transforms to apply to input image and target segmask
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    net.cuda()
    
    dataset = dataset[0]

    datasets_train = get_dataset(dataset, os.path.join(datadir, dataset), split='train')
    datasets_val = get_dataset(dataset, os.path.join(datadir, dataset), split='val')
    datasets_test = get_dataset(dataset, os.path.join(datadir, dataset), split='test')

    if weights is not None:
        weights = np.loadtxt(weights)
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum,
                          weight_decay=0.0005)

    #TODO: What is augment_collate doing?
    if augmentation:
        collate_fn = lambda batch: augment_collate(batch, crop=crop_size, flip=True)
    else:
        collate_fn = torch.utils.data.dataloader.default_collate

    train_loader = torch.utils.data.DataLoader(datasets_train, batch_size=batch_size,
                                            shuffle=True, num_workers=0,
                                            collate_fn=collate_fn,
                                            pin_memory=True)

    val_loader = torch.utils.data.DataLoader(datasets_val, batch_size=batch_size,
                                            shuffle=True, num_workers=0,
                                            collate_fn=collate_fn,
                                            pin_memory=True)

    #TODO: num workers, pin_memory, batch_size must be config arguments.

    # test_loader shuffle must be false
    test_loader = torch.utils.data.DataLoader(datasets_test, batch_size=batch_size,
                                            shuffle=False, num_workers=0,
                                            collate_fn=collate_fn,
                                            pin_memory=True)


    iteration = 0
    data_metric = {'train': None, 'val' : None, 'test' : None}
    metrics = {'losses': deque(maxlen=10), 'ious': deque(maxlen=10), 'recalls': deque(maxlen=10)}
    data_metric['train'] = copy(metrics)
    data_metric['val'] = copy(metrics)
    data_metric['test'] = copy(metrics)
    max_epochs = math.ceil(iterations/batch_size)

    for epochs in range(max_epochs):
        if phase == 'train':
            net.train()
            train_loader = iter(train_loader)
            for im, label in tqdm(train_loader):

                # Clear out gradients
                opt.zero_grad()
                
                # load data/label
                im = make_variable(im, requires_grad=False)
                label = make_variable(label, requires_grad=False)
        
                # forward pass and compute loss
                preds = net(im)
                loss = supervised_loss(preds, label)
                iou = IoU(preds, label)
                rc = recall(preds, label)

                # backward pass
                loss.backward()
                # TODO: Right now this is running average, ideally we want true average. Make that change
                data_metric['train']['losses'].append(loss.item())
                data_metric['train']['ious'].append(iou.item())
                data_metric['train']['recalls'].append(rc.item())
                # step gradients
                opt.step()
                
                #writer
                vizz = preprocess_viz(im, preds, label)
                writer.add_scalar('train_loss', np.mean(data_metric['train']['losses']), iteration)
                writer.add_scalar('train_IOU', np.mean(data_metric['train']['ious']), iteration)
                writer.add_scalar('train_Recall', np.mean(data_metric['train']['recalls']), iteration)
                imutil = vutils.make_grid(torch.from_numpy(vizz), nrow=3, normalize=True, scale_each=True)
                writer.add_image('{}_image_data'.format(phase), imutil, iteration)
                iteration = iteration + 1
            
            print(epochs, loss)

            # Run val epoch for every 5 train epochs
            if epochs%1 == 0:
                net.eval()
                for im, label in val_loader:
                    # load data/label
                    im = make_variable(im, requires_grad=False)
                    label = make_variable(label, requires_grad=False)
            
                    # forward pass and compute loss
                    preds = net(im)
                    loss = supervised_loss(preds, label)

                    iou = IoU(preds, label)
                    rc = recall(preds, label)

                    data_metric['val']['losses'].append(loss.item())
                    data_metric['val']['ious'].append(iou.item())
                    data_metric['val']['recalls'].append(rc.item())

                    vizz = preprocess_viz(im, preds, label)
                    writer.add_scalar('train_loss', np.mean(data_metric['val']['losses']), iteration)
                    writer.add_scalar('train_IOU', np.mean(data_metric['val']['ious']), iteration)
                    writer.add_scalar('train_Recall', np.mean(data_metric['val']['recalls']), iteration)
                    imutil = vutils.make_grid(torch.from_numpy(vizz), nrow=3, normalize=True, scale_each=True)
                    writer.add_image('{}_image_data'.format('val'), imutil, iteration)
        
            # Run test epoch for every 50 train epochs
            if epochs%50 == 0:
                net.eval()
                for im, label in test_loader:
                    # load data/label
                    im = make_variable(im, requires_grad=False)
                    label = make_variable(label, requires_grad=False)

                    # forward pass and compute loss
                    preds = net(im)
                    loss = supervised_loss(preds, label)
                    iou = IoU(preds, label)
                    rc = recall(preds, label)

                    data_metric['test']['losses'].append(loss.item())
                    data_metric['test']['ious'].append(iou.item())
                    data_metric['test']['recalls'].append(rc.item())

            # log results
            if epochs%1 == 0:
                logging.info('Iteration {}:\t{}'
                                .format(iteration, np.mean(losses)))
                writer.add_scalar('loss', np.mean(losses), iteration)

                vizz = preprocess_viz(im, preds, label)
                writer.add_scalar('train_loss', np.mean(data_metric['test']['losses']), iteration)
                writer.add_scalar('train_IOU', np.mean(data_metric['test']['ious']), iteration)
                writer.add_scalar('train_Recall', np.mean(data_metric['test']['recalls']), iteration)
                imutil = vutils.make_grid(torch.from_numpy(vizz), nrow=3, normalize=True, scale_each=True)
                writer.add_image('{}_image_data'.format('test'), imutil, iteration)

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
