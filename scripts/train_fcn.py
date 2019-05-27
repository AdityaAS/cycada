import sys
sys.path.append('./')
import logging
import os.path
from collections import deque

import click
import math
import json
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


def compose_transforms(downscale):
    transform = []
    target_transform = []
    if downscale is not None:
        transform.append(torchvision.transforms.Resize(480 // downscale))
        target_transform.append(
            torchvision.transforms.Resize(480 // downscale,
                                         interpolation=Image.NEAREST))

    nettransform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
        ])

    transform.extend([
        torchvision.transforms.Resize(480),
        nettransform
        ])

    target_transform.extend([
        torchvision.transforms.Resize(480, interpolation=Image.NEAREST),
        to_tensor_raw
        ])

    transform = torchvision.transforms.Compose(transform)
    target_transform = torchvision.transforms.Compose(target_transform)
    return transform, target_transform
'''
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
'''
'''
def main(output, phase, dataset, datadir, batch_size, lr, step, iterations, 
        momentum, snapshot, downscale, augmentation, fyu, crop_size, 
        weights, model, gpu, num_cls):'''

def main(config_path):
    config = None
    with open(config_path, 'r') as f:
        config = json.load(f)

    if config["weights"] is not None:
        raise RuntimeError("weights don't work because eric is bad at coding")
    os.environ['CUDA_VISIBLE_DEVICES'] = config["gpu"]
    config_logging()
    
    # Initialize SummaryWriter - For tensorboard visualizations
    logdir = 'runs/{:s}/{:s}'.format(config["model"], config["dataset"])
    logdir = logdir + "/"
    print(logdir)
    #import pdb; pdb.set_trace()
    writer = SummaryWriter(logdir)

    # Get appropriate model based on cmd line architecture
    net = get_model(config["model"], num_cls=config["num_cls"])
    # Get appropriate transforms to apply to input image and target segmask
    transform, target_transform = compose_transforms(config["downscale"])


    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    net.cuda()
    dataset = config["dataset"] 
    num_workers = config["num_workers"]
    pin_memory = config["pin_memory"]
    dataset = dataset[0]

    datasets_train = get_dataset(config["dataset"], os.path.join(config["datadir"], config["dataset"]), split='train',transform=transform,
                        target_transform=target_transform)

    datasets_val = get_dataset(config["dataset"], os.path.join(config["datadir"], config["dataset"]), split='val',transform=transform,
                        target_transform=target_transform)

    datasets_test = get_dataset(config["dataset"], os.path.join(config["datadir"], config["dataset"]), split='test',transform=transform,
                        target_transform=target_transform)

    if config["weights"] is not None:
        weights = np.loadtxt(config["weights"])
    opt = torch.optim.SGD(net.parameters(), lr=config["lr"], momentum=config["momentum"],
                          weight_decay=0.0005)


    if config["augmentation"]:
        collate_fn = lambda batch: augment_collate(batch, crop=config["crop_size"], flip=True)
    else:
        collate_fn = torch.utils.data.dataloader.default_collate

    train_loader = torch.utils.data.DataLoader(datasets_train, batch_size=config["batch_size"],
                                            shuffle=True, num_workers=num_workers,
                                            collate_fn=collate_fn,
                                            pin_memory=pin_memory)

    val_loader = torch.utils.data.DataLoader(datasets_val, batch_size=config["batch_size"],
                                            shuffle=True, num_workers=num_workers,
                                            collate_fn=collate_fn,
                                            pin_memory=pin_memory)

    test_loader = torch.utils.data.DataLoader(datasets_test, batch_size=config["batch_size"],
                                            shuffle=True, num_workers=num_workers,
                                            collate_fn=collate_fn,
                                            pin_memory=pin_memory)


    iteration = 0
    data_metric = {'train': None, 'val' : None, 'test' : None}
    metrics = {'losses': deque(maxlen=10), 'ious': deque(maxlen=10), 'recalls': deque(maxlen=10)}
    data_metric['train'] = copy(metrics)
    data_metric['val'] = copy(metrics)
    data_metric['test'] = copy(metrics)
    max_epochs = math.ceil(config["iterations"]/config["batch_size"])
    for epochs in range(max_epochs):
        if config["phase"] == 'train':
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
                writer.add_image('{}_image_data'.format('train'), imutil, iteration)
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
    p = sys.argv[1]
    if os.path.exists(p):
        main(sys.argv[1])
    else :
        print("Incorrect Path")
