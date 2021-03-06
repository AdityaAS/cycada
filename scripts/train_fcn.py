import sys
sys.path.append('./')
import logging
import os
from os.path import join, exists
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

import shutil
from cycada.data.data_loader import get_fcn_dataset as get_fcn_dataset
from cycada.models import get_model
from cycada.models.models import models
from cycada.transforms import augment_collate
from cycada.util import config_logging
from cycada.util import to_tensor_raw
from cycada.util import roundrobin_infinite
from cycada.util import preprocess_viz
from cycada.tools.util import make_variable
from cycada.loss_fns import supervised_loss
from cycada.metrics import fast_hist
from cycada.metrics import result_stats
from cycada.metrics import sklearnScores
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--load', dest='load', type=str, help="path to load model", default=False)
parser.add_argument('--c', dest='config', type=str, help="Config file", default=False)
args = parser.parse_args()


def main(config_path):
    config = None
    
    config_file = config_path.split('/')[-1]
    version = config_file.split('.')[0][1:]

    with open(config_path, 'r') as f:
        config = json.load(f)

    config["version"] = version
    config_logging()
    
    # Initialize SummaryWriter - For tensorboard visualizations
    logdir = 'runs/{:s}/{:s}/{:s}/{:s}'.format(config["model"], config["dataset"], 'v{}'.format(config["version"]), 'tflogs')
    logdir = logdir + "/"

    checkpointdir = join('runs', config["model"], config["dataset"], 'v{}'.format(config["version"]), 'checkpoints')

    print("Logging directory: {}".format(logdir))
    print("Checkpoint directory: {}".format(checkpointdir))

    versionpath = join('runs', config["model"], config["dataset"], 'v{}'.format(config["version"]))

    if not exists(versionpath):
        os.makedirs(versionpath)
        os.makedirs(checkpointdir)
        os.makedirs(logdir)
    elif exists(versionpath) and config["force"]:
        shutil.rmtree(versionpath)
        os.makedirs(versionpath)
        os.makedirs(checkpointdir)
        os.makedirs(logdir)
    else:
        print("Version {} already exists! Please run with different version number".format(config["version"]))
        logging.info("Version {} already exists! Please run with different version number".format(config["version"]))
        sys.exit(-1)

    writer = SummaryWriter(logdir)
    # Get appropriate model based on config parameters
    net = get_model(config["model"], num_cls=config["num_cls"])
    if args.load:
        net.load_state_dict(torch.load(args.load))
        print("============ Loading Model ===============")

    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    dataset = config["dataset"] 
    num_workers = config["num_workers"]
    pin_memory = config["pin_memory"]
    dataset = dataset[0]

    datasets_train = get_fcn_dataset(config["dataset"], config["data_type"], join(config["datadir"], config["dataset"]), split='train')
    datasets_val = get_fcn_dataset(config["dataset"], config["data_type"], join(config["datadir"], config["dataset"]), split='val')
    datasets_test = get_fcn_dataset(config["dataset"], config["data_type"], join(config["datadir"], config["dataset"]), split='test')

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

    # val_loader = torch.utils.data.DataLoader(datasets_val, batch_size=config["batch_size"],
    #                                         shuffle=True, num_workers=num_workers,
    #                                         collate_fn=collate_fn,
    #                                         pin_memory=pin_memory)

    test_loader = torch.utils.data.DataLoader(datasets_test, batch_size=config["batch_size"],
                                            shuffle=False, num_workers=num_workers,
                                            collate_fn=collate_fn,
                                            pin_memory=pin_memory)

    data_metric = {'train': None, 'val' : None, 'test' : None}
    Q_size = len(train_loader)/config["batch_size"]

    metrics = {'losses': list(), 'ious': list(), 'recalls': list()}
    
    data_metric['train'] = copy(metrics)
    data_metric['val'] = copy(metrics)
    data_metric['test'] = copy(metrics)
    num_cls = config["num_cls"]
    hist = np.zeros((num_cls, num_cls))
    iteration = 0
    
    for epoch in range(config["num_epochs"]+1):
        if config["phase"] == 'train':
            net.train()
            iterator = tqdm(iter(train_loader))

            # Epoch train
            print("Train Epoch!")
            for im, label in iterator:
                if torch.isnan(im).any() or torch.isnan(label).any():
                    import pdb; pdb.set_trace();
                iteration += 1
                # Clear out gradients
                opt.zero_grad()
                # load data/label
                im = make_variable(im, requires_grad=False)
                label = make_variable(label, requires_grad=False)
                #print(im.size())
        
                # forward pass and compute loss
                preds = net(im)
                #score = preds.data
                #_, pred = torch.max(score, 1)

                #hist += fast_hist(label.cpu().numpy().flatten(), pred.cpu().numpy().flatten(),num_cls)

                #acc_overall, acc_percls, iu, fwIU = result_stats(hist)
                loss = supervised_loss(preds, label)
                # iou = jaccard_score(preds, label)
                precision, rc, fscore, support, iou = sklearnScores(preds, label.type(torch.IntTensor))
                #print(acc_overall, np.nanmean(acc_percls), np.nanmean(iu), fwIU) 
                # backward pass
                loss.backward()

                # TODO: Right now this is running average, ideally we want true average. Make that change
                # Total average will be memory intensive, let it be running average for the moment.
                data_metric['train']['losses'].append(loss.item())
                data_metric['train']['ious'].append(iou)
                data_metric['train']['recalls'].append(rc)
                # step gradients
                opt.step()
                
                # Train visualizations - each iteration
                if iteration % config["train_tf_interval"] == 0:
                    vizz = preprocess_viz(im, preds, label)
                    writer.add_scalar('train/loss', loss, iteration)
                    writer.add_scalar('train/IOU', iou, iteration)
                    writer.add_scalar('train/recall', rc, iteration)
                    imutil = vutils.make_grid(torch.from_numpy(vizz), nrow=3, normalize=True, scale_each=True)
                    writer.add_image('{}_image_data'.format('train'), imutil, iteration)

                iterator.set_description("TRAIN V: {} | Epoch: {}".format(config["version"], epoch))
                iterator.refresh()

                if iteration % 20000 == 0:
                    torch.save(net.state_dict(), join(checkpointdir, 'iter_{}_{}.pth'.format(iteration, epoch)))   

            # clean before test/val
            opt.zero_grad()

            # Train visualizations - per epoch
            vizz = preprocess_viz(im, preds, label)
            writer.add_scalar('trainepoch/loss', np.mean(data_metric['train']['losses']), global_step=epoch)
            writer.add_scalar('trainepoch/IOU', np.mean(data_metric['train']['ious']), global_step=epoch)
            writer.add_scalar('trainepoch/recall', np.mean(data_metric['train']['recalls']), global_step=epoch)
            imutil = vutils.make_grid(torch.from_numpy(vizz), nrow=3, normalize=True, scale_each=True)
            writer.add_image('{}_image_data'.format('trainepoch'), imutil, global_step=epoch)

            print("Loss :{}".format(np.mean(data_metric['train']['losses'])))
            print("IOU :{}".format(np.mean(data_metric['train']['ious'])))
            print("recall :{}".format(np.mean(data_metric['train']['recalls'])))

            if epoch % config["checkpoint_interval"] == 0:
                torch.save(net.state_dict(), join(checkpointdir, 'iter{}.pth'.format(epoch)))   

            # Train epoch done. Free up lists
            for key in data_metric['train'].keys():
                data_metric['train'][key] = list()

            if epoch % config["val_epoch_interval"] == 0:
                net.eval()
                print("Val_epoch!")
                iterator = tqdm(iter(val_loader))
                for im, label in iterator:
                    # load data/label
                    im = make_variable(im, requires_grad=False)
                    label = make_variable(label, requires_grad=False)
            
                    # forward pass and compute loss
                    preds = net(im)
                    loss = supervised_loss(preds, label)
                    precision, rc, fscore, support, iou = sklearnScores(preds, label.type(torch.IntTensor))

                    data_metric['val']['losses'].append(loss.item())
                    data_metric['val']['ious'].append(iou)
                    data_metric['val']['recalls'].append(rc)

                    iterator.set_description("VAL V: {} | Epoch: {}".format(config["version"], epoch))
                    iterator.refresh()

                # Val visualizations
                vizz = preprocess_viz(im, preds, label)
                writer.add_scalar('valepoch/loss', np.mean(data_metric['val']['losses']), global_step=epoch)
                writer.add_scalar('valepoch/IOU', np.mean(data_metric['val']['ious']), global_step=epoch)
                writer.add_scalar('valepoch/Recall', np.mean(data_metric['val']['recalls']), global_step=epoch)
                imutil = vutils.make_grid(torch.from_numpy(vizz), nrow=3, normalize=True, scale_each=True)
                writer.add_image('{}_image_data'.format('val'), imutil, global_step=epoch)

                # Val epoch done. Free up lists
                for key in data_metric['val'].keys():
                    data_metric['val'][key] = list()

            # Epoch Test
            if epoch % config["test_epoch_interval"] == 0:
                net.eval()
                print("Test_epoch!")
                iterator = tqdm(iter(test_loader))
                for im, label in iterator:
                    # load data/label
                    im = make_variable(im, requires_grad=False)
                    label = make_variable(label, requires_grad=False)

                    # forward pass and compute loss
                    preds = net(im)
                    loss = supervised_loss(preds, label)
                    precision, rc, fscore, support, iou = sklearnScores(preds, label.type(torch.IntTensor))

                    data_metric['test']['losses'].append(loss.item())
                    data_metric['test']['ious'].append(iou)
                    data_metric['test']['recalls'].append(rc)

                    iterator.set_description("TEST V: {} | Epoch: {}".format(config["version"], epoch))
                    iterator.refresh()

                # Test visualizations
                writer.add_scalar('testepoch/loss', np.mean(data_metric['test']['losses']), global_step=epoch)
                writer.add_scalar('testepoch/IOU', np.mean(data_metric['test']['ious']), global_step=epoch)
                writer.add_scalar('testepoch/Recall', np.mean(data_metric['test']['recalls']), global_step=epoch)

                # Test epoch done. Free up lists
                for key in data_metric['test'].keys():
                    data_metric['test'][key] = list()

            if config["step"] is not None and epoch % config["step"] == 0:
                logging.info('Decreasing learning rate by 0.1 factor')
                step_lr(optimizer, 0.1)

    logging.info('Optimization complete.')

if __name__ == '__main__':

    p = args.config#sys.argv[1]
    config_path = join('./configs/fcn/', p)

    if exists(config_path):
        main(config_path)
    else :
        print(p)
        print("Incorrect Path")
