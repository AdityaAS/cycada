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
import apex
from apex import amp
from apex.parallel import DistributedDataParallel
from PIL import Image
from copy import copy
from torch.autograd import Variable

print(os.getcwd())

import shutil
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
from cycada.metrics import fast_hist
from cycada.metrics import result_stats
from cycada.metrics import IoU, recall, sklearnScores
from tqdm import tqdm

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

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

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    else :
        print("NOT ENOUGH GPUS")


    # Get appropriate model based on config parameters
    net = get_model(config["model"], num_cls=config["num_cls"])
    disc_net = nn.Sequential(nn.Conv2d(3, 2, 10), nn.MaxPool2d(2, stride=2), nn.ReLU(), Flatten(), nn.Linear(110450, 1), nn.Sigmoid()).cuda()
    disc_net.load_state_dict(torch.load("/home/users/aditya/sohan/cycada/scripts/iter10.pth"))    #model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    disc_net.eval()
    dataset = config["dataset"] 
    num_workers = config["num_workers"]
    pin_memory = config["pin_memory"]
    dataset = dataset[0]

    datasets_train = get_dataset(config["dataset"], config["data_type"], join(config["datadir"], config["dataset"]), split='train')
    datasets_val = get_dataset(config["dataset"], config["data_type"], join(config["datadir"], config["dataset"]), split='test')
    #datasets_test = get_dataset(config["dataset"], config["data_type"], join(config["datadir"], config["dataset"]), split='test')

    if config["weights"] is not None:
        weights = np.loadtxt(config["weights"])
    #torch.cuda.set_device()
    #torch.distributed.init_process_group(backend='nccl', init_method='env://')
    #torch.backends.cudnn.benchmark = True

    #opt = torch.optim.SGD(net.parameters(), lr=config["lr"], momentum=config["momentum"],
    #                      weight_decay=0.0005)
    opt = torch.optim.Adam(net.parameters(), lr=config["lr"])
    #opt = apex.optimizers.FusedAdam(net.parameters(),weight_decay=0.0005)
    
    #net, opt = amp.initialize(net, opt, opt_level="O1", loss_scale="dynamic")
    #net = nn.DataParallel(net)
    #net = DistributedDataParallel(net)
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

    #test_loader = torch.utils.data.DataLoader(datasets_test, batch_size=config["batch_size"],
    #                                        shuffle=False, num_workers=num_workers,
    #                                        collate_fn=collate_fn,
    #                                        pin_memory=pin_memory)

    data_metric = {'train': None, 'val' : None, 'test' : None}
    Q_size = len(train_loader)/config["batch_size"]

    metrics = {'losses': list(), 'ious': list(), 'recalls': list()}
    
    data_metric['train'] = copy(metrics)
    data_metric['val'] = copy(metrics)
    data_metric['test'] = copy(metrics)
    num_cls = config["num_cls"]
    hist = np.zeros((num_cls, num_cls))
    iteration = 0
    
    for epoch in range(config["num_epochs"]):
        if config["phase"] == 'train':
            net.train()
            iterator = tqdm(iter(train_loader))

            # Epoch train
            print("Train Epoch!")
            for im, label in iterator:

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
                pr, rc, fs, _ = sklearnScores(preds, label)
                #hist += fast_hist(label.cpu().numpy().flatten(), pred.cpu().numpy().flatten(),num_cls)

                #acc_overall, acc_percls, iu, fwIU = result_stats(hist)
                p_source = 1.0 - disc_net(im)
                loss = supervised_loss(preds, label)*1.0/p_source
                loss = loss.mean()
                #with amp.scale_loss(loss, opt) as scaled_loss:
                #    scaled_loss.backward()
                iou = IoU(preds, label)
                #rc = recall(preds, label)
                #print(acc_overall, np.nanmean(acc_percls), np.nanmean(iu), fwIU) 
                # backward pass
                loss.backward()

                # TODO: Right now this is running average, ideally we want true average. Make that change
                # Total average will be memory intensive, let it be running average for the moment.
                data_metric['train']['losses'].append(loss.item())
                data_metric['train']['ious'].append(iou.item())
                data_metric['train']['recalls'].append(rc)
                # step gradients
                opt.step()
                
                # Train visualizations - each iteration
                #if iteration % config["train_tf_interval"] == 0:
                #    vizz = preprocess_viz(im, preds, label)
                #    writer.add_scalar('train/loss', loss.item(), iteration)
                #    writer.add_scalar('train/IOU', iou.item(), iteration)
                #    writer.add_scalar('train/recall', rc.item(), iteration)
                #    imutil = vutils.make_grid(torch.from_numpy(vizz), nrow=3, normalize=True, scale_each=True)
                #    writer.add_image('{}_image_data'.format('train'), imutil, iteration)

                iterator.set_description("TRAIN V: {} | Epoch: {}".format(config["version"], epoch))
                iterator.refresh()

            # Train visualizations - per epoch
            #vizz = preprocess_viz(im, preds, label)
            #writer.add_scalar('trainepoch/loss', np.mean(data_metric['train']['losses']), global_step=epoch)
            #writer.add_scalar('trainepoch/IOU', np.mean(data_metric['train']['ious']), global_step=epoch)
            #writer.add_scalar('trainepoch/recall', np.mean(data_metric['train']['recalls']), global_step=epoch)
            #imutil = vutils.make_grid(torch.from_numpy(vizz), nrow=3, normalize=True, scale_each=True)
            #writer.add_image('{}_image_data'.format('trainepoch'), imutil, global_step=epoch)
            
            print("Loss :{}".format(np.mean(data_metric['train']['losses'])))
            print("Precision :{}".format(np.mean(data_metric['train']['ious'])))
            print("recall :{}".format(np.mean(data_metric['train']['recalls'])))

            if epoch % config["checkpoint_interval"] == 0:
                torch.save(net.state_dict(), join(checkpointdir, 'iter{}.pth'.format(epoch)))   

            # Train epoch done. Free up lists
            for key in data_metric['train'].keys():
                data_metric['train'][key] = list()
            '''
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
                    pr, rc, fs, _ = sklearnScores(preds, label)
                    #iou = IoU(preds, label)
                    #rc = recall(preds, label)

                    data_metric['val']['losses'].append(loss.item())
                    data_metric['val']['ious'].append(pr)
                    data_metric['val']['recalls'].append(rc)

                    iterator.set_description("VAL V: {} | Epoch: {}".format(config["version"], epoch))
                    iterator.refresh()

                print("Loss :{}".format(np.mean(data_metric['train']['losses'])))
                print("Precision :{}".format(np.mean(data_metric['train']['ious'])))
                print("recall :{}".format(np.mean(data_metric['train']['recalls'])))

                # Val visualizations
                #vizz = preprocess_viz(im, preds, label)
                #writer.add_scalar('valepoch/loss', np.mean(data_metric['val']['losses']), global_step=epoch)
                #writer.add_scalar('valepoch/IOU', np.mean(data_metric['val']['ious']), global_step=epoch)
                #writer.add_scalar('valepoch/Recall', np.mean(data_metric['val']['recalls']), global_step=epoch)
                #imutil = vutils.make_grid(torch.from_numpy(vizz), nrow=3, normalize=True, scale_each=True)
                #writer.add_image('{}_image_data'.format('val'), imutil, global_step=epoch)

                # Val epoch done. Free up lists
                for key in data_metric['val'].keys():
                    data_metric['val'][key] = list()
            # Epoch Test
            '''
            '''
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
                    pr, rc, fs, _ = sklearnScores(preds, label)
                    iou = IoU(preds, label)
                    #rc = recall(preds, label)

                    data_metric['test']['losses'].append(loss.item())
                    data_metric['test']['ious'].append(iou.item())
                    data_metric['test']['recalls'].append(rc)

                    iterator.set_description("TEST V: {} | Epoch: {}".format(config["version"], epoch))
                    iterator.refresh()

                # Test visualizations
                #writer.add_scalar('testepoch/loss', np.mean(data_metric['test']['losses']), global_step=epoch)
                #writer.add_scalar('testepoch/IOU', np.mean(data_metric['test']['ious']), global_step=epoch)
                #writer.add_scalar('testepoch/Recall', np.mean(data_metric['test']['recalls']), global_step=epoch)

                # Test epoch done. Free up lists
                for key in data_metric['test'].keys():
                    data_metric['test'][key] = list()
                '''

            if config["step"] is not None and epoch % config["step"] == 0:
                logging.info('Decreasing learning rate by 0.1 factor')
                step_lr(optimizer, 0.1)

    logging.info('Optimization complete.')

if __name__ == '__main__':

    p = sys.argv[1]
    config_path = join('./configs/fcn/', p)
    print(config_path)

    if exists(config_path):
        main(config_path)
    else :
        print(p)
        print("Incorrect Path")