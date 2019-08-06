import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score

def seg_accuracy(score, label, num_cls):
    _, preds = torch.max(score.data, 1)
    hist = fast_hist(label.cpu().numpy().flatten(),
            preds.cpu().numpy().flatten(), num_cls)
    intersections = np.diag(hist)
    unions = (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-8) * 100
    acc = np.diag(hist).sum() / hist.sum()
    return intersections, unions, acc


def IoU(preds, label):
    
    SMOOTH = 1e-6

    max_vals , pred_label = torch.max(preds, dim=1)

    pred_label = pred_label.long()
    label = label.long()
    
    intersection = (pred_label & label).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (pred_label | label).float().sum((1, 2))         # Will be zzero if both are 0
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    iou = iou.mean()

    return iou

def recall(preds, label):

    max_vals , pred_label = torch.max(preds, dim=1)

    pred_label = pred_label.long()
    label = label.long()

    intersection = (pred_label & label).float().sum((1, 2))
    total = label.float().sum((1, 2)) 
    ratio = intersection/total
    ratio = ratio.mean()

    return ratio


def precision(preds, label):
    SMOOTH = 1e-6

    max_vals , pred_label = torch.max(preds, dim=1)

    pred_label = pred_label.long()
    label = label.long()
    
    intersection = (pred_label & label).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = ((pred_label & label).byte() + (pred_label.byte() & (~label.byte()))).float().sum((1, 2))         # Will be zzero if both are 0
    
    precision = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    precision = precision.mean()

    return precision

def sklearnScores(preds, label):
    #import pdb; pdb.set_trace()
    max_vals , pred_label = torch.max(preds, dim=1)

    pred_label = pred_label.view(-1, 1).long().cpu().data.numpy()
    label = label.view(-1, 1).long().cpu().data.numpy()

    precision, recall, fscore, support = score(label, pred_label, average='micro')

    return precision, recall, fscore, support

def accuracy(preds, labels):

    max_vals , pred_label = torch.max(preds, dim=1)

    pred_label = torch.squeeze(pred_label).cpu().numpy()
    label = torch.squeeze(labels).cpu().numpy()
    return accuracy_score(label, pred_label)

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