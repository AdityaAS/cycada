import torch
import numpy as np

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