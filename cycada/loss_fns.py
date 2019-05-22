import torch

def supervised_loss(score, label, weights=None):
    loss_fn_ = torch.nn.NLLLoss(weight=weights, size_average=True, 
            ignore_index=255)
    loss = loss_fn_(F.log_softmax(score, dim=1), label)
    return loss
   
def discriminator_loss(score, target_val, lsgan=False):
    if lsgan:
        loss = 0.5 * torch.mean((score - target_val)**2)
    else:
        _,_,h,w = score.size()
        target_val_vec = Variable(target_val * torch.ones(1,h,w),requires_grad=False).long().cuda()
        loss = supervised_loss(score, target_val_vec)
    return loss