import torch 
from torch.autograd import Variable
import os

def make_variable(tensor, volatile=False, requires_grad=True):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    if volatile:
        requires_grad = False
    return Variable(tensor, volatile=volatile, requires_grad=requires_grad)

def save_model(model, name):
    os.makedirs('/'.join(name.split('/')[:-1]), exist_ok=True)
    checkpoint = {
        'state_dict': model.state_dict(),
    }
    torch.save(checkpoint, name)

def save_opt(model, name):
    os.makedirs('/'.join(name.split('/')[:-1]), exist_ok=True)
    checkpoint = {
        'optimizer': model.state_dict(),
    }
    torch.save(checkpoint, name)

def load_model(model, name):
    if(torch.cuda.is_available()):
        checkpoint = torch.load(name)
    else:
        checkpoint = torch.load(name, map_location=lambda storage, loc: storage)

    model.load_state_dict(checkpoint['state_dict'])
    return model

def load_opt(model, name):
    if(torch.cuda.is_available()):
        checkpoint = torch.load(name)
    else:
        checkpoint = torch.load(name, map_location=lambda storage, loc: storage)

    model.load_state_dict(checkpoint['optimizer'])
    return model
