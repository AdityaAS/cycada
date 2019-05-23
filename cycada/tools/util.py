import torch 
from torch.autograd import Variable
# TODO: GET RID OF THIS - Variable is deprecated in Pytorch 
# Priority - Low
def make_variable(tensor, volatile=False, requires_grad=True):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    if volatile:
        requires_grad = False
    return Variable(tensor, volatile=volatile, requires_grad=requires_grad)

