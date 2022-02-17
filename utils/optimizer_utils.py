import warnings
warnings.filterwarnings('ignore')

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau
from torch_lr_finder import LRFinder
import torch.nn as nn

def get_optimizer(model_obj,train_loader,loss_type=None,scheduler = False):
    loss_type= str(loss_type).upper()
    parameters = model_obj.parameters()

    if loss_type  == 'L2' :
        optimizer = SGD( params = parameters,lr = 0.01,momentum = 0.9,weight_decay= 0.001)
    else:
        optimizer = SGD( params = parameters,lr = 1e-7,momentum = 0.9)
        criterion = nn.NLLLoss()
        lr_finder = LRFinder(model_obj,optimizer,criterion,device = 'cuda' if torch.cuda.is_available() else 'cpu')
        lr_finder.range_test(train_loader,end_lr = 10,num_iter = 100)
        grapg,lr_rate = lr_finder.plot()
        lr_finder.reset()

        optimizer = SGD( params = parameters,lr =lr_rate,momentum = 0.9)

    if scheduler == True:
        scheduler = ReduceLROnPlateau(optimizer,mode = 'min')
        return optimizer,scheduler
    else:
        return optimizer


#L1 Loss

def L1_loss(model_obj,loss):
        
    l1 = 0
    lambda_l1 = 0.0001
    for p in model_obj.parameters():
        l1 = l1+p.abs().sum()
        loss = loss+ lambda_l1* l1
    return loss

