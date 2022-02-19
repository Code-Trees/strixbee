import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau
from torch_lr_finder import LRFinder
import torch.nn as nn
from strixbee.utils.data_iter import get_data,get_data_stats
from strixbee.utils.data_transforms import AlbumDataset
from strixbee.model.cifar10_model import Cifar10Net1


def get_optimizer(model_obj,loss_type=None,scheduler = False):
    loss_type= str(loss_type).upper()
    parameters = model_obj.parameters()
    if loss_type  == 'L2' :
        optimizer = SGD( params = parameters,lr = 0.01,momentum = 0.9,weight_decay= 0.001)
    else:
        optimizer = SGD( params = parameters,lr = 0.01,momentum = 0.9)
    if scheduler == True:
        scheduler = StepLR(optimizer,step_size = 20,gamma = 0.1)
        return optimizer,scheduler
    else:
        return optimizer,_

#
#L1 Loss

def L1_loss(model_obj,loss):
        
    l1 = 0
    lambda_l1 = 0.0001
    for p in model_obj.parameters():
        l1 = l1+p.abs().sum()
        loss = loss+ lambda_l1* l1
    return loss

