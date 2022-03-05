import warnings
warnings.filterwarnings('ignore')

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau
import numpy as np
from torch_lr_finder import LRFinder

def get_optimizer2(model_obj,loss_type=None,scheduler = False):
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

def get_optimizer(model_obj,loss_type=None,scheduler = False,scheduler_type = 'steplr',lr = 0.01):
    loss_type= str(loss_type).upper()
    parameters = model_obj.parameters()

   
    # optimizer = SGD( params = model_obj.parameters(),lr = lr,momentum = 0.9,weight_decay= 0.001 if loss_type =='L2' else 0 )
    optimizer = SGD( params = parameters,lr = lr,momentum = 0.9 )
    
    if (scheduler == True) & (scheduler_type == 'steplr'):
        scheduler = StepLR(optimizer,step_size = 27,gamma=0.1)
        return optimizer,scheduler

    elif (scheduler == True) & (scheduler_type == 'reducelronplateau'):

        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.01, patience=5, verbose=True, threshold=0.0001,threshold_mode='rel', cooldown=0, min_lr=1e-7, eps=1e-08)
        return optimizer,scheduler
    else:
        return optimizer

def run_lrfinder(model_obj,device,train_loader,test_loader,start_lr,end_lr,loss_type=None):
    lrs  =[]
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_iter = 15*len(train_loader)

    for i in range(0,len(start_lr)):
        opti = SGD( params = model_obj.parameters(),lr = start_lr[i],momentum = 0.9,weight_decay = 0.03) 
        criterion = nn.CrossEntropyLoss()
        lr_finder = LRFinder(model_obj,opti,criterion,device = device)
        lr_finder.range_test(train_loader ,start_lr=start_lr[i],end_lr=end_lr[i], num_iter=num_iter, step_mode='linear')
        try:
            grapg,lr_rate = lr_finder.plot()
        except:
            pass
        print(f"Loss: {lr_finder.best_loss} LR :{lr_rate}")
        lr_finder.reset()
        lrs.append(lr_rate)

        opti = SGD( params = model_obj.parameters(),lr =lr_rate,momentum = 0.9,weight_decay = 0.03)
        print(opti)
    return lrs

#L1 Loss

def L1_loss(model_obj,loss):
        
    l1 = 0
    lambda_l1 = 0.0001
    for p in model_obj.parameters():
        l1 = l1+p.abs().sum()
        loss = loss+ lambda_l1* l1
    return loss

