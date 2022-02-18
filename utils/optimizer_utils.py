import warnings
warnings.filterwarnings('ignore')

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau
from torch_lr_finder import LRFinder
import torch.nn as nn

def run_lrfinder(model_obj,train_loader,loss_type=None,loops = 2)):
    lrs = []
    for i in range(0,loops):

        optimizer = SGD( params = model_class.parameters(),lr = 1e-7,momentum = 0.9,weight_decay= 0.001 if loss_type =='L2' else 0)
        criterion = nn.CrossEntropyLoss()
        lr_finder = LRFinder(model_class,optimizer,criterion,device = 'cuda' if torch.cuda.is_available() else 'cpu')
        lr_finder.range_test(train_loader ,val_loader = test_loadet,end_lr = 100,num_iter = 100,step_mode = 'exp')
        try:
            grapg,lr_rate = lr_finder.plot()
        except:
            pass
        print (f"Lr for min loss :{lr_finder.history['lr'][np.argmin(lr_finder.history['loss'])]},\n loss for suggestd lr  {lr_finder.history['loss'][lr_finder.history['lr'].index(lr_rate)]}")
        lr_finder.reset()

        print(f"Learning rate as LRFinder :{lr_rate}")
        lrs.append(lr_rate)
        
        optimizer = SGD( params = model_obj.parameters(),lr =lr_rate,momentum = 0.9,weight_decay= 0.001 if loss_type =='L2' else 0)
        print(optimizer)
    return lrs


def get_optimizer(model_obj,loss_type=None,scheduler = False,lr = 0.01):
    loss_type= str(loss_type).upper()
    parameters = model_obj.parameters()

    optimizer = SGD( params = model_obj.parameters(),lr = lr,momentum = 0.9,weight_decay= 0.001 if loss_type =='L2' else 0 )
    
    if scheduler == True:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
        return optimizer,scheduler
    else:
        return optimizer


#
#L1 Loss

def L1_loss(model_obj,loss):
        
    l1 = 0
    lambda_l1 = 0.0001
    for p in model_obj.parameters():
        l1 = l1+p.abs().sum()
        loss = loss+ lambda_l1* l1
    return loss

