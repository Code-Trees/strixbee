
#custom Resnet
import warnings
warnings.filterwarnings('ignore')


import torch
import torch.nn.functional as F
import torch.nn as nn

# from strixbee.model.custom_layer import *

class Cifar_Net_R(nn.Module):
    def __init__(self,norm_type = 'bn',drop_out = 0.0): 
        super(Cifar_Net_R,self).__init__()
        self.prep_layer = nn.Sequential(
                                        nn.Conv2d(in_channels= 3, out_channels=64 , kernel_size = (3,3), stride = 1, padding = 1,bias= False),
                                        nn.BatchNorm2d(num_features = 64),
                                        nn.ReLU()
                                        )
        
        self.layer1 = nn.Sequential(    
                                    nn.Conv2d(in_channels= 64, out_channels=128 , kernel_size = (3,3), stride = 1, padding = 1,bias= False),
                                    nn.MaxPool2d(kernel_size= (2,2),stride= 2),
                                    nn.BatchNorm2d(num_features = 128),
                                    nn.ReLU()
                                    )
        
        self.ResBlock1 = nn.Sequential(    
                                    nn.Conv2d(in_channels= 128, out_channels=128 , kernel_size = (3,3), stride = 1, padding = 1,bias= False),
                                    nn.BatchNorm2d(num_features = 128),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels= 128, out_channels=128 , kernel_size = (3,3), stride = 1, padding = 1,bias= False),
                                    nn.BatchNorm2d(num_features = 128),
                                    nn.ReLU()
                                    )
        
        self.layer2 = nn.Sequential(    
                                    nn.Conv2d(in_channels= 128, out_channels=256 , kernel_size = (3,3), stride = 1, padding = 1,bias= False),
                                    nn.MaxPool2d(kernel_size= (2,2),stride= 2),
                                    nn.BatchNorm2d(num_features = 256),
                                    nn.ReLU()
                                 )
        
        self.layer3 = nn.Sequential(    
                                    nn.Conv2d(in_channels= 256, out_channels=512 , kernel_size = (3,3), stride = 1, padding = 1,bias= False),
                                    nn.MaxPool2d(kernel_size= (2,2),stride= 2),
                                    nn.BatchNorm2d(num_features = 512),
                                    nn.ReLU()
                                    )

        self.ResBlock2 = nn.Sequential(    
                                    nn.Conv2d(in_channels= 512, out_channels=512 , kernel_size = (3,3), stride = 1, padding = 1,bias= False),
                                    nn.BatchNorm2d(num_features = 512),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels= 512, out_channels=512 , kernel_size = (3,3), stride = 1, padding = 1,bias= False),
                                    nn.BatchNorm2d(num_features = 512),
                                    nn.ReLU()
                                    )
        # self.pool = nn.MaxPool2d(4,4
        self.pool = nn.AvgPool2d(kernel_size = (4,4),stride = 2)  
        self.fc = nn.Linear(in_features = 512,out_features=10)
    
    def forward(self,x):
        X = self.prep_layer(x)
        X1 = self.layer1(X)
        R1 = self.ResBlock1(X)

        X = X1+R1

        X = self.layer2(X)
        X2 = self.layer3(X)
        R2 = self.ResBlock2(X)

        X = X2+R2

        X = self.pool(X)
        X = X.view(-1,512)
        X = self.fc(X)
        return F.softmax(X)