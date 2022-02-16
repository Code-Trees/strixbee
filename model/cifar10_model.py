import warnings
warnings.filterwarnings('ignore')
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
import torch.nn as nn

from model.custom_layer import *


class Cifar10Net1(nn.Module):
    def __init__(self,dropout_val = 0,norm_type = 'bn'):
        super(Cifar10Net1,self).__init__()
        self.norm_type = norm_type
        self.drop = dropout_val

        self.block1 = nn.Sequential(
            ConvBlock(in_channels=3,out_channels=8,kernel_size = (3,3),stride = 1, padding = 0,norm_type = self.norm_type,dropout_val= self.drop),
            ConvBlock(in_channels=8,out_channels=16,kernel_size = (3,3),stride = 1, padding = 1,norm_type = self.norm_type,dropout_val= self.drop),
            ConvBlock(in_channels=16,out_channels=32,kernel_size = (3,3),stride = 1, padding = 1,norm_type = self.norm_type,dropout_val= self.drop),
            )
        
        self.pool1 =ConvBlock(in_channels=32,out_channels=32,kernel_size = (3,3),stride = 2, padding = 1,norm_type = self.norm_type,dropout_val= self.drop)


        self.block2 = nn.Sequential(
            ConvBlock(in_channels=32,out_channels=16,kernel_size = (1,1),stride = 1, padding = 1,norm_type = self.norm_type,dropout_val= self.drop),
            Depthwise_sep_conv(in_channels=16,out_channels=32,dropout_val=self.drop,norm_type = self.norm_type,stride = 1,padding = 1),
            Depthwise_sep_conv(in_channels=32,out_channels=64,dropout_val=self.drop,norm_type = self.norm_type,stride = 1,padding = 1),
            # ConvBlock(in_channels=32,out_channels=64,kernel_size = (3,3),stride = 1, padding = 1,norm_type = self.norm_type,dropout_val= self.drop),
            )

        self.pool2 = ConvBlock(in_channels=64,out_channels=64,kernel_size = (3,3),stride = 2, padding = 0,norm_type = self.norm_type,dropout_val= self.drop)
        
        self.block3 = nn.Sequential(
            ConvBlock(in_channels=64,out_channels=32,kernel_size = (1,1),stride = 1, padding =1,norm_type = self.norm_type,dropout_val= self.drop),
            Depthwise_sep_conv(in_channels=32,out_channels=64,dropout_val=self.drop,norm_type = self.norm_type,stride = 1,padding =1),
            Depthwise_sep_conv(in_channels=64,out_channels=128,dropout_val=self.drop,norm_type = self.norm_type,stride = 1,padding =1),
            )

        self.conv_block1 = ConvBlock(in_channels=128,out_channels=32,kernel_size = (1,1),stride = 1, padding =0,norm_type = self.norm_type,dropout_val= self.drop)
        self.conv_block2 = ConvBlock(in_channels=32,out_channels=32,kernel_size = (3,3),stride = 1, padding =0,norm_type = self.norm_type,dropout_val= self.drop,dilation =2)
        self.conv_block3 = ConvBlock(in_channels=32,out_channels=64,kernel_size = (3,3),stride = 1, padding =0,norm_type = self.norm_type,dropout_val= self.drop,dilation =2)

        # self.conv_block1 = nn.Conv2d(in_channels=128,out_channels=32,kernel_size = (1,1),stride = 1, padding =0,bias= False)
        # self.conv_block2 = nn.Conv2d(in_channels=32,out_channels=32,kernel_size = (3,3),stride = 1, padding =0,dilation =2,bias= False)
        self.conv_block3 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size = (3,3),stride = 1, padding =0,bias= False)
        
        
        self.GAP = nn.AdaptiveAvgPool2d(output_size=(1,1))
        # self.GAP = nn.AvgPool2d(kernel_size = (5,5))
        self.linear = nn.Linear(in_features =64 ,out_features = 10)
        # self.conv_block4 = nn.Conv2d(in_channels=64,out_channels=10,kernel_size = (3,3),stride = 1, padding =0)
        # self.conv_block5 = nn.Conv2d(in_channels=64,out_channels=10,kernel_size = (1,1),stride = 1, padding =0)
        # self.conv_block4 = nn.Conv2d(in_channels=32,out_channels=10,kernel_size = (1,1),stride = 1, padding =0,bias= False)



    def forward(self,x):
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = self.block3(x)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.GAP(x)
        # x = self.conv_block4(x)
        # x = self.conv_block5(x)
        # return x
        x = x.view(-1,64)
        x = self.linear(x)
        return F.log_softmax(x,dim = -1)