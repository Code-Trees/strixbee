#model1
class Cifar10Net2(nn.Module):
    def __init__(self,dropout_val = 0,norm_type = 'bn'):
        super(Cifar10Net2,self).__init__()
        self.norm_type = norm_type
        self.drop = dropout_val
 
        self.block1 = nn.Sequential(
            ConvBlock(in_channels=3,out_channels=32,kernel_size = (3,3),stride = 1, padding = 1,norm_type = self.norm_type,dropout_val= self.drop),
            ConvBlock(in_channels=32,out_channels=32,kernel_size = (3,3),stride = 1, padding = 1,norm_type = self.norm_type,dropout_val= self.drop)
            )
        
        # self.pool1 =nn.MaxPool2d(kernel_size=(2,2),stride = 2)
        self.pool1 =ConvBlock(in_channels=32,out_channels =32,kernel_size = (3,3),stride = 2,padding = 1)


        self.block2 = nn.Sequential(
            ConvBlock(in_channels=32,out_channels=64,kernel_size = (3,3),stride = 1, padding = 1,norm_type = self.norm_type,dropout_val= self.drop),
            Depthwise_sep_conv(in_channels = 64, out_channels=128, dropout_val=self.drop, norm_type=self.norm_type, stride=2, padding=1)
            )

        # self.pool2 = nn.MaxPool2d(kernel_size = (2,2),stride  = 2)
        # self.pool2 =ConvBlock(in_channels=32,out_channels =32,kernel_size = (3,3),stride = 2,padding = 0)


        self.block3 = nn.Sequential(
            Depthwise_sep_conv(in_channels = 128, out_channels=128, dropout_val=self.drop, norm_type=self.norm_type, stride=1, padding=1),
            ConvBlock(in_channels=128,out_channels=128,kernel_size = (1,1),stride = 1, padding = 0,norm_type = self.norm_type,dropout_val= self.drop),
            )
        
        self.block4 = nn.Sequential(
            Depthwise_sep_conv(in_channels = 128, out_channels=256, dropout_val=self.drop, norm_type=self.norm_type, stride=1, padding=1),
            ConvBlock(in_channels=256,out_channels=32,kernel_size = (1,1),stride = 1, padding = 0,norm_type = self.norm_type,dropout_val= self.drop),
            )
        
        self.block5 = nn.Sequential(
            ConvBlock(in_channels=32,out_channels=32,kernel_size = (3,3),dilation =2,stride = 1, padding = 1,norm_type = self.norm_type,dropout_val= self.drop),
            ConvBlock(in_channels=32,out_channels=32,kernel_size = (3,3),dilation=2,stride = 1, padding = 1,norm_type = self.norm_type,dropout_val= self.drop),
            )
        
        self.con1 =ConvBlock(in_channels=32,out_channels=64,kernel_size = (3,3),stride = 1, padding = 1,norm_type = self.norm_type,dropout_val= self.drop)
        self.con3 = Depthwise_sep_conv(in_channels = 64, out_channels=128, dropout_val=self.drop, norm_type=self.norm_type, stride=1, padding=1)
        self.con2 = nn.Conv2d(in_channels = 128,out_channels = 10,kernel_size = (3,3),stride = 1, padding = 0,bias= True)

        self.GAP = nn.AdaptiveAvgPool2d(output_size=(1,1))
        

    def forward(self,x):
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        # x = self.pool2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        # # x = self.pool3(x)
        # x = self.block4(x)
        x = self.con1(x)
        x = self.con3(x)
        x = self.con2(x)
        x = self.GAP(x)

        x = x.view(-1,10)
        # return x
        return F.log_softmax(x,dim = -1)