#custom Resnet
class ResNetc(nn.Module):
    def __init__(self,norm_type = 'bn',drop_val = 0):
        super(ResNetc,self).__init__()
        self.norm_type = norm_type
        self.drop_val = drop_val
        
        
        self.prep_layer = ConvBlock(in_channels = 3, out_channels=64, stride = 1, padding = 1, kernel_size=(3, 3), norm_type=self.norm_type, dropout_val=self.drop_val, dilation=1)

        self.layer1 = nn.Sequential(
                                        nn.Conv2d(in_channels = 64, out_channels=128,kernel_size=(3, 3), stride = 1, padding = 1,),
                                        nn.MaxPool2d(kernel_size=(2,2),stride = 2),
                                        Normalize(norm_type = self.norm_type ,num_features= 128),
                                        nn.ReLU()
                                     )
        
        self.layer1_R = nn.Sequential(
                                        ConvBlock(in_channels=128, out_channels=128, stride = 1, padding = 1, kernel_size=(3, 3), norm_type=self.norm_type, dropout_val=self.drop_val, dilation=1),
                                        ConvBlock(in_channels=128, out_channels=128, stride = 1, padding = 1, kernel_size=(3, 3), norm_type=self.norm_type, dropout_val=self.drop_val, dilation=1),
                                    )
        
        self.layer2  = nn.Sequential(
                                        nn.Conv2d(in_channels = 128, out_channels=256,kernel_size=(3, 3), stride = 1, padding = 1,),
                                        nn.MaxPool2d(kernel_size=(2,2),stride = 2),
                                        Normalize(norm_type = self.norm_type ,num_features= 256),
                                        nn.ReLU()
                                     )
        
        self.layer3 = nn.Sequential(
                                        nn.Conv2d(in_channels = 256, out_channels=512,kernel_size=(3, 3), stride = 1, padding = 1,),
                                        nn.MaxPool2d(kernel_size=(2,2),stride = 2),
                                        Normalize(norm_type = self.norm_type ,num_features= 512),
                                        nn.ReLU()
                                     )
        
        self.layer3_R = nn.Sequential(
                                        ConvBlock(in_channels=512, out_channels=512, stride = 1, padding = 1, kernel_size=(3, 3), norm_type=self.norm_type, dropout_val=self.drop_val, dilation=1),
                                        ConvBlock(in_channels=512, out_channels=512, stride = 1, padding = 1, kernel_size=(3, 3), norm_type=self.norm_type, dropout_val=self.drop_val, dilation=1),
                                    )
        
        self.pool = nn.MaxPool2d(kernel_size=(4,4),stride = 2)

        self.fc = nn.Linear(in_features = 512, out_features = 10, bias=False)

    def forward(self,X):
        X = self.prep_layer(X)

        X = self.layer1(X)
        R1 = self.layer1_R(X)
        X = X + R1

        X = self.layer2(X)

        X = self.layer3(X)
        R2 = self.layer3_R(X)

        X = self.pool(X)

        X = X.view(-1,512)
        X = self.fc(X)
        
        X = F.softmax(X)
        
        return X

def test_model():
    net = ResNetc()
    y = net(torch.randn(1, 3, 32, 32))
    print(f"Model Final Output size is {y.size()}")

test_model()