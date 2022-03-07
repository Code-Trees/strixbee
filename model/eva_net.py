import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, inC, outC, addFlag=False):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=inC, out_channels=outC, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outC)
        self.conv2 = nn.Conv2d(in_channels=inC, out_channels=outC, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outC)

        self.shortcut = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out += self.shortcut(x)
        return out

class MaxBlock(nn.Module):
    def __init__(self, inC, outC):
        super(MaxBlock,self).__init__()

        self.conv = nn.Sequential(           
            nn.Conv2d(in_channels=inC, out_channels=outC, kernel_size=3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(2, 2), 
            nn.BatchNorm2d(outC),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)

class ResNetCustom(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNetCustom, self).__init__()
        self.in_planes = 64

        # PrepLayer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(MaxBlock, ResBlock, 128)
        self.layer2 = self._make_layer(MaxBlock, None, 256)
        self.layer3 = self._make_layer(MaxBlock, ResBlock, 512)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, max_block, r_block, planes):
        layers = []
        
        layers.append(max_block(self.in_planes, planes))

        if(r_block != None):
          layers.append(r_block(planes, planes))
        
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x))) # prep layer

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def EVA_ResNet():
    return ResNetCustom()