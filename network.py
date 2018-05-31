import numpy as np
import math
import torch
import torch.nn as nn
from torchvision.models import ResNet,DenseNet
from collections import OrderedDict
from torchvision import models
import torch.utils.model_zoo as model_zoo
import copy
import torch.nn.functional as F

# Define a residual block
class BasicBlock(nn.Module):
    def __init__(self, block, inplanes, planes, stride=1, first=False):
        super(BasicBlock, self).__init__()
        self.conv1 = block(inplanes, planes, kernel_size=3, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = block(planes, planes, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(planes)

        if first:
            self.downsample = nn.Sequential(block(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                                          nn.BatchNorm2d(planes))
        self.stride = stride
        self.first = first

    def forward(self, x):
        if first:
            residual = self.downsample(x)
        else :
            residual = x
            
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu(out)

        return out


class WideResNet(nn.Module):
    def __init__(self, block, resnet_block, widening_factor=4, kernel_size=3, classes=1000):
        super(WideResNet, self).__init__()
        
        self.block = block
        self.in_channel = 16
        
        self.conv1 = block(3, self.in_channel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer_(resnet_block, 64, widening_factor, stride=2)
        self.layer2 = self._make_layer_(resnet_block, 128, widening_factor, stride=2)
        self.layer3 = self._make_layer_(resnet_block, 256, widening_factor, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        self.fc = nn.ModuleList([nn.Linear(256,classes)]) #initialize with one class
        self.index = 0
        
    def set_index (self, index):
        if( index < len(self.fc)):
            self.index = index
                                 
    def add_task(self, module):
        self.fc.append(module)
        return len(self.fc) - 1 #return actual index of the added module
        
    def _make_layer_(self, block, planes, blocks, stride=1):
        strides = [stride] + [1]*(blocks-1)
        layers = []
        
        for i in range(0, blocks):
            layers.append(block(self.block, self.in_channel, planes, stride=strides[i], first=(i==0)))
            self.inplanes = planes

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc[self.index](x)
        return x

if __name__=="__main__":
    net = WideResNet(nn.Conv2d, BasicBlock, widening_factor=4, classes=1000)
    print(net)
    dict(net.named_parameters()).keys()
