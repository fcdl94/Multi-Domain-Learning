import torch.nn as nn
import torch
from collections import OrderedDict

# Define a residual block
class BasicBlock(nn.Module):
    def __init__(self, block, inplanes, planes, kernel_size=3, stride=1, first=False):
        super(BasicBlock, self).__init__()
        self.conv1 = block(inplanes, planes, kernel_size=kernel_size, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = block(planes, planes, kernel_size=kernel_size)
        self.bn2 = nn.BatchNorm2d(planes)

        if first:
            self.downsample = nn.Sequential(block(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                                          nn.BatchNorm2d(planes))
        self.stride = stride
        self.first = first

    def forward(self, x):
        if self.first:
            residual = self.downsample(x)
        else:
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
    def __init__(self, block, resnet_block, widening_factor=4, kernel_size=3, classes=[1000]):
        super(WideResNet, self).__init__()
        
        self.block = block
        self.in_channel = 16
        
        self.conv1 = block(3, self.in_channel, kernel_size=kernel_size, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer_(resnet_block, 64, kernel_size, widening_factor, stride=2)
        self.layer2 = self._make_layer_(resnet_block, 128, kernel_size, widening_factor, stride=2)
        self.layer3 = self._make_layer_(resnet_block, 256, kernel_size, widening_factor, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        self.fc = nn.ModuleList([nn.Linear(256, c) for c in classes])
        self.index = 0
        
    def set_index(self, index):
        if index < len(self.fc):
            self.index = index
                                 
    def add_task(self, module):
        self.fc.append(module)
        return len(self.fc) - 1  # return actual index of the added module
        
    def _make_layer_(self, block, planes, kernel_size, blocks, stride=1):
        strides = [stride] + [1]*(blocks-1)
        layers = []
        for i in range(0, blocks):
            layers.append(block(self.block, self.in_channel, planes,kernel_size, stride=strides[i], first=(i == 0)))
            self.in_channel = planes

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


def wide_resnet(block, model_classes, pretrained=None, frozen=False):
    model = WideResNet(block, BasicBlock, 4, 3, classes=model_classes)
    if pretrained:
        try:
            state = model.state_dict()
            state.update(torch.load(pretrained)['state_dict'])
            model.load_state_dict(state)
        except:
            model_dict = model.state_dict()
            state_dict = torch.load(pretrained)['state_dict']
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                if name in model_dict:
                    new_state_dict[name] = v
            state = model.state_dict()
            state.update(new_state_dict)
            model.load_state_dict(state)

    if frozen:
        for name, param in model.named_parameters():
            if "fc" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    return model


if __name__ == "__main__":
    net = WideResNet(nn.Conv2d, BasicBlock, widening_factor=4, classes=1000)
    print(net)
    print(dict(net.named_parameters()).keys())
