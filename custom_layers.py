import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn as nn

# Author: Massimo Mancini


# Piggyback implementation
class MaskedConv2d(nn.modules.conv.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=False):

        super(MaskedConv2d, self).__init__(
                in_channels, out_channels, kernel_size, stride, padding, dilation,
                groups, bias)

        self.weight.requires_grad = False

        if bias:
            self.bias.requires_grad = False

        self.threshold = 0.0
        self.mask = Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))

        self.reset_mask()

    def reset_mask(self):
        self.mask.data.fill_(0.01)

    def forward(self, input):
        binary_mask = self.mask.clone()
        binary_mask.data = (binary_mask.data > self.threshold).float()
        W = binary_mask*self.weight
        return F.conv2d(input, W, self.bias, self.stride, self.padding, self.dilation, self.groups)
