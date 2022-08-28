import os
# from cv_main import ComplexConv2d, ComplexBatchNorm2d, ComplexLinear, complex_relu
from torch.autograd import Variable
from time import time
# import matplotlib.pyplot as plt
import numpy as np
import gzip, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
from torch.nn import Module, Parameter, init
from torch.nn import Conv2d, Linear, BatchNorm2d
from torch.nn.functional import relu ,dropout
from torchvision import datasets, transforms

from torchvision import datasets, transforms
from complexLayers import ComplexBatchNorm2d
from complexFunctions import complex_relu, complex_max_pool2d, complex_dropout


class Complex_dropout(Module):
    def __init__(self,p=0.5,training=True):
        super(Complex_dropout, self).__init__()
        self.p = p
        self.training = training

    def forward(self, inputr,inputi):
        # need to have the same dropout mask for real and imaginary part,
        # this not a clean solution!
        #mask = torch.ones_like(input).type(torch.float32)
        input = torch.complex(inputr,inputi)
        mask = torch.ones(*input.shape, dtype = torch.float32)
        mask = dropout(mask, self.p, self.training)*1/(1-self.p)
        mask.type(input.dtype)
        return (mask.cuda()*input).real, (mask.cuda()*input).imag


class complex_relu(Module):

    def __init__(self):
        super(complex_relu, self).__init__()
        self.complexrelu = nn.ReLU(True)

    def forward(self, input_r, input_i):
        return self.complexrelu(input_r), self.complexrelu(input_i)

class complex_MaxPool2d(Module):

    def __init__(self):
        super(complex_MaxPool2d, self).__init__()
        self.complexMaxPool2d = nn.MaxPool2d(2)

    def forward(self, input_r, input_i):
        return self.complexMaxPool2d(input_r), self.complexMaxPool2d(input_i)

# def complex_relu(input_r, input_i):
#     return relu(input_r), relu(input_i)
#
# def complex_MaxPool2d(n, input_r, input_i):
#     return nn.MaxPool2d(n)(input_r), nn.MaxPool2d(n)(input_i)

class ComplexConv2d(Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 dilation=1, groups=1, bias=True):
        super(ComplexConv2d, self).__init__()
        self.conv_r = Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_i = Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input_r, input_i):
        assert (input_r.size() == input_i.size())
        # return self.conv_r(input_r), self.conv_r(input_i)
        return self.conv_r(input_r) - self.conv_i(input_i), self.conv_r(input_i) + self.conv_i(input_r)

#
#
class ComplexLinear(Module):

    def __init__(self, in_features, out_features):
        super(ComplexLinear, self).__init__()
        self.fc_r = Linear(in_features, out_features)
        self.fc_i = Linear(in_features, out_features)

    def forward(self, input_r, input_i):
        return self.fc_r(input_r) - self.fc_i(input_i), self.fc_r(input_i) + self.fc_i(input_r)





