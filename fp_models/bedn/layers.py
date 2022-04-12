# import APIs
from turtle import forward
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import os
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as D
import torchvision.transforms.functional as Fv

from torchvision.utils import save_image
from torchvision.utils import draw_segmentation_masks as maskShow
from torch.autograd import Function

# from torchvision import datasets,transforms

# define Sign-Activation for 1-bit quantized activation
class SignActivation(Function):
    @staticmethod
    def forward(self, input):
        output = input.new(input.size())
        output[input >= 0] = 1
        output[input < 0] = -1
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input
# aliases
binarize = SignActivation.apply

class Binarization(nn.Module):
    def __init__(self) :
        super(Binarization, self).__init__()
    def forward(self, x):
        return binarize(x)

#  define binarized neural networks layers
class BinaryTanh(nn.Module):
    def __init__(self):
        super(BinaryTanh, self).__init__()
        self.Tanh = nn.Tanh()
        self.binarize = Binarization()

    def forward(self, input):
        output = self.Tanh(input).mul(2) #mul(2) for acceleration
        output = self.binarize(output)
        return output
        
class BinaryLinear(nn.Linear):
    def forward(self, input):
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=binarize(self.weight.org)
        out = nn.functional.linear(input, self.weight)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)
        return out

class BednBinConv2d(nn.Conv2d):
    def __init__(self, *kargs, **kwargs):
        super(BednBinConv2d, self).__init__(*kargs, **kwargs)
        self.weight.bin = True
        self.binarize = Binarization()
    def forward(self, input):
        weights = self.binarize(self.weight)
        out = nn.functional.conv2d(input, weights, None, self.stride,
                                   self.padding, self.dilation, self.groups)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)
        return out

class BednBinConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, *kargs, **kwargs):
        super(BednBinConvTranspose2d, self).__init__(*kargs, **kwargs)
        self.binarize = Binarization()
        self.weight.bin = True
    def forward(self, input):
        weights = self.binarize(self.weight)
        out = nn.functional.conv_transpose2d(input, weights, None, self.stride,
                                   self.padding, self.dilation, self.groups)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)
        return out

bnorm_momentum= 0.5
# define Binarized encoder decoder network (BEDN)
class Model(nn.Module):

    def __init__(self, ncls=3):
        super(Model, self).__init__()
        
        self.encoders = nn.Sequential(
            nn.Conv2d(3, 64 , kernel_size=3, padding=1),
            nn.BatchNorm2d(64,eps=1e-4, momentum=bnorm_momentum),
            BinaryTanh(),

            BednBinConv2d(64 , 64 , kernel_size=3, padding=1),
            nn.BatchNorm2d(64,eps=1e-4, momentum=bnorm_momentum),
            BinaryTanh(),

            BednBinConv2d(64 , 128 , kernel_size=3, stride=2 ,padding=[1,1]),
            nn.BatchNorm2d(128,eps=1e-4, momentum=bnorm_momentum),
            BinaryTanh(),

            BednBinConv2d(128, 128 , kernel_size=3, padding=1),
            nn.BatchNorm2d(128,eps=1e-4, momentum=bnorm_momentum),
            BinaryTanh(),

            BednBinConv2d(128, 256, kernel_size=3, stride=2, padding=[1,1]),
            nn.BatchNorm2d(256,eps=1e-4, momentum=bnorm_momentum),
            BinaryTanh(),

            BednBinConv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256,eps=1e-4, momentum=bnorm_momentum),
            BinaryTanh(),
        )
        self.decoders = nn.Sequential(
            BednBinConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=[1,1]),
            nn.BatchNorm2d(128,eps=1e-4, momentum=bnorm_momentum),
            BinaryTanh(),

            BednBinConv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128,eps=1e-4, momentum=bnorm_momentum),
            BinaryTanh(),
            
            BednBinConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=[1,1]),
            nn.BatchNorm2d(64,eps=1e-4, momentum=bnorm_momentum),
            BinaryTanh(),

            BednBinConv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64,eps=1e-4, momentum=bnorm_momentum),
            BinaryTanh(),

            BednBinConv2d(64, ncls, kernel_size=3, padding=1),
            nn.BatchNorm2d(ncls,eps=1e-4, momentum=bnorm_momentum),
        )

    def forward(self, x):
        x = self.encoders(x)
        #x = x.view(-1, 256)
        x = self.decoders(x)
        return x