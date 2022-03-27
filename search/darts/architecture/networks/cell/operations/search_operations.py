import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .binarization.binarized_blocks import BinConvBnHTanh, BinTConvBnHTanh, BinConvBn,ConvBn, BinConv
from .binarization.binarized_blocks import ConvBnHTanhBin, TConvBnHTanhBin 
from .fp.fp_block import FpConvBnHardtanh, FpTConvBnHardtanh

####################
# Fp ops
class FpConv1x1(FpConvBnHardtanh):
    def __init__(self, in_channels, out_channels, bn_layer=True, activation='htanh', affine=True, kernel_size=1, padding=0, dilation=1, padding_mode='zeros', stride=1):
        super(FpConv1x1, self).__init__(in_channels, out_channels, bn_layer, activation, affine, kernel_size, padding, dilation, padding_mode, stride)
        


class FpConv3x3(FpConvBnHardtanh):
    def __init__(self, in_channels, out_channels, bn_layer=True, activation='htanh', affine=True, kernel_size=3, padding=1, dilation=1, padding_mode='zeros', stride=1):
        super(FpConv3x3, self).__init__(in_channels=in_channels, out_channels=out_channels, bn_layer=bn_layer, activation=activation, affine=affine, kernel_size=kernel_size, padding=padding, dilation=dilation, padding_mode=padding_mode, stride=stride)
        


class FpDilConv3x3(FpConvBnHardtanh):
    def __init__(self, in_channels, out_channels, bn_layer=True, activation='htanh', affine=True, kernel_size=3, padding=0, dilation=1, padding_mode='zeros', stride=1):
        super(FpDilConv3x3, self).__init__(in_channels=in_channels, out_channels=out_channels, bn_layer=bn_layer, activation=activation, affine=affine, kernel_size=kernel_size, padding=padding, dilation=dilation, padding_mode=padding_mode, stride=stride)
        

class FpTConv1x1(FpTConvBnHardtanh):
    def __init__(self, in_channels, out_channels, bn_layer=True, activation='htanh', affine=True, kernel_size=1, padding=0, dilation=1, padding_mode='zeros', stride=1, output_padding=1):
        super(FpTConv1x1, self).__init__(in_channels, out_channels, bn_layer, activation, affine, kernel_size, padding, dilation, padding_mode, stride, output_padding)
        


class FpTConv3x3(FpTConvBnHardtanh):
    def __init__(self, in_channels, out_channels, bn_layer=True, activation='htanh', affine=True, kernel_size=3, padding=0, dilation=1, padding_mode='zeros', stride=1, output_padding=1):
        super(FpTConv3x3, self).__init__(in_channels, out_channels, bn_layer, activation, affine, kernel_size, padding, dilation, padding_mode, stride, output_padding)
        


class FpTConv5x5(FpTConvBnHardtanh):
    def __init__(self, in_channels, out_channels, bn_layer=True, activation='htanh', affine=True, kernel_size=5, padding=0, dilation=1, padding_mode='zeros', stride=1, output_padding=1):
        super(FpTConv5x5, self).__init__(in_channels, out_channels, bn_layer, activation, affine, kernel_size, padding, dilation, padding_mode, stride, output_padding)
        


class FpDilTConv3x3(FpTConvBnHardtanh):
    def __init__(self, in_channels, out_channels, bn_layer=True, activation='htanh', affine=True, kernel_size=3, padding=0, dilation=1, padding_mode='zeros', stride=1, output_padding=1):
        super(FpDilTConv3x3, self).__init__(in_channels, out_channels, bn_layer, activation, affine, kernel_size, padding, dilation, padding_mode, stride, output_padding)
        


class FpConv3x3bn(FpConvBnHardtanh):
    def __init__(self, in_channels, out_channels, bn_layer=True, activation=False, affine=True, kernel_size=1, padding=0, dilation=1, padding_mode='zeros', stride=1):
        super(FpConv3x3bn, self).__init__(in_channels, out_channels, bn_layer, activation, affine, kernel_size, padding, dilation, padding_mode, stride)
        

####################
# bin ops
class BinConvT1x1(BinTConvBnHTanh):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=2, padding=0, dilation=1, affine=True, padding_mode='zeros', jit=False,dropout2d=0.1,binarization=1, activation='htanh'):
        super(BinConvT1x1, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, affine, padding_mode, jit,dropout2d, binarization,activation)
        


class BinConvT3x3(BinTConvBnHTanh):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, dilation=1, affine=True,padding_mode='zeros',jit=False,dropout2d=0.1, binarization=1, activation='htanh'):
        super(BinConvT3x3, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, affine, padding_mode, jit,dropout2d,binarization,activation)
        


class BinConvT5x5(BinTConvBnHTanh):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=2, padding=1, dilation=1, affine=True, padding_mode='zeros', jit=False,dropout2d=0.1, binarization=1, activation='htanh'):
        super(BinConvT5x5, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, affine, padding_mode, jit,dropout2d, binarization,activation)
        


class BinDilConv3x3(BinConvBnHTanh):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=2, dilation=2,affine=True, padding_mode='zeros',jit=False,dropout2d=0.1, binarization=1, activation='htanh'):
        super(BinDilConv3x3, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,affine, padding_mode,jit,dropout2d,binarization,activation)
        


class BinConv1x1(BinConvBnHTanh):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, affine=True, padding_mode='zeros',jit=False,dropout2d=0.1, binarization=1, activation='htanh'):
        super(BinConv1x1, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,affine, padding_mode,jit,dropout2d,binarization,activation)
        
# last layer
class BinConvbn1x1(BinConvBn):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, affine=True, padding_mode='zeros', jit=False,dropout2d=0.1, binarization=1, activation='htanh'):
        super(BinConvbn1x1, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,affine, padding_mode, jit,dropout2d,binarization)


class Convbn1x1(ConvBn):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, affine=True, padding_mode='zeros', jit=False,dropout2d=0.1, binarization=1, activation='htanh'):
        super(Convbn1x1, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,affine, padding_mode, jit,dropout2d,binarization)


class BinConv3x3(BinConvBnHTanh):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, affine=True, padding_mode='zeros', jit=False,dropout2d=0.1, binarization=1, activation='htanh'):
        super(BinConv3x3, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, affine, padding_mode, jit, dropout2d, binarization,activation)
        
class BasicBinConv1x1(BinConv):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, affine=True, padding_mode='zeros', jit=False, binarization=1, activation='htanh'):
        super(BasicBinConv1x1, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, affine, padding_mode, jit,binarization)


class Identity(nn.Module):
    def __init__(self, params=None):
        super(Identity, self).__init__()
        
    
    def forward(self, x):
        return x
    
    def get_config(self):
        return {'params':None}

    @classmethod
    def model(cls, config):
        return cls(**config)


class Bilinear(nn.Upsample):
    def __init__(self, size= None, scale_factor = 2, mode: str = 'bilinear', align_corners = False):
        super(Bilinear, self).__init__(scale_factor=scale_factor, mode=mode, align_corners=align_corners)


class AvgPool(nn.Module):
    def __init__(self, in_channels, affine, stride = None, kernel_size=3, padding= 1, activation='htanh'):
        super(AvgPool, self).__init__()
        self.latency_table = {}
        padding = kernel_size//2
        self.activation_func = activation
        self.affine = affine
        self.layers = nn.Sequential(
            nn.AvgPool2d(kernel_size, stride, padding)
        )
        if activation == 'relu':
            self.layers.add_module('activation', nn.ReLU())
        self.layers.add_module('batchnorm', nn.BatchNorm2d(in_channels, affine=affine))
        if activation == 'htanh':
            self.layers.add_module('activation', nn.Hardtanh())
        self.in_channels = in_channels
    def forward(self, x):
        return self.layers(x)
    
    def get_config(self):
        return {'in_channels':self.in_channels, 'kernel_size':self.layers[0].kernel_size, 'stride':self.layers[0].stride,'padding':self.layers[0].padding, 'affine':self.affine, 'activation':self.activation_func}
    
    @classmethod
    def model(cls, config):
        return cls(**config)


class MaxPool(nn.Module):
    def __init__(self, in_channels, affine, kernel_size=3, stride = None, padding= 1, dilation=1, activation='htanh'):
        super(MaxPool, self).__init__()
        self.latency_table = {}
        padding = kernel_size//2
        self.activation_func = activation
        self.affine = affine
        self.layers = nn.Sequential(
            nn.MaxPool2d(kernel_size, stride, padding,dilation)
        )
        if activation == 'relu':
            self.layers.add_module('activation', nn.ReLU())
        self.layers.add_module('batchnorm', nn.BatchNorm2d(in_channels, affine=affine))
        if activation == 'htanh':
            self.layers.add_module('activation', nn.Hardtanh())
        self.in_channels = in_channels
    def forward(self, x):
        return self.layers(x)
    
    def get_config(self):
        return {'in_channels':self.in_channels, 'kernel_size':self.layers[0].kernel_size, 'stride':self.layers[0].stride,'padding':self.layers[0].padding, 'affine':self.affine, 'activation':self.activation_func}
    
    @classmethod
    def model(cls, config):
        return cls(**config)


class Cat(nn.Module):
    def __init__(self, dim=1, not_bin=0):
        super(Cat, self).__init__()
        self.dim = dim
        self.not_bin = not_bin
    def forward(self, x):
        return torch.cat(x, self.dim)


class Sum(nn.Module):
    def __init__(self, dim=1, not_bin=0):
        super(Sum, self).__init__()
        self.dim = dim
        self.not_bin = not_bin
    def forward(self, x):
        return sum(x)