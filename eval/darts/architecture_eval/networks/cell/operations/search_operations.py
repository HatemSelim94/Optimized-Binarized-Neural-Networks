import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .latency import get_latency
from .memory import MemoryCounter
from .flops import FlopsCounter
from .binarization.binarized_blocks import BinConvBnHTanh, BinTConvBnHTanh, BinConvBn,ConvBn
from .binarization.binarized_blocks import ConvBnHTanhBin, TConvBnHTanhBin 
from .fp.fp_block import FpConvBnHardtanh, FpTConvBnHardtanh

####################
# Fp ops
class FpConv1x1(FpConvBnHardtanh):
    def __init__(self, in_channels, out_channels, bn_layer=True, activation=True, affine=True, kernel_size=1, padding=0, dilation=1, padding_mode='zeros', stride=1):
        super(FpConv1x1, self).__init__(in_channels, out_channels, bn_layer, activation, affine, kernel_size, padding, dilation, padding_mode, stride)
        


class FpConv3x3(FpConvBnHardtanh):
    def __init__(self, in_channels, out_channels, bn_layer=True, activation=True, affine=True, kernel_size=3, padding=1, dilation=1, padding_mode='zeros', stride=1):
        super(FpConv3x3, self).__init__(in_channels=in_channels, out_channels=out_channels, bn_layer=bn_layer, activation=activation, affine=affine, kernel_size=kernel_size, padding=padding, dilation=dilation, padding_mode=padding_mode, stride=stride)
        


class FpDilConv3x3(FpConvBnHardtanh):
    def __init__(self, in_channels, out_channels, bn_layer=True, activation=True, affine=True, kernel_size=3, padding=0, dilation=1, padding_mode='zeros', stride=1):
        super(FpDilConv3x3, self).__init__(in_channels=in_channels, out_channels=out_channels, bn_layer=bn_layer, activation=activation, affine=affine, kernel_size=kernel_size, padding=padding, dilation=dilation, padding_mode=padding_mode, stride=stride)
        

class FpTConv1x1(FpTConvBnHardtanh):
    def __init__(self, in_channels, out_channels, bn_layer=True, activation=True, affine=True, kernel_size=1, padding=0, dilation=1, padding_mode='zeros', stride=1, output_padding=1):
        super(FpTConv1x1, self).__init__(in_channels, out_channels, bn_layer, activation, affine, kernel_size, padding, dilation, padding_mode, stride, output_padding)
        


class FpTConv3x3(FpTConvBnHardtanh):
    def __init__(self, in_channels, out_channels, bn_layer=True, activation=True, affine=True, kernel_size=3, padding=0, dilation=1, padding_mode='zeros', stride=1, output_padding=1):
        super(FpTConv3x3, self).__init__(in_channels, out_channels, bn_layer, activation, affine, kernel_size, padding, dilation, padding_mode, stride, output_padding)
        


class FpTConv5x5(FpTConvBnHardtanh):
    def __init__(self, in_channels, out_channels, bn_layer=True, activation=True, affine=True, kernel_size=5, padding=0, dilation=1, padding_mode='zeros', stride=1, output_padding=1):
        super(FpTConv5x5, self).__init__(in_channels, out_channels, bn_layer, activation, affine, kernel_size, padding, dilation, padding_mode, stride, output_padding)
        


class FpDilTConv3x3(FpTConvBnHardtanh):
    def __init__(self, in_channels, out_channels, bn_layer=True, activation=True, affine=True, kernel_size=3, padding=0, dilation=1, padding_mode='zeros', stride=1, output_padding=1):
        super(FpDilTConv3x3, self).__init__(in_channels, out_channels, bn_layer, activation, affine, kernel_size, padding, dilation, padding_mode, stride, output_padding)
        


class FpConv3x3bn(FpConvBnHardtanh):
    def __init__(self, in_channels, out_channels, bn_layer=True, activation=False, affine=True, kernel_size=1, padding=0, dilation=1, padding_mode='zeros', stride=1):
        super(FpConv3x3bn, self).__init__(in_channels, out_channels, bn_layer, activation, affine, kernel_size, padding, dilation, padding_mode, stride)
        

####################
# bin ops
class BinConvT1x1(BinTConvBnHTanh):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=2, padding=0, dilation=1, affine=True, padding_mode='zeros', jit=False,dropout2d=0.1):
        super(BinConvT1x1, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, affine, padding_mode, jit,dropout2d)
        


class BinConvT3x3(BinTConvBnHTanh):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, dilation=1, affine=True,padding_mode='zeros',jit=False,dropout2d=0.1):
        super(BinConvT3x3, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, affine, padding_mode, jit,dropout2d)
        


class BinConvT5x5(BinTConvBnHTanh):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=2, padding=1, dilation=1, affine=True, padding_mode='zeros', jit=False,dropout2d=0.1):
        super(BinConvT5x5, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, affine, padding_mode, jit,dropout2d)
        


class BinDilConv3x3(BinConvBnHTanh):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=2, dilation=2,affine=True, padding_mode='zeros',jit=False,dropout2d=0.1):
        super(BinDilConv3x3, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,affine, padding_mode,jit,dropout2d)
        


class BinConv1x1(BinConvBnHTanh):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, affine=True, padding_mode='zeros',jit=False,dropout2d=0.1):
        super(BinConv1x1, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,affine, padding_mode,jit,dropout2d)
        

class BinConvbn1x1(BinConvBn):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, affine=True, padding_mode='zeros', jit=False,dropout2d=0.1):
        super(BinConvbn1x1, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,affine, padding_mode, jit,dropout2d)


class Convbn1x1(ConvBn):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, affine=True, padding_mode='zeros', jit=False,dropout2d=0.1):
        super(Convbn1x1, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,affine, padding_mode, jit,dropout2d)


class BinConv3x3(BinConvBnHTanh):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, affine=True, padding_mode='zeros', jit=False,dropout2d=0.1):
        super(BinConv3x3, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, affine, padding_mode, jit, dropout2d)
        


class EvalIdentity(nn.Module):
    def __init__(self, params=None):
        super(EvalIdentity, self).__init__()
        
    
    def forward(self, x):
        return x
    
    def ops(self, x):
        return 0

    def max_memory(self, x):
        with torch.no_grad():
            max_mem = 2*np.prod(x.shape)
            return max_mem
    
    def params(self, x):
        with torch.no_grad():
            return 0
    
    def latency(self, x):
        return 0
    def get_config(self):
        return {'params':None}

    @classmethod
    def model(cls, config):
        return cls(**config)


class EvalBilinear(nn.Upsample):
    def __init__(self, size= None, scale_factor = 2, mode: str = 'bilinear', align_corners = False):
        super(EvalBilinear, self).__init__(scale_factor=scale_factor, mode=mode, align_corners=align_corners)


class AvgPool(nn.Module):
    def __init__(self, in_channels, affine, stride = None, kernel_size=3, padding= 1):
        super(AvgPool, self).__init__()
        self.latency_table = {}
        padding = kernel_size//2
        self.avgpool = nn.AvgPool2d(kernel_size, stride, padding)
        self.batchnorm = nn.BatchNorm2d(in_channels, affine=affine)
        self.hardtanh= nn.Hardtanh(-1,1,inplace=True)
        self.in_channels = in_channels
        
    
    def forward(self, x):
        x = self.avgpool(x)
        #print(x.shape)
        x = self.batchnorm(x)
        x =self.hardtanh(x)
        return x

    def ops(self, x):
        with torch.no_grad():
            avg_output = self.avgpool(x[0,:,:,:].unsqueeze(0))
            output = self.batchnorm(avg_output)
            bn_flops = FlopsCounter.count('batchnorm',(output.shape, self.batchnorm.affine))
            avg_flops = FlopsCounter.count('avgpool', (avg_output.shape,))
            flops = bn_flops + avg_flops
            ops = flops + 0  
#            return (ops, flops), output
            return ops

    
    def max_memory(self, x):
        with torch.no_grad():
            avg_output = self.avgpool(x[0,:,:,:].unsqueeze(0))
            output = self.batchnorm(avg_output)
            avg_mem = MemoryCounter.count('avgpool',(x.shape, avg_output.shape))
            bn_mem = MemoryCounter.count('batchnorm',(avg_output.shape, output.shape, self.batchnorm.affine))
            max_mem = max(avg_mem, bn_mem)
            #return (max_mem, bn_mem), output
            return max_mem
    
    def latency(self, x):
        if self.latency_table.get(str(x.shape[-3:]), None) is None:
            with torch.no_grad():
                l = get_latency(self.model(self.get_config()), x[0,:,:,:].unsqueeze(0))
                self.latency_table[str(x.shape[-3:])] = l
        else:
            l = self.latency_table.get(str(x.shape[-3:]), x[0,:,:,:].unsqueeze(0).clone().detach())
        #with torch.no_grad():
        #    output = self(x)
        #return l, output
        return l
    
    def params(self, x):
        #with torch.no_grad():
            #output = self(x)
        #return (0,0), output
        return 0
    
    def get_config(self):
        return {'in_channels':self.in_channels, 'kernel_size':self.avgpool.kernel_size, 'stride':self.avgpool.stride,'padding':self.avgpool.padding, 'affine':self.batchnorm.affine}
    
    @classmethod
    def model(cls, config):
        return cls(**config)


class MaxPool(nn.Module):
    def __init__(self, in_channels, affine, kernel_size=3, stride = None, padding= 1, dilation=1):
        super(MaxPool, self).__init__()
        self.latency_table = {}
        padding = kernel_size//2
        self.maxpool = nn.MaxPool2d(kernel_size, stride, padding,dilation)
        self.batchnorm = nn.BatchNorm2d(in_channels, affine=affine)
        self.in_channels = in_channels
        self.hardtanh= nn.Hardtanh(-1,1,inplace=True)
        
    
    def forward(self, x):
        #print(x.shape)
        x = self.maxpool(x)
        #print(x.shape)
        x = self.batchnorm(x)
        x =self.hardtanh(x)
        return x

    def ops(self, x):
        with torch.no_grad():
            avg_output = self.maxpool(x[0,:,:,:].unsqueeze(0))
            output = self.batchnorm(avg_output)
            bn_flops = FlopsCounter.count('batchnorm',(output.shape, self.batchnorm.affine))
            avg_flops = FlopsCounter.count('maxpool', (avg_output.shape,))
            flops = bn_flops + avg_flops
            ops = flops + 0  
            #return (ops, flops), output
            return ops

    
    def max_memory(self, x):
        with torch.no_grad():
            avg_output = self.maxpool(x[0,:,:,:].unsqueeze(0))
            output = self.batchnorm(avg_output)
            avg_mem = MemoryCounter.count('maxpool',(x[0,:,:,:].shape, avg_output.shape))
            bn_mem = MemoryCounter.count('batchnorm',(avg_output.shape, output.shape, self.batchnorm.affine))
            max_mem = max(avg_mem, bn_mem)
            #return (max_mem, bn_mem), output
            return max_mem
    
    def latency(self, x):
        if self.latency_table.get(str(x.shape[-3:]), None) is None:
            with torch.no_grad():
                l = get_latency(self.model(self.get_config()), x[0,:,:,:].unsqueeze(0))
                self.latency_table[str(x.shape[-3:])] = l
        else:
            l = self.latency_table.get(str(x.shape[-3:]), x[0,:,:,:].unsqueeze(0).clone().detach())
        #with torch.no_grad():
        #    output = self(x)
        #return l, output
        return l
    
    def params(self, x):
        #with torch.no_grad():
        #    output = self(x)
        #return (0,0), output
        return 0
    
    def get_config(self):
        return {'in_channels':self.in_channels, 'kernel_size':self.maxpool.kernel_size, 'stride':self.maxpool.stride,'padding':self.maxpool.padding, 'affine':self.batchnorm.affine}
    
    @classmethod
    def model(cls, config):
        return cls(**config)


class EvalCat(nn.Module):
    def __init__(self, dim=1, not_bin=0):
        super(EvalCat, self).__init__()
        self.dim = dim
        self.not_bin = not_bin
    def forward(self, x):
        return torch.cat(x, self.dim)
    
    def ops(self, x):
        #with torch.no_grad():
        #    output = self(x)
        #flops = 0 
        ops = 0
        #return (ops, flops), output
        return ops
    
    def max_memory(self, x):
        #with torch.no_grad():
            #output = self(x)
        input_mem = np.prod(x[0].shape)*(len(x)- self.not_bin)*(32/8) + np.prod(x[0].shape)*(self.not_bin) * (1/8)
        output_mem = np.prod(x[0].shape)*(len(x))*(32/8)
        params_size = 0
        cat_mem = input_mem + output_mem + params_size
        #return cat_mem, output_mem, output
        return cat_mem
    
    def params(self, x):
        #with torch.no_grad():
        #    output = self(x)
        #return (0,0), output
        return 0


class EvalSum(nn.Module):
    def __init__(self, dim=1, not_bin=0):
        super(EvalSum, self).__init__()
        self.dim = dim
        self.not_bin = not_bin
    def forward(self, x):
        return sum(x)
    
    def ops(self, x):
        #with torch.no_grad():
        #    output = self(x)
        flops = np.prod(x[0].shape)*(len(x)- self.not_bin)*(32/8)  
        bops = np.prod(x[0].shape)*(self.not_bin) * (1/8)
        ops = flops + bops
        #return (ops, flops), output
        return ops
    
    def max_memory(self, x):
        #with torch.no_grad():
        #    output = self(x)
        input_mem = np.prod(x[0].shape)*(len(x)- self.not_bin)*(32/8) + np.prod(x[0].shape)*(self.not_bin) * (1/8)
        output_mem = np.prod(x[0].shape)*(len(x))*(32/8)
        params_size = 0
        sum_mem = input_mem + output_mem + params_size
        #return cat_mem, output_mem, output
        return sum_mem
    
    def params(self, x):
        #with torch.no_grad():
        #    output = self(x)
        #return (0,0), output
        return 0
