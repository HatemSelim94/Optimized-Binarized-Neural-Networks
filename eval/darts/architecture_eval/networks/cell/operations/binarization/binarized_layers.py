import torch.nn as nn
import torch.nn.functional as F
from .basic import Binarization1 as Binarization
from .basic import Binarization2 

import collections
from itertools import repeat


binarize = Binarization.apply
binarize2= Binarization2.apply
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


_pair = _ntuple(2)

class EvalBinActivation(nn.Module):
    def __init__(self, jit=False, binarization=1):
        self.jit = jit
        self.binarization = binarization
        self.basic_layer=True
        self.binarized_input = True
        super(EvalBinActivation, self).__init__()    
    def forward(self, x):
        if self.jit:
            output = x.sign()
            output[x==0]=1
            return output
        if self.binarized_input:
            if self.binarization ==1:
                return binarize(x)
            elif self.binarization ==2:
                return binarize2(x)
        else:
            return x


class EvalBinConv2d(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride = 1, padding = 0, dilation = 1, groups: int = 1, bias: bool = False, padding_mode: str = 'zeros', device=None, dtype=None, jit=False):
        self.jit=jit
        self.basic_layer=True
        self.binarized_weight = True
        if dilation ==1:
            padding = [int(kernel_size//2), int(kernel_size//2)]
        else:
            padding = [int(padding), int(padding)]
        super(EvalBinConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        self.weight.bin = True
    
    def forward(self, x):
        if self.jit:
            binary_weights = self.weight.sign()
            binary_weights[self.weight==0] = 1
        else:
            if self.binarized_weight:
                binary_weights = binarize(self.weight)
            else:
                binary_weights = self.weight
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            binary_weights, stride= self.stride,
                            padding=_pair(0), dilation=self.dilation, groups=self.groups)
        return F.conv2d(x, binary_weights,stride= self.stride,
                        padding=self.padding, dilation=self.dilation,groups =self.groups, bias=None)

        #if self.padding_mode != 'zeros':
        #    return F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding, dilation=self.dilation)
        #else:
        #    return F.conv2d(F.pad(x, self._reversed_padding_repeated_twice(),self.padding_mode), binary_weights, stride=self.stride, padding=self.padding, dilation=self.dilation)

class EvalBinConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride = 1, padding= 0, output_padding = 0, groups: int = 1, bias: bool = True, dilation: int = 1, padding_mode: str = 'zeros', device=None, dtype=None, jit=False):
        super(EvalBinConvTranspose2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation, 'zeros', device, dtype)
        self.jit = jit
        self.basic_layer=True
        self.binarized_weight = True
        if dilation ==1:
            self.padding = [self.kernel_size[0]//2, self.kernel_size[1]//2]
            self.output_padding = [1,1] if stride > 1 else [0,0]
        elif dilation > 1:
            if stride >1:
                self.output_padding = [1,1]
            else:
                self.output_padding = [0,0]
        
        self.weight.bin = True
    
    def forward(self, x):
        if self.jit:
            binary_weights = self.weight.sign()
            binary_weights[self.weight==0] = 1
        else:
            if self.binarized_weight:
                binary_weights = binarize(self.weight)
            else:
                binary_weights = self.weight
        return F.conv_transpose2d(x, binary_weights, stride=self.stride, dilation=self.dilation, padding=self.padding, output_padding=self.output_padding)

