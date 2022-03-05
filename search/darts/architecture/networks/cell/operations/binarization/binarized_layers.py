import torch.nn as nn
import torch.nn.functional as F
from .basic import Binarization1 as Binarizatoin



binarize = Binarizatoin.apply


class BinActivation(nn.Module):
    def __init__(self):
        super(BinActivation, self).__init__()    
    def forward(self, x):
        return binarize(x)


class BinConv2d(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride = 1, padding = 0, dilation = 1, groups: int = 1, bias: bool = False, padding_mode: str = 'zeros', device=None, dtype=None):
        super(BinConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        self.padding = [self.kernel_size[0]//2, self.kernel_size[1]//2]
        self.weight.bin = True
    
    def forward(self, x):
        binary_weights = binarize(self.weight)
        return F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding, dilation=self.dilation)

class BinConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride = 1, padding= 0, output_padding = 0, groups: int = 1, bias: bool = True, dilation: int = 1, padding_mode: str = 'zeros', device=None, dtype=None) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation, padding_mode, device, dtype)
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
        binary_weights = binarize(self.weight)
        return F.conv_transpose2d(x, binary_weights, stride=self.stride, dilation=self.dilation, padding=self.padding, output_padding=self.output_padding)

