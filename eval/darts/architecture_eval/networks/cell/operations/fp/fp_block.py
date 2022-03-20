import torch.nn as nn


class FpConvBnHardtanh(nn.Module):
    def __init__(self, in_channels, out_channels,bn_layer=True, activation=True, affine=True,kernel_size=1, padding=0, dilation=1, padding_mode='zeros', stride=1):
        super(FpConvBnHardtanh, self).__init__()
        if dilation == 1:
            padding = kernel_size//2
        else:
            padding = dilation
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode, dilation=dilation,bias=False))
        if bn_layer:
            self.layers.add_module('BatchNorm',
                nn.BatchNorm2d(out_channels, affine=affine)
            )
        if activation:
            self.layers.add_module('Activation',
                nn.Hardtanh(min_val=-1, max_val=1, inplace=True)
            )
    def forward(self, x):
        x = self.layers(x)
        return x


class FpTConvBnHardtanh(nn.Module):
    def __init__(self, in_channels, out_channels,bn_layer=True, activation=True, affine=True,kernel_size=1, padding=0, dilation=1, padding_mode='zeros', stride=1, output_padding=1):
        super(FpTConvBnHardtanh, self).__init__()
        if dilation ==1:
            padding = kernel_size//2
            output_padding = 1 if stride > 1 else 0
        elif dilation > 1:
            if stride >1:
                output_padding = 1
            else:
                output_padding = 0
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode, bias=False, dilation=dilation, output_padding=output_padding))
        if bn_layer:
            self.layers.add_module('BatchNorm',
                nn.BatchNorm2d(out_channels, affine=affine)
            )
        if activation:
            self.layers.add_module('Activation',
                nn.Hardtanh(min_val=-1, max_val=1, inplace=True)
            )
    def forward(self, x):
        x = self.layers(x)
        return x
