import torch.nn as nn


class FpConvBnHardtanh(nn.Module):
    def __init__(self, in_channels, out_channels,bn_layer=True, activation='htanh', affine=True,kernel_size=1, padding=0, dilation=1, padding_mode='zeros', stride=1):
        super(FpConvBnHardtanh, self).__init__()
        if dilation == 1:
            padding = kernel_size//2
        else:
            padding = dilation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bn_layer = bn_layer
        self.activation = activation
        self.affine = affine 
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode, dilation=dilation,bias=False))
        if activation == 'relu':
            self.layers.add_module('Activation',
                nn.ReLU()
            )
        if bn_layer:
            self.layers.add_module('BatchNorm',
                nn.BatchNorm2d(out_channels, affine=affine)
            )
        if activation=='htanh':
            self.layers.add_module('Activation',
                nn.Hardtanh(min_val=-1, max_val=1, inplace=True)
            )
    def forward(self, x):
        x = self.layers(x)
        return x
    
    def get_config(self):
        return  {'in_channels':self.in_channels, 'out_channels':self.out_channels,'bn_layer':self.bn_layer, 'activation':self.activation, 'affine':self.affine,'kernel_size':self.layers[0].kernel_size[0], 'padding':self.layers[0].padding[0], 'dilation':self.layers[0].dilation[0], 'padding_mode':self.layers[0].padding_mode, 'stride':self.layers[0].stride[0]}
    
    @classmethod
    def model(cls, config):
        return cls(**config)

class FpTConvBnHardtanh(nn.Module):
    def __init__(self, in_channels, out_channels,bn_layer=True, activation='htanh', affine=True,kernel_size=1, padding=0, dilation=1, padding_mode='zeros', stride=1, output_padding=1):
        super(FpTConvBnHardtanh, self).__init__()
        if dilation ==1:
            padding = kernel_size//2
            output_padding = 1 if stride > 1 else 0
        elif dilation > 1:
            if stride >1:
                output_padding = 1
            else:
                output_padding = 0
        self.in_channels = in_channels
        self.out_channels= out_channels
        self.bn_layer = bn_layer
        self.activation = activation
        self.affine = affine

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode, bias=False, dilation=dilation, output_padding=output_padding))
        
        if activation == 'relu':
            self.layers.add_module('Activation',
                nn.ReLU()
            )

        if bn_layer:
            self.layers.add_module('BatchNorm',
                nn.BatchNorm2d(out_channels, affine=affine)
            )
        if activation=='htanh':
            self.layers.add_module('Activation',
                nn.Hardtanh(min_val=-1, max_val=1, inplace=True)
            )
    def forward(self, x):
        x = self.layers(x)
        return x

    def get_config(self):
        return {'in_channels':self.in_channels, 'out_channels':self.out_channels,'bn_layer':self.bn_layer, 'activation':self.activation, 'affine':self.affine,'kernel_size':self.layers[0].kernel_size[0], 'padding':self.layers[0].padding[0], 'dilation':self.layers[0].dilation[0], 'padding_mode':self.layers[0].padding_mode, 'stride':self.layers[0].stride[0], 'output_padding':self.layers[0].output_padding[0]}
    
    @classmethod
    def model(cls, config):
        return cls(**config)