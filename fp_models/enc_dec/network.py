import imp
from multiprocessing.spawn import import_main_path
import torch
import torch.nn as nn
from collections import OrderedDict

class ConvBnActivation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,bn=True,activation='relu', stride=2, layer_num=''):
        super(ConvBnActivation, self).__init__()
        self.activations = {'relu':nn.ReLU, 'hardtanh':nn.Hardtanh}
        self.layers = nn.Sequential()
        self.layers.add_module(f'Conv_{layer_num}',nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, bias=False, padding=kernel_size//2))
        if bn:
            self.layers.add_module(f'BatchNorm_{layer_num}', nn.BatchNorm2d(out_channels))
        if activation is not None:
            self.layers.add_module(f'Activation_{layer_num}',self.activations[activation](True))
    def forward(self, x):
        return self.layers(x)


class TConvBnActivation(nn.Module):
    def __init__(self,in_channels, out_channels,kernel_size,bn=True,activation='relu', stride=2, layer_num=''):
        super(TConvBnActivation, self).__init__()
        self.activations = {'relu':nn.ReLU, 'hardtanh':nn.Hardtanh}
        self.layers = nn.Sequential()
        self.layers.add_module(f'TConv_{layer_num}',nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding= kernel_size//2, output_padding=1, bias=False))
        if bn:
            self.layers.add_module(f'BatchNorm_{layer_num}', nn.BatchNorm2d(out_channels))
        if activation is not None:
            self.layers.add_module(f'Activation_{layer_num}',self.activations[activation](True))
    def forward(self, x):
        return self.layers(x)

class Encoder(nn.Module):
    def __init__(self, layers_num, in_channels, kernel_size=3,activation='relu'):
        super(Encoder, self).__init__()
        self.layers_num = layers_num
        self.layers = nn.Sequential()
        in_c = in_channels
        for i in range(layers_num):
            out_c = in_c*2
            ConvBnActivation(in_channels=in_c, out_channels=out_c,kernel_size=kernel_size,bn=True,activation=activation, stride=2,layer_num=i)
            in_c = out_c
        self.out_channels = out_c
    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self, layers_num, in_channels, kernel_size=3,activation='relu'):
        super(Decoder, self).__init__()
        self.layers_num = layers_num
        self.layers = nn.Sequential()
        in_c = in_channels
        for i in range(layers_num):
            out_c = in_c//2
            TConvBnActivation(in_channels=in_c, out_channels=out_c,kernel_size=kernel_size,bn=True,activation=activation, stride=2, layer_num=i)
            in_c = out_c
        self.out_channels = out_c
    def forward(self, x):
        return self.layers(x)


class Network(nn.Module):
    def __init__(self,args):
        super(Network, self).__init__()
        self.first_layer = ConvBnActivation(3, args.stem_channels, args.kernel_size)
        self.encoder = Encoder(args.encoder_layers, in_channels=args.stem_channels,kernel_size=args.kernel_size, activation=args.activation)
        self.decoder = Decoder(args.decoder_layers, self.encoder.out_channels,kernel_size=args.kernel_size, activation=args.activation)
        self.final_layer = ConvBnActivation(self.decoder.out_channels, args.num_of_classes, 1, activation=None, stride=1,layer_num='final_layer')
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    def forward(self, x):
        shape = x.shape
        x = self.first_layer(x)
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.final_layer(x)
        if x.shape != shape:
            return self.upsample(x)
        return x