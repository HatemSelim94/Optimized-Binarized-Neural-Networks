import torch
import torch.nn as nn
import torch.nn.functional as F

class Binarization1(torch.autograd.Function): # Courbariaux, Hubara
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = torch.ones_like(input)
        output[input <= 0] = -1
        return output
    @staticmethod
    def backward(ctx, grad_output):
        input, =ctx.saved_tensors
        grad_input = None
        #return grad_input, None  # gradients of input and quantization(none) in forward function
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.clone()
            grad_input[torch.abs(input)>1.001] = 0
        return grad_input
binarize = Binarization1.apply


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=(1, 1), groups=1, bias=False):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.conv.weight.bin = True

    def forward(self, input):
        #output = self.conv(input)
        binarized_weights = binarize(self.conv.weight)
        output = F.conv2d(binarize(input), binarized_weights, stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation, groups=self.conv.groups) 

        return output

class TConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, dilation=(1, 1), groups=1, bias=False):
        super().__init__()

        self.tconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding,output_padding=output_padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.tconv.weight.bin = True

    def forward(self, input):
        #output = self.conv(input)
        binarized_weights = binarize(self.tconv.weight)
        output = F.conv_transpose2d(binarize(input), binarized_weights, stride=self.tconv.stride, padding=self.tconv.padding, output_padding=self.tconv.output_padding, dilation=self.tconv.dilation, groups=self.tconv.groups) 

        return output

class Scale_2d(nn.Module):
    def __init__(self, out_ch,init_val=0.001):
        super(Scale_2d, self).__init__()
        self.par = nn.Parameter(torch.randn((out_ch)))
        self.affine = None
        nn.init.constant_(self.par, init_val)
    def forward(self,x):
        return x*self.par[None,:,None,None]