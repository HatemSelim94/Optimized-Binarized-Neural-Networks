import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
import math

class QuantizationBase(Function):
    @staticmethod
    def forward(ctx, input, quantization_class):
        output = input.clone().detach()
        ctx.save_for_backward(output)
        ctx.quantization_class = quantization_class
        output = quantization_class.forward_function(output)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        input, = ctx.saved_tensors
        grad_input = ctx.quantization_class.backward_function(input, grad_input)
        return grad_input, None
    
class Quantization:
    def __init__(self, forward_function, backward_function):
        self.forward_function = forward_function
        self.backward_function = backward_function

quantize = QuantizationBase.apply

class BnnActivation(nn.Module):
    def __init__(self, quantization_class):
        super(BnnActivation, self).__init__()
        self.quantization_class = quantization_class
    def forward(self, input):
        output = quantize(input, self.quantization_class)
        return output

class BnnConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                stride=1, padding=1, dilation=1, groups=1, bias=None,
                padding_mode='zeros', quantization_class=None, threshhold=1, step=None):
        super(BnnConv2d, self).__init__()
        self._set_param(in_channels, out_channels, kernel_size, 
                stride, padding, dilation, groups, bias,
                padding_mode)
        weights = torch.empty((out_channels,in_channels//groups, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(weights, a=math.sqrt(5))
        self.conv_weights = nn.Parameter(weights, requires_grad=True)
        self.conv_weights.bin = True 
        self.quantization_class = quantization_class
        self.threshold = threshhold
        self.step = step
    
    def forward(self, input):
        if self.step == 1:
            output = F.conv2d(input, self.conv_weights, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            #if self.threshold:
                #if self.training:
                    #self.conv_weights.data.clamp_(-self.threshold, self.threshold)
            binarized_weights = quantize(self.conv_weights, self.quantization_class)
            output = F.conv2d(input, binarized_weights, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return output
    
    def _set_param(self, in_channels, out_channels, kernel_size, 
                stride, padding, dilation, groups, bias,
                padding_mode): # could be replaced with arg kwargs and/or an inhertance from nn.conv2d but I find it better
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode


class BnnTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding=0, groups=1, bias=None, dilation=1, padding_mode='zeros', quantization_class=None, threshhold=1, step=None):
        super(BnnTConv2d, self).__init__()
        self._set_param(in_channels, out_channels, kernel_size, 
                stride, padding, dilation, groups, bias,
                padding_mode, output_padding)
        weights = torch.empty((in_channels,out_channels//groups, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(weights, a=math.sqrt(5))
        self.conv_weights = nn.Parameter(weights, requires_grad=True)
        self.conv_weights.bin = True 
        self.quantization_class = quantization_class
        self.threshold = threshhold
        self.step = step
    
    def forward(self, input):
        if self.step == 1:
            output = F.conv_transpose2d(input, self.conv_weights,self.bias, self.stride, self.padding,output_padding=self.output_padding,dilation=self.dilation, groups =self.groups)
        else:
            #if self.threshold:
            #    if self.training:
            #        self.conv_weights.data.clamp_(-self.threshold, self.threshold)
            binarized_weights = quantize(self.conv_weights, self.quantization_class)
            output = F.conv_transpose2d(input, binarized_weights, self.bias, self.stride, self.padding,output_padding=self.output_padding,dilation=self.dilation, groups =self.groups)
        return output
    
    def _set_param(self, in_channels, out_channels, kernel_size, 
                stride, padding, dilation, groups, bias,
                padding_mode, output_padding): # could be replaced with arg kwargs and/or an inhertance from nn.conv2d but I find it better
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode
        self.output_padding = output_padding


def bnn_hubara_forward(input):
    return torch.ones_like(input).masked_fill_(input.lt(0),-1)

def bnn_hubara_backward(input, grad_input):
    mask = torch.abs(input).gt(1)
    grad_input[mask] = 0
    return grad_input

bnn_hubara_binarization = Quantization(bnn_hubara_forward, bnn_hubara_backward)


def melius_forward(input):
    return input.sign()

def melius_backward(input, grad_input):
    mask = torch.abs(input).gt(1.3)
    grad_input[mask] = 0
    return grad_input

melius_binarization = Quantization(melius_forward, melius_backward)


def structured_bnn_activation_forward(input):
    return input.sign()

def structured_bnn_activation_backward(input, grad_input):
    mask0 = input < -1 
    mask1 = input >=-1 and input<0
    mask2 = input < 1 and input >=0
    mask3 = input >= 1
    
    grad_input[mask0] = 0
    grad_input[mask1] = 2 + 2 * grad_input[mask1]
    grad_input[mask2] = 2 - 2 * grad_input[mask2]
    grad_input[mask3] = 0
    return grad_input

structured_bnn_activation = Quantization(structured_bnn_activation_forward, structured_bnn_activation_backward)


#######################################################################################

class XnorActivationConv2d(nn.Module): # Xnor Rastegari 2016  BinActivatoin merged with BinConv
    def __init__(self, in_channels, out_channels, kernel_size, 
                stride=1, padding=0, dilation=1, groups=1, bias=False,
                padding_mode='zeros'):
        self.input_sign_function = BnnActivation()
        self.weight_sign_function = BnnActivation()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                stride, padding, dilation, groups, bias,
                padding_mode)

    def forward(self, input):
        _, _ , kh, kw = self.weight.shape
        A = torch.mean(torch.abs(input), dim=1, keepdim=True)
        k = torch.ones([1,1,kh,kw])*(1/(kh*kw))
        K = F.conv2d(A,k, bias=False, padding=(kh//2, kw//2))
        alpha = torch.mean(torch.abs(self.weight))
        K_alpha = K*alpha
        #sign function(sometimes the sign() function is used 
        # since a float variable is almost impossible to be 0 unless 
        # if it is the first iteration and it was initialized with zeros)
        input_sign = self.input_sign_function(input)
        weight_Sign = self.weight_sign_function(self.weight)
        output = self.conv(input_sign, weight_Sign,).mul(K_alpha)
        return output 


class XnorPlusConv2d(nn.Module): # XNOR-Net++: Improved binary neural networks
    def __init__(self, in_channels, out_channels, kernel_size, 
                stride=1, padding=0, dilation=1, groups=1, bias=None,
                padding_mode='zeros', input_dim= (220,220)):
        super(XnorPlusConv2d, self).__init__()
        self._set_param(in_channels, out_channels, kernel_size, 
                stride, padding, dilation, groups, bias,
                padding_mode, input_dim)
        out_h, out_w = self.get_conv_out()
        self.conv_weights = nn.Parameter(torch.normal(0.0,1.0,size=(out_channels,in_channels//groups, kernel_size, kernel_size)), requires_grad=True)
        self.alpha = nn.Parameter(torch.rand([out_channels]), requires_grad=True)
        self.beta = nn.Parameter(torch.rand([out_h]), requires_grad=True)
        self.gamma = nn.Parameter(torch.rand([out_w]), requires_grad=True)
        self.scalling_factor = nn.Parameter(torch.randn((out_channels, out_h, out_w)), requires_grad=False)
    
    def forward(self, input):
        if self.training:
            binarized_weights = quantize(self.conv_weights, bnn_hubara_binarization)
            out = F.conv2d(input, binarized_weights, self.bias, self.stride, self.padding, self.dilation, self.groups)
            out = out* self.alpha[None,:,None, None]
            out = out* self.beta[None,None,:, None]
            out = out* self.gamma[None, None, None,:]
            self.scalling_factor.copy_(torch.einsum('i,j,k->ijk', self.alpha, self.beta, self.gamma).detach().clone())
            return out
        else:
            # testing with one image per time
            binarized_weights = quantize(self.conv_weights, bnn_hubara_binarization)
            out = F.conv2d(input, binarized_weights, self.bias, self.stride, self.padding, self.dilation, self.groups)
            out = out * self.scalling_factor.unsqueeze(0)
            return out
    
    def get_conv_out(self):
        out_h = (self.input_dim[0] + 2*self.padding - self.kernel_size - (self.kernel_size-1)*(self.dilation-1))//self.stride +1 
        out_w = (self.input_dim[1] + 2*self.padding - self.kernel_size - (self.kernel_size-1)*(self.dilation-1))//self.stride +1 
        return out_h, out_w
    
    def _set_param(self, in_channels, out_channels, kernel_size, 
                stride, padding, dilation, groups, bias,
                padding_mode, input_dim): # could be replaced with arg kwargs and/or an inhertance from nn.conv2d but I find it better
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode
        self.input_dim = input_dim


class XnorPlusTConv2d(nn.Module): # XNOR-Net++: Improved binary neural networks
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                stride=2, padding=1, dilation=1, groups=1, bias=None,
                output_padding=1, input_dim= (220,220)):
        super(XnorPlusTConv2d, self).__init__()
        self._set_param(in_channels, out_channels, kernel_size, 
                stride, padding, dilation, groups, bias,
                output_padding, input_dim)
        out_h, out_w = self.get_tconv_out()
        self.tconv_weights = nn.Parameter(torch.normal(0.0,1.0,size=(in_channels,out_channels//groups, kernel_size, kernel_size)), requires_grad=True)
        self.alpha = nn.Parameter(torch.rand([out_channels]), requires_grad=True)
        self.beta = nn.Parameter(torch.rand([out_h]), requires_grad=True)
        self.gamma = nn.Parameter(torch.rand([out_w]), requires_grad=True)
        self.scalling_factor = nn.Parameter(torch.randn((out_channels, out_h, out_w)), requires_grad=False)
        self.conv = 1
    
    def forward(self, input):
        if self.training:
            binarized_weights =quantize(self.tconv_weights, bnn_hubara_binarization)
            out = F.conv_transpose2d(input, binarized_weights, self.bias, self.stride, self.padding, self.output_padding,self.groups, self.dilation)
            print(self.beta.shape)
            out = out* self.alpha[None,:,None, None]
            out = out* self.beta[None,None,:, None]
            out = out* self.gamma[None, None, None,:]
            self.scalling_factor.copy_(torch.einsum('i,j,k->ijk', self.alpha, self.beta, self.gamma).detach().clone())
            return out
        else:
            # testing with one image per time
            binarized_weights =quantize(self.tconv_weights, bnn_hubara_binarization)
            out = F.conv_transpose2d(input, binarized_weights, self.bias, self.stride, self.padding, self.output_padding,self.groups, self.dilation)
            out = out * self.scalling_factor.unsqueeze(0)
            return out
    
    def get_tconv_out(self):
        out_h = (self.input_dim[0] -1)*self.stride - 2*self.padding + self.dilation*(self.kernel_size-1) + self.output_padding+1
        out_w = (self.input_dim[1] -1)*self.stride - 2*self.padding + self.dilation*(self.kernel_size-1) + self.output_padding+1
        return out_h, out_w
    
    def _set_param(self, in_channels, out_channels, kernel_size, 
                stride, padding, dilation, groups, bias,
                output_padding, input_dim): # could be replaced with arg kwargs and/or an inhertance from nn.conv2d but I find it better
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.output_padding = output_padding
        self.input_dim = input_dim






if __name__ == '__main__':
    input = torch.randn([1,3,320,320])
    conv_layer = BnnConv2d(3,60, 3,padding=1, stride=2,quantization_class=bnn_hubara_binarization)
    out = conv_layer(input)
    print(out.shape)
    tconv_layer = BnnTConv2d(3,60, 3,padding=1, output_padding=1, stride=2, quantization_class=bnn_hubara_binarization) 
    out = tconv_layer(input)
    print(out.shape)