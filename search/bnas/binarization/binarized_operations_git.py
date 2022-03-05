import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
import math

from .binarization_git import BnnActivation, BnnConv2d, BnnTConv2d, bnn_hubara_binarization


Normal_Reduction_OPS = {
  'none' : lambda C, stride, affine: Zero(stride),
  'avg_pool_3x3' : lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
  'max_pool_3x3' : lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
  'skip_connect' : lambda C, stride, affine: Identity() if stride == 1 else BnnConv(C, C, 1, stride, affine=affine),
  'conv_3x3': lambda C, stride, affine: BnnConv(C, C, 3, stride, affine=affine),
  'conv_1x1': lambda C, stride, affine: BnnConv(C, C, 1, stride, affine=affine),
  'dil_conv_3x3' : lambda C, stride, affine: BnnDilConv(C, C, 3, stride, 2, 2, affine=affine),
  'dil_conv_5x5' : lambda C, stride, affine: BnnDilConv(C, C, 5, stride, 4, 2, affine=affine)
}

Upsampling_OPS = {
  'none' : lambda C, stride, affine : Zero(stride, True),
  'skip_connect' : lambda C, stride, affine: Identity() if stride == 1 else BnnTConv(C, C, 1, stride, affine=affine),
  'bilinear_upsample' : lambda  C, stride, affine:  nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=False),
  't_conv_3x3': lambda C, stride, affine: BnnTConv(C, C, 3, stride, affine=affine),
  't_conv_5x5': lambda C, stride, affine: BnnTConv(C, C, 5, stride, affine=affine),
  'dil_t_conv_3x3_r4': lambda C, stride, affine: BnnDilTConv(C, C, 3, stride, 4,4, affine=affine),
  'dil_t_conv_3x3_r6': lambda C, stride, affine: BnnDilTConv(C, C, 3, stride, 6,6, affine=affine),
  'dil_t_conv_3x3_r8': lambda C, stride, affine: BnnDilTConv(C, C, 3, stride, 8, 8, affine=affine),
}


FP_Normal_Reduction_OPS = {
  'none' : lambda C, stride, affine: Zero(stride),
  'avg_pool_3x3' : lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
  'max_pool_3x3' : lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
  'skip_connect' : lambda C, stride, affine: Identity() if stride == 1 else BnnConv(C, C, 1, stride, affine=affine),
  'conv_3x3': lambda C, stride, affine: ConvBnReLU(C, C, 3, stride, affine=affine),
  'conv_1x1': lambda C, stride, affine: ConvBnReLU(C, C, 1, stride,affine=affine),
  'dil_conv_3x3' : lambda C, stride, affine: ConvBnReLU(C, C, 3, stride, 2, 2, affine=affine),
  'dil_conv_5x5' : lambda C, stride, affine: ConvBnReLU(C, C, 5, stride, 4, 2, affine=affine)
}

FP_Upsampling_OPS = {
  'none' : lambda C, stride, affine : Zero(stride, True),
  'skip_connect' : lambda C, stride, affine: Identity() if stride == 1 else BnnTConv(C, C, 1, stride, affine=affine),
  'bilinear_upsample' : lambda  C, stride, affine:  nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=False),
  't_conv_3x3': lambda C, stride, affine: TConvBnReLU(C, C, 3, stride, affine=affine),
  't_conv_5x5': lambda C, stride, affine: TConvBnReLU(C, C, 5, stride, affine=affine),
  'dil_t_conv_3x3_r4': lambda C, stride, affine: DilTConvBnReLU(C, C, 3, stride, 4,4, affine=affine),
  'dil_t_conv_3x3_r6': lambda C, stride, affine: DilTConvBnReLU(C, C, 3, stride, 6,6, affine=affine),
  'dil_t_conv_3x3_r8': lambda C, stride, affine: DilTConvBnReLU(C, C, 3, stride, 8, 8, affine=affine),
}

class SingleConv2d(nn.Module):
  def __init__(self, C_in, C_out, kernel_size, stride, padding=0 ,step=None):
      super(BnnConv, self).__init__()
      padding = kernel_size//2
      self.op = BnnConv2d(C_in, C_out, kernel_size, stride, padding, quantization_class=bnn_hubara_binarization, step=step),
  def forward(self, x):
    return self.op(x)


class BnnConv(nn.Module):
  def __init__(self, C_in, C_out, kernel_size, stride, padding=0 ,step=None,affine=True):
      super(BnnConv, self).__init__()
      padding = kernel_size//2
      self.op = nn.Sequential(
        BnnActivation(bnn_hubara_binarization),
        BnnConv2d(C_in, C_out, kernel_size, stride, padding, quantization_class=bnn_hubara_binarization, step=step),
        nn.BatchNorm2d(C_out, affine=affine),
        nn.Tanh()
      )
  def forward(self, x):
    return self.op(x)


class BnnTConv(nn.Module):
  def __init__(self, C_in, C_out, kernel_size, stride, padding=1,dilation=1,step=None,affine=True):
      super(BnnTConv, self).__init__()
      padding = (kernel_size//2) 
      output_padding = 1 if stride > 1 else 0
      self.op = nn.Sequential(
        BnnActivation(bnn_hubara_binarization),
        BnnTConv2d(C_in, C_out, kernel_size, stride, padding, dilation=dilation,output_padding=output_padding,quantization_class=bnn_hubara_binarization, step=step),
        nn.BatchNorm2d(C_out, affine=affine),
        nn.Tanh()
      )
  def forward(self, x):
    return self.op(x)


class ConvBnReLU(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding=0, dilation=1,affine=True):
    super(ConvBnReLU, self).__init__()
    self.op = nn.Sequential(
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      #nn.ReLU(inplace=False)
      #nn.SELU(inplace=False)
      nn.SiLU(inplace=False)
    )

  def forward(self, x):
    return self.op(x)

class TConvBnReLU(nn.Module):
  def __init__(self, C_in, C_out, kernel_size, stride, padding=1,dilation=1, bias = False, affine=True):
    super(TConvBnReLU, self).__init__()
    output_padding = 1 if stride > 1 else 0
    padding = (kernel_size//2) 
    self.op = nn.Sequential(
      nn.ConvTranspose2d(C_in, C_out, kernel_size, stride=stride,padding=padding, dilation=dilation, output_padding=output_padding,bias=bias),
      nn.BatchNorm2d(C_out, affine=affine),
      #nn.ReLU(inplace=False)
      #nn.SELU(inplace=False)
      nn.SiLU(inplace=False)
    )

  def forward(self, x):
    return self.op(x)


class DilTConvBnReLU(nn.Module):
  def __init__(self, C_in, C_out, kernel_size, stride, padding,dilation=1, bias = False, affine=True):
    super(DilTConvBnReLU, self).__init__()
    output_padding = 1 if stride > 1 else 0
    self.op = nn.Sequential(
      nn.ConvTranspose2d(C_in, C_out, kernel_size, stride=stride,padding=padding, dilation=dilation, output_padding=output_padding,bias=bias),
      nn.BatchNorm2d(C_out, affine=affine),
      #nn.ReLU(inplace=False)
      nn.SELU(inplace=False)
    )

  def forward(self, x):
    return self.op(x)

class BnnDilConv(nn.Module):
  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation ,step=None,affine=True):
      super(BnnDilConv, self).__init__()
      self.op = nn.Sequential(
        BnnActivation(bnn_hubara_binarization),
        BnnConv2d(C_in, C_out, kernel_size, stride, padding,dilation,quantization_class=bnn_hubara_binarization, step=step),
        nn.BatchNorm2d(C_out, affine=affine),
        nn.Tanh()
      )
  def forward(self, x):
    return self.op(x)


class BnnDilTConv(nn.Module):
  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation ,step=None,affine=True):
      super(BnnDilTConv, self).__init__()
      output_padding = 1 if stride > 1 else 0
      self.op = nn.Sequential(
        BnnActivation(bnn_hubara_binarization),
        BnnTConv2d(C_in, C_out, kernel_size, stride, padding,output_padding=output_padding ,dilation=dilation,quantization_class=bnn_hubara_binarization, step=step),
        nn.BatchNorm2d(C_out, affine=affine),
        nn.Tanh()
      )
  def forward(self, x):
    return self.op(x)



class DilConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
    super(DilConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)


class SepConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(SepConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_in, affine=affine),
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)


class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x


class Zero(nn.Module):

  def __init__(self, stride, upsample=False):
    super(Zero, self).__init__()
    self.stride = stride
    self.upsample = upsample
  def forward(self, x):
    if self.upsample:
      return F.interpolate(x.mul(0.),scale_factor=self.stride)
    else:
      if self.stride == 1:
        return x.mul(0.)
      return x[:,:,::self.stride,::self.stride].mul(0.)


class FactorizedReduce(nn.Module):
  def __init__(self, C_in, C_out, affine=True):
    # downsample the image by two using two convulotions
    super(FactorizedReduce, self).__init__()
    assert C_out % 2 == 0
    self.relu = nn.ReLU(inplace=False)
    self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False) 
    # Learnable affine parameters in all batch normalizations are disabled 
    # during the search process to avoid rescaling the outputs of the candidate operations.
    self.bn = nn.BatchNorm2d(C_out, affine=affine)

  def forward(self, x):
    x = self.relu(x)
    out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
    out = self.bn(out)
    return out


class SignalUpsample(nn.Module):
  def __init__(self, C_in, C_out, affine=True) -> None:
      super(SignalUpsample, self).__init__()
      assert C_out %2 == 0
      self.relu = nn.ReLU(inplace=False)
      self.bn = nn.BatchNorm2d(C_out, affine=affine)
      self.upsample= nn.UpsamplingBilinear2d(scale_factor=2)
      self.conv = nn.Conv2d(C_in, C_out, 1, padding=0) # match channels

  def forward(self, x):
    x = self.relu(x)
    out = self.conv(x)
    out = self.bn(x)
    out = self.upsample(x)
    return out


class Bnn_Binarization(Function): # Courbariaux, Hubara
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = torch.ones_like(input)
        output[input < 0] = -1
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



activation = Bnn_Binarization.apply

class Activation(nn.Module):
    def __init__(self):
        super(Activation, self).__init__()
    def forward(self, x):
        #x = F.Tanh(x)
        return activation(x)

class QuantizedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        self.name = "QuantizedConv2d"
        self.layerNR = kwargs.pop('layerNr', None)
        super(QuantizedConv2d, self).__init__(*args, **kwargs)

        n = self.kernel_size[0] * self.kernel_size[1] * self.out_channels
        self.weight.data.normal_(0, math.sqrt(2. / n)) # weight init

    def forward(self, input):
        #if self.bias is None or self.bias == False:
        quantized_weight = activation(self.weight)
        output = F.conv2d(input, quantized_weight, self.bias, self.stride,
                              self.padding, self.dilation, self.groups)
        return output
        #else:
         #   quantized_weight = activation(self.weight)
         #   quantized_bias = activation(self.bias)
         #   output = F.conv2d(input, quantized_weight, quantized_bias, self.stride,
         #                     self.padding, self.dilation, self.groups)
         #   return output


class QuantizedTransposedConv2d(nn.ConvTranspose2d):
    def __init__(self, *args, **kwargs):
        self.name = "QuantizedTransposedConv2d"
        self.layerNR = kwargs.pop('layerNr', None)
        super(QuantizedTransposedConv2d, self).__init__(*args, **kwargs)

        n = self.kernel_size[0] * self.kernel_size[1] * self.out_channels
        self.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, input):
        #if self.bias is None:
        quantized_weight = activation(self.weight)
        output = F.conv_transpose2d(input, quantized_weight, self.bias, self.stride,
                              self.padding, self.output_padding, self.groups ,self.dilation)
        return output
        #else:
        #    quantized_weight = activation(self.weight)
        #    quantized_bias = activation(self.bias)
        #    output = F.conv_transpose2d(input, quantized_weight, quantized_bias, self.stride,
        #                      self.padding, self.output_padding, self.groups ,self.dilation)
        #return output

if __name__ == '__main__':
  layer = Zero(2)
  layer2 = Zero(2, True)
  inp = torch.randn([1,3,2,2])
  out = layer(inp)
  out2 = layer2(inp)