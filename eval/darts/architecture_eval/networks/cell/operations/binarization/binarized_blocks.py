import torch
import torch.nn as nn
from .binarized_layers import EvalBinActivation, EvalBinConv2d, EvalBinConvTranspose2d
from .plot import plot_tensor_dist, plot_weight


# *
class BinConvBnHTanh(nn.Module):
    '''
    binarize -> conv -> batchnorm -> hardtanh
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding = 0, dilation = 1, affine=True, padding_mode='zeros', jit=False, dropout2d=0.1, binarization=1, activation='htanh'):
        super(BinConvBnHTanh, self).__init__()
        self.activation_func = activation
        self.conv = EvalBinConv2d(in_channels, out_channels, kernel_size, stride=stride,padding= padding, dilation=dilation,padding_mode=padding_mode,jit=jit)
        self.batchnorm = nn.BatchNorm2d(out_channels, affine=affine)
        if activation =='htanh':
            self.activation = nn.Hardtanh(-1, 1)
        if activation == 'relu':
            self.activation = nn.ReLU()
        
        self.dropout = nn.Dropout2d(p=dropout2d)
        self.binarize = EvalBinActivation(jit, binarization)
        self.latency_table = {}
    def forward(self, x):
        x = self.binarize(x)
        x = self.conv(x)

        if self.activation == 'relu':
            x = self.activation(x)
            x= self.dropout(x)

        x = self.batchnorm(x)

        if self.activation == 'htanh':
            x = self.activation(x)
            x= self.dropout(x)
        return x

    def plot_activation(self, x, path=None):
        if path is None:
            return self(x)
        else:

            raise NotImplementedError

    def plot_weights(self, x, path=None):
        with torch.no_grad():
            if path is None:
                return self(x)
            else:
                plot_weight(self.conv.weight)

    def get_config(self):
        return {'in_channels':self.conv.in_channels, 'out_channels':self.conv.out_channels, 'kernel_size':self.conv.kernel_size[0], 'stride':self.conv.stride[0], 'padding':self.conv.padding[0], 'dilation':self.conv.dilation[0], 'affine':self.batchnorm.affine, 'padding_mode':self.conv.padding_mode, 'jit':self.binarize.jit,'dropout2d':self.dropout.p, 'binarization':self.binarize.binarization, 'activation':self.activation_func}
    
    @classmethod
    def model(cls, config):
        return cls(**config)

# *
class BinTConvBnHTanh(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding = 0, dilation = 1, affine=True, padding_mode='zeros',jit=False, dropout2d=0.1, binarization=1, activation='htanh'):
        super(BinTConvBnHTanh, self).__init__()
        self.activation_func = activation
        self.conv = EvalBinConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,padding= padding, dilation=dilation, padding_mode=padding_mode,jit=jit)
        self.batchnorm = nn.BatchNorm2d(out_channels, affine=affine)
        if activation =='htanh':
            self.activation = nn.Hardtanh(-1, 1)
        if activation == 'relu':
            self.activation = nn.ReLU()
        self.dropout = nn.Dropout2d(p=dropout2d)
        self.binarize = EvalBinActivation(jit,binarization)
        self.latency_table = {}

    def forward(self, x):
        x = self.binarize(x)
        x = self.conv(x)
        if self.activation == 'relu':
            x = self.activation(x)
            x= self.dropout(x)

        x = self.batchnorm(x)

        if self.activation == 'htanh':
            x = self.activation(x)
            x= self.dropout(x)
        return x
    
    def plot_activation(self, x, path=None):
        if path is None:
            return self(x)
        else:

            raise NotImplementedError

    def plot_weights(self, x, path=None):
        with torch.no_grad():
            if path is None:
                return self(x)
            else:
                plot_weight(self.conv.weight.sign())

    def get_config(self):
        return {'in_channels':self.conv.in_channels, 'out_channels':self.conv.out_channels, 'kernel_size':self.conv.kernel_size[0], 'stride':self.conv.stride[0], 'padding':self.conv.padding[0], 'dilation':self.conv.dilation[0], 'affine':self.batchnorm.affine, 'padding_mode':self.conv.padding_mode, 'jit':self.binarize.jit, 'dropout2d':self.dropout.p, 'binarization':self.binarize.binarization,'activation':self.activation_func}
    
    @classmethod
    def model(cls, config):
        return cls(**config)


# * last layer
class BinConvBn(nn.Module):
    '''
    binarize -> conv -> batchnorm
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding = 0, dilation = 1, affine=True, padding_mode='zeros', jit =False,dropout2d=0.1, binarization=1):
        super(BinConvBn, self).__init__()
        #self.ops = nn.Sequential
        self.conv = EvalBinConv2d(in_channels, out_channels, kernel_size, stride=stride,padding= padding, dilation=dilation, padding_mode=padding_mode, jit=jit)
        self.batchnorm = nn.BatchNorm2d(out_channels, affine=affine)
        self.binarize = EvalBinActivation(jit,binarization)
        self.latency_table = {}
    def forward(self, x):
        x = self.binarize(x)
        x = self.conv(x)
        x = self.batchnorm(x)
        return x

    def plot_activation(self, x, path=None):
        if path is None:
            return self(x)
        else:

            raise NotImplementedError

    def plot_weights(self, x, path=None):
        with torch.no_grad():
            if path is None:
                return self(x)
            else:
                plot_weight(self.conv.weight)

    def get_config(self):
        return {'in_channels':self.conv.in_channels, 'out_channels':self.conv.out_channels, 'kernel_size':self.conv.kernel_size, 'stride':self.conv.stride, 'padding':self.conv.padding, 'dilation':self.conv.dilation, 'affine':self.batchnorm.affine, 'padding_mode':self.conv.padding_mode,'jit':self.binarize.jit,'dropout2d':self.dropout.p, 'binarization':self.binarize.binarization}
    
    @classmethod
    def model(cls, config):
        return cls(**config)


class ConvBn(nn.Module):
    '''
    conv -> batchnorm
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding = 0, dilation = 1, affine=True, padding_mode='zeros', jit=False,binarization=1, activation='htanh'):
        super(ConvBn, self).__init__()
        #self.ops = nn.Sequential
        self.conv = EvalBinConv2d(in_channels, out_channels, kernel_size, stride=stride,padding= padding, dilation=dilation, padding_mode=padding_mode, jit=jit)
        self.batchnorm = nn.BatchNorm2d(out_channels, affine=affine)
        self.latency_table = {}

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        return x

    def plot_activation(self, x, path=None):
        if path is None:
            return self(x)
        else:

            raise NotImplementedError

    def plot_weights(self, x, path=None):
        with torch.no_grad():
            if path is None:
                return self(x)
            else:
                plot_weight(self.conv.weight)

    def get_config(self):
        return {'in_channels':self.conv.in_channels, 'out_channels':self.conv.out_channels, 'kernel_size':self.conv.kernel_size, 'stride':self.conv.stride, 'padding':self.conv.padding, 'dilation':self.conv.dilation, 'affine':self.batchnorm.affine, 'padding_mode':self.conv.padding_mode, 'jit':self.conv.jit}
    
    @classmethod
    def model(cls, config):
        return cls(**config)

# *
class BinConv(nn.Module):
    '''
    bin -> conv 
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding = 0, dilation = 1, affine=True, padding_mode='zeros', jit=False,binarization=1):
        super(BinConv, self).__init__()
        #self.ops = nn.Sequential
        self.binarize = EvalBinActivation(jit,binarization)
        self.conv = EvalBinConv2d(in_channels, out_channels, kernel_size, stride=stride,padding= padding, dilation=dilation, padding_mode=padding_mode, jit=jit)
        self.latency_table = {}
    def forward(self, x):
        x = self.binarize(x)
        x = self.conv(x)
        return x

    def plot_activation(self, x, path=None):
        if path is None:
            return self(x)
        else:

            raise NotImplementedError

    def plot_weights(self, x, path=None):
        with torch.no_grad():
            if path is None:
                return self(x)
            else:
                plot_weight(self.conv.weight)

    def get_config(self):
        return {'in_channels':self.conv.in_channels, 'out_channels':self.conv.out_channels, 'kernel_size':self.conv.kernel_size, 'stride':self.conv.stride, 'padding':self.conv.padding, 'dilation':self.conv.dilation, 'padding_mode':self.conv.padding_mode, 'jit':self.conv.jit, 'binarization':self.binarize.binarization}
    
    @classmethod
    def model(cls, config):
        return cls(**config)

########################################################
# not used
class ConvBnHTanhBin (nn.Module):
    '''
    conv -> batchnorm -> hardtanh -> binarize
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding = 0, dilation = 1, affine=True, padding_mode='zeros', jit=False,binarization=1):
        super(ConvBnHTanhBin, self).__init__()
        #self.ops = nn.Sequential
        self.conv = EvalBinConv2d(in_channels, out_channels, kernel_size, stride=stride,padding= padding, dilation=dilation,padding_mode=padding_mode, jit=jit)
        self.batchnorm = nn.BatchNorm2d(out_channels, affine=affine)
        self.htanh = nn.Hardtanh(-1, 1, True)
        self.binarize = EvalBinActivation(jit, binarization=1)
        self.latency_table={}

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.htanh(x)
        x = self.binarize(x)
        return x

    def plot_activation(self, x, path=None):
        if path is None:
            return self(x)
        else:

            raise NotImplementedError

    def plot_weights(self, x, path=None):
        with torch.no_grad():
            if path is None:
                return self(x)
            else:
                plot_weight(self.conv.weight)

    def get_config(self):
        return {'in_channels':self.conv.in_channels, 'out_channels':self.conv.out_channels, 'kernel_size':self.conv.kernel_size[0], 'stride':self.conv.stride[0], 'padding':self.conv.padding[0], 'dilation':self.conv.dilation[0], 'affine':self.batchnorm.affine, 'padding_mode':self.conv.padding_mode,'jit':self.binarize.jit}
    
    @classmethod
    def model(cls, config):
        return cls(**config)


class TConvBnHTanhBin (nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding = 0, dilation = 1, affine = True, padding_mode='zeros', jit=False,binarization=1):
        super(TConvBnHTanhBin, self).__init__()
        #self.ops = nn.Sequential
        self.conv = EvalBinConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,padding= padding, dilation=dilation, padding_mode=padding_mode, jit=jit)
        self.batchnorm = nn.BatchNorm2d(out_channels, affine=affine )
        self.htanh = nn.Hardtanh(-1, 1, True)
        self.binarize = EvalBinActivation(jit,binarization)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.htanh(x)
        x = self.binarize(x)
        return x

    def plot_activation(self, x, path=None):
        if path is None:
            return self(x)
        else:

            raise NotImplementedError

    def plot_weights(self, x, path=None):
        with torch.no_grad():
            if path is None:
                return self(x)
            else:
                plot_weight(self.conv.weight.sign())

    def get_config(self):
        return {'in_channels':self.conv.in_channels, 'out_channels':self.conv.out_channels, 'kernel_size':self.conv.kernel_size[0], 'stride':self.conv.stride[0], 'padding':self.conv.padding[0], 'dilation':self.conv.dilation[0], 'affine':self.batchnorm.affine, 'padding_mode':self.conv.padding_mode, 'jit':self.binarize.jit}
    
    @classmethod
    def model(cls, config):
        return cls(**config)