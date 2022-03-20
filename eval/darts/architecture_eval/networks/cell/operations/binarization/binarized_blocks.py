import torch
import torch.nn as nn
from .binarized_layers import EvalBinActivation, EvalBinConv2d, EvalBinConvTranspose2d
from .flops import FlopsCounter
from .memory import MemoryCounter
from .latency import get_latency
from .plot import plot_tensor_dist, plot_weight

class BinaryBlock(nn.Module):
    def __init__(self) -> None:
        super(BinaryBlock, self).__init__()
        raise NotImplementedError
    
    def forward(self, x):
        raise NotImplementedError

# *
class BinConvBnHTanh(nn.Module):
    '''
    binarize -> conv -> batchnorm -> hardtanh
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding = 0, dilation = 1, affine=True, padding_mode='zeros', jit=False, dropout2d=0.1, binarizatoin=1):
        super(BinConvBnHTanh, self).__init__()
        #self.ops = nn.Sequential
        self.conv = EvalBinConv2d(in_channels, out_channels, kernel_size, stride=stride,padding= padding, dilation=dilation,padding_mode=padding_mode,jit=jit)
        self.batchnorm = nn.BatchNorm2d(out_channels, affine=affine)
        self.tanh = nn.Tanh()
        self.htanh = nn.Hardtanh(-1, 1)
        #self.relu = nn.ReLU()
        self.activation = nn.ReLU()
        #self.max_pool = nn.MaxPool2d(2, padding=1)
        self.dropout = nn.Dropout2d(p=dropout2d)
        self.binarize = EvalBinActivation(jit, binarizatoin)
        self.latency_table = {}
    def forward(self, x):
        x = self.binarize(x)
        x = self.conv(x)
        #x = self.relu(x)
        x = self.activation(x)
        x= self.dropout(x)
        #x = self.max_pool(x)
        x = self.batchnorm(x)
        #x = self.tanh(x)
        #x = self.htanh(x)
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
        return {'in_channels':self.conv.in_channels, 'out_channels':self.conv.out_channels, 'kernel_size':self.conv.kernel_size[0], 'stride':self.conv.stride[0], 'padding':self.conv.padding[0], 'dilation':self.conv.dilation[0], 'affine':self.batchnorm.affine, 'padding_mode':self.conv.padding_mode, 'jit':self.binarize.jit,'dropout2d':self.dropout.p, 'binarization':self.binarize.binarization}
    
    @classmethod
    def model(cls, config):
        return cls(**config)

# *
class BinTConvBnHTanh(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding = 0, dilation = 1, affine=True, padding_mode='zeros',jit=False, dropout2d=0.1, binarizatoin=1):
        super(BinTConvBnHTanh, self).__init__()
        #self.ops = nn.Sequential
        self.conv = EvalBinConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,padding= padding, dilation=dilation, padding_mode=padding_mode,jit=jit)
        self.batchnorm = nn.BatchNorm2d(out_channels, affine=affine)
        self.htanh = nn.Hardtanh(-1, 1)
        self.relu = nn.ReLU()
        #self.max_pool = nn.MaxPool2d(2,padding=1)
        self.dropout = nn.Dropout2d(p=dropout2d)
        self.binarize = EvalBinActivation(jit,binarizatoin)
        self.latency_table = {}
        self.tanh=nn.Tanh()
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.binarize(x)
        x = self.conv(x)
        #x = self.relu(x)
        x = self.activation(x)
        x = self.dropout(x)
        #x = self.max_pool(x)
        x = self.batchnorm(x)
        #x = self.tanh(x)
        #x = self.htanh(x)
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
        return {'in_channels':self.conv.in_channels, 'out_channels':self.conv.out_channels, 'kernel_size':self.conv.kernel_size[0], 'stride':self.conv.stride[0], 'padding':self.conv.padding[0], 'dilation':self.conv.dilation[0], 'affine':self.batchnorm.affine, 'padding_mode':self.conv.padding_mode, 'jit':self.binarize.jit, 'dropout2d':self.dropout.p, 'binarization':self.binarize.binarization}
    
    @classmethod
    def model(cls, config):
        return cls(**config)


# * last layer
class BinConvBn(nn.Module):
    '''
    binarize -> conv -> batchnorm
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding = 0, dilation = 1, affine=True, padding_mode='zeros', jit =False,dropout2d=0.1, binarizatoin=1):
        super(BinConvBn, self).__init__()
        #self.ops = nn.Sequential
        self.conv = EvalBinConv2d(in_channels, out_channels, kernel_size, stride=stride,padding= padding, dilation=dilation, padding_mode=padding_mode, jit=jit)
        self.batchnorm = nn.BatchNorm2d(out_channels, affine=affine)
        self.binarize = EvalBinActivation(jit,binarizatoin)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(dropout2d)
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
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding = 0, dilation = 1, affine=True, padding_mode='zeros', jit=False,binarizatoin=1):
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
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding = 0, dilation = 1, affine=True, padding_mode='zeros', jit=False,binarizatoin=1):
        super(BinConv, self).__init__()
        #self.ops = nn.Sequential
        self.binarize = EvalBinActivation(jit,binarizatoin)
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
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding = 0, dilation = 1, affine=True, padding_mode='zeros', jit=False,binarizatoin=1):
        super(ConvBnHTanhBin, self).__init__()
        #self.ops = nn.Sequential
        self.conv = EvalBinConv2d(in_channels, out_channels, kernel_size, stride=stride,padding= padding, dilation=dilation,padding_mode=padding_mode, jit=jit)
        self.batchnorm = nn.BatchNorm2d(out_channels, affine=affine)
        self.htanh = nn.Hardtanh(-1, 1, True)
        self.binarize = EvalBinActivation(jit, binarizatoin=1)
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
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding = 0, dilation = 1, affine = True, padding_mode='zeros', jit=False,binarizatoin=1):
        super(TConvBnHTanhBin, self).__init__()
        #self.ops = nn.Sequential
        self.conv = EvalBinConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,padding= padding, dilation=dilation, padding_mode=padding_mode, jit=jit)
        self.batchnorm = nn.BatchNorm2d(out_channels, affine=affine )
        self.htanh = nn.Hardtanh(-1, 1, True)
        self.binarize = EvalBinActivation(jit,binarizatoin)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.htanh(x)
        x = self.binarize(x)
        return x
    
    def ops(self, x):
        with torch.no_grad():
            output = self(x)
        # weight binarization and hardtanh won't be taken into account
        conv_ops = FlopsCounter.count('tconv', (output.shape, self.conv.weight.shape))
        batchnorm_flops = FlopsCounter.count('batchnorm',  (output.shape, self.batchnorm.affine))
        binarization_flops = FlopsCounter.count('binarize', (output.shape, None))

        flops = batchnorm_flops + binarization_flops
        bops = conv_ops
        
        ops = flops + bops/64
        return (ops, flops), output 

    def max_memory(self, x):
        with torch.no_grad():
            output = self(x)
        conv_memory, output_mem = MemoryCounter.count('tconv', (x.shape, output.shape, self.conv.weight.shape, 1))
        batchnorm_memory, output_mem = MemoryCounter.count('batchnorm', (output_mem, output.shape, self.batchnorm.affine))
        binarization_memory, output_mem = MemoryCounter.count('binarize', (output_mem, output.shape))

        max_memory = max(conv_memory, batchnorm_memory, binarization_memory)
        return max_memory, output_mem,output
    
    def params(self, x):
        with torch.no_grad():
            output = self(x)
        bin_params = sum([p.numel() for p in self.conv.parameters()])
        fp_params = sum([p.numel() for p in self.batchnorm.parameters()]) 
        params = fp_params*32/8 + bin_params*1/8
        return params, output

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