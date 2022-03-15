import torch
import torch.nn as nn
from .binarized_layers import BinActivation, BinConv2d, BinConvTranspose2d
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


class BinConvBnHTanh(nn.Module):
    '''
    binarize -> conv -> batchnorm -> hardtanh
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding = 0, dilation = 1, affine=True, padding_mode='zeros', jit=False):
        super(BinConvBnHTanh, self).__init__()
        #self.ops = nn.Sequential
        self.conv = BinConv2d(in_channels, out_channels, kernel_size, stride=stride,padding= padding, dilation=dilation,padding_mode=padding_mode)
        self.batchnorm = nn.BatchNorm2d(out_channels, affine=affine)
        self.htanh = nn.Hardtanh(-1, 1, True)
        self.binarize = BinActivation(jit)
        self.latency_table = {}
    def forward(self, x):
        x = self.binarize(x)
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.htanh(x)
        return x

    def ops(self, x):
        with torch.no_grad():
            output = self(x[0,:,:,:].unsqueeze(0))
        # weight binarization and hardtanh won't be taken into account
        conv_ops = FlopsCounter.count('conv', (output.shape, self.conv.weight.shape))
        batchnorm_flops = FlopsCounter.count('batchnorm',  (output.shape, self.batchnorm.affine))
        binarization_flops = FlopsCounter.count('binarize', (output.shape, None))

        flops = batchnorm_flops + binarization_flops
        bops = conv_ops
        
        ops = flops + bops/64
        #return (ops,flops), output 
        return ops

    def max_memory(self, x):
        with torch.no_grad():
            output = self(x[0,:,:,:].unsqueeze(0))
        conv_memory, output_mem = MemoryCounter.count('conv', (x.shape, output.shape, self.conv.weight.shape, 1))
        batchnorm_memory, output_mem = MemoryCounter.count('batchnorm', (output_mem, output.shape, self.batchnorm.affine))
        binarization_memory, output_mem = MemoryCounter.count('binarize', (output_mem, output.shape))

        max_memory = max(conv_memory, batchnorm_memory, binarization_memory)
        #return (max_memory, output_mem),output
        return max_memory
    
    def params(self, x):
        #with torch.no_grad():
        #    output = self(x[0,:,:,:])
        bin_params = sum([p.numel() for p in self.conv.parameters()])*1/8
        fp_params = sum([p.numel() for p in self.batchnorm.parameters()]) *32/8
        params = fp_params + bin_params
        #return (params, fp_params), output
        return params
    
    def latency(self, x):
        if self.latency_table.get(str(x.shape[-3:]), None) is None:
            with torch.no_grad():
                l = get_latency(self.model(self.get_config()),x[0,:,:,:].unsqueeze(0))
                self.latency_table[str(x.shape[-3:])] = l
        else:
            l = self.latency_table.get(str(x.shape[-3:]))
        #with torch.no_grad():
        #    output = self(x)
        #return l, output
        return l

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
        return {'in_channels':self.conv.in_channels, 'out_channels':self.conv.out_channels, 'kernel_size':self.conv.kernel_size[0], 'stride':self.conv.stride[0], 'padding':self.conv.padding[0], 'dilation':self.conv.dilation[0], 'affine':self.batchnorm.affine, 'padding_mode':self.conv.padding_mode, 'jit':self.binarize.jit}
    
    @classmethod
    def model(cls, config):
        return cls(**config)


class BinTConvBnHTanh(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding = 0, dilation = 1, affine=True, padding_mode='zeros',jit=False):
        super(BinTConvBnHTanh, self).__init__()
        #self.ops = nn.Sequential
        self.conv = BinConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,padding= padding, dilation=dilation, padding_mode=padding_mode)
        self.batchnorm = nn.BatchNorm2d(out_channels, affine=affine)
        self.htanh = nn.Hardtanh(-1, 1, True)
        self.binarize = BinActivation(jit)
        self.latency_table = {}

    def forward(self, x):
        x = self.binarize(x)
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.htanh(x)
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
        #return (ops, flops), output 
        return ops

    def max_memory(self, x):
        with torch.no_grad():
            output = self(x[0,:,:,:].unsqueeze(0))
        conv_memory, output_mem = MemoryCounter.count('tconv', (x.shape, output.shape, self.conv.weight.shape, 1))
        batchnorm_memory, output_mem = MemoryCounter.count('batchnorm', (output_mem, output.shape, self.batchnorm.affine))
        binarization_memory, output_mem = MemoryCounter.count('binarize', (output_mem, output.shape))

        max_memory = max(conv_memory, batchnorm_memory, binarization_memory)
        #return (max_memory, output_mem),output
        return max_memory
    
    def params(self, x):
        #with torch.no_grad():
        #    output = self(x)
        bin_params = sum([p.numel() for p in self.conv.parameters()])
        fp_params = sum([p.numel() for p in self.batchnorm.parameters()]) 
        params = fp_params*32/8 + bin_params*1/8
        #return params, output
        return params
    
    def latency(self, x):
        if self.latency_table.get(str(x.shape[-3:]), None) is None:
            with torch.no_grad():
                l = get_latency(self.model(self.get_config()), x[0,:,:,:].unsqueeze(0))
                self.latency_table[str(x.shape[-3:])] = l
        else:
            l = self.latency_table.get(str(x.shape[-3:]))
        #with torch.no_grad():
        #    output = self(x)
        #return l, output
        return l

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


class ConvBnHTanhBin (nn.Module):
    '''
    conv -> batchnorm -> hardtanh -> binarize
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding = 0, dilation = 1, affine=True, padding_mode='zeros', jit=False):
        super(ConvBnHTanhBin, self).__init__()
        #self.ops = nn.Sequential
        self.conv = BinConv2d(in_channels, out_channels, kernel_size, stride=stride,padding= padding, dilation=dilation,padding_mode=padding_mode)
        self.batchnorm = nn.BatchNorm2d(out_channels, affine=affine)
        self.htanh = nn.Hardtanh(-1, 1, True)
        self.binarize = BinActivation(jit)
        self.latency_table={}

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
        conv_ops = FlopsCounter.count('binconv', (output.shape, self.conv.weight.shape))
        batchnorm_flops = FlopsCounter.count('batchnorm',  (output.shape, self.batchnorm.affine))
        binarization_flops = FlopsCounter.count('binarize', (output.shape, None))

        flops = batchnorm_flops + binarization_flops
        bops = conv_ops
        
        ops = flops + bops/64
        return (ops, flops), output 

    def max_memory(self, x):
        with torch.no_grad():
            output = self(x)
        conv_memory, output_mem = MemoryCounter.count('binconv', (x.shape, output.shape, self.conv.weight.shape, 1))
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
                plot_weight(self.conv.weight)

    def get_config(self):
        return {'in_channels':self.conv.in_channels, 'out_channels':self.conv.out_channels, 'kernel_size':self.conv.kernel_size[0], 'stride':self.conv.stride[0], 'padding':self.conv.padding[0], 'dilation':self.conv.dilation[0], 'affine':self.batchnorm.affine, 'padding_mode':self.conv.padding_mode,'jit':self.binarize.jit}
    
    @classmethod
    def model(cls, config):
        return cls(**config)


class TConvBnHTanhBin (nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding = 0, dilation = 1, affine = True, padding_mode='zeros', jit=False):
        super(TConvBnHTanhBin, self).__init__()
        #self.ops = nn.Sequential
        self.conv = BinConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,padding= padding, dilation=dilation, padding_mode=padding_mode)
        self.batchnorm = nn.BatchNorm2d(out_channels, affine=affine )
        self.htanh = nn.Hardtanh(-1, 1, True)
        self.binarize = BinActivation(jit)

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

class BinConvBn(nn.Module):
    '''
    binarize -> conv -> batchnorm
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding = 0, dilation = 1, affine=True, padding_mode='zeros', jit =False):
        super(BinConvBn, self).__init__()
        #self.ops = nn.Sequential
        self.conv = BinConv2d(in_channels, out_channels, kernel_size, stride=stride,padding= padding, dilation=dilation, padding_mode=padding_mode)
        self.batchnorm = nn.BatchNorm2d(out_channels, affine=affine)
        self.binarize = BinActivation(jit)
        self.latency_table = {}
    def forward(self, x):
        x = self.binarize(x)
        x = self.conv(x)
        x = self.batchnorm(x)
        return x

    def ops(self, x):
        with torch.no_grad():
            output = self(x[0,:,:,:].unsqueeze(0))
        # weight binarization and hardtanh won't be taken into account
        conv_ops = FlopsCounter.count('conv', (output.shape, self.conv.weight.shape))
        batchnorm_flops = FlopsCounter.count('batchnorm',  (output.shape, self.batchnorm.affine))
        binarization_flops = FlopsCounter.count('binarize', (output.shape, None))

        flops = batchnorm_flops + binarization_flops
        bops = conv_ops
        
        ops = flops + bops/64
        #return (ops,flops), output 
        return ops

    def max_memory(self, x):
        with torch.no_grad():
            output = self(x[0,:,:,:].unsqueeze(0))
        conv_memory, output_mem = MemoryCounter.count('conv', (x.shape, output.shape, self.conv.weight.shape, 1))
        batchnorm_memory, output_mem = MemoryCounter.count('batchnorm', (output_mem, output.shape, self.batchnorm.affine))
        binarization_memory, output_mem = MemoryCounter.count('binarize', (output_mem, output.shape))

        max_memory = max(conv_memory, batchnorm_memory, binarization_memory)
        #return (max_memory, output_mem),output
        return max_memory
    
    def params(self, x):
        #with torch.no_grad():
        #    output = self(x[0,:,:,:])
        bin_params = sum([p.numel() for p in self.conv.parameters()])*1/8
        fp_params = sum([p.numel() for p in self.batchnorm.parameters()]) *32/8
        params = fp_params + bin_params
        #return (params, fp_params), output
        return params
    
    def latency(self, x):
        if self.latency_table.get(str(x.shape[-3:]), None) is None:
            with torch.no_grad():
                l = get_latency(self.model(self.get_config()),x[0,:,:,:].unsqueeze(0))
                self.latency_table[str(x.shape[-3:])] = l
        else:
            l = self.latency_table.get(str(x.shape[-3:]))
        #with torch.no_grad():
        #    output = self(x)
        #return l, output
        return l

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
        return {'in_channels':self.conv.in_channels, 'out_channels':self.conv.out_channels, 'kernel_size':self.conv.kernel_size, 'stride':self.conv.stride, 'padding':self.conv.padding, 'dilation':self.conv.dilation, 'affine':self.batchnorm.affine, 'padding_mode':self.conv.padding_mode,'jit':self.binarize.jit}
    
    @classmethod
    def model(cls, config):
        return cls(**config)

class ConvBn(nn.Module):
    '''
    conv -> batchnorm
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding = 0, dilation = 1, affine=True, padding_mode='zeros', jit=False):
        super(ConvBn, self).__init__()
        #self.ops = nn.Sequential
        self.conv = BinConv2d(in_channels, out_channels, kernel_size, stride=stride,padding= padding, dilation=dilation, padding_mode=padding_mode, jit=jit)
        self.batchnorm = nn.BatchNorm2d(out_channels, affine=affine)
        self.latency_table = {}
    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        return x

    def ops(self, x):
        with torch.no_grad():
            output = self(x[0,:,:,:].unsqueeze(0))
        # weight binarization and hardtanh won't be taken into account
        conv_ops = FlopsCounter.count('conv', (output.shape, self.conv.weight.shape))
        batchnorm_flops = FlopsCounter.count('batchnorm',  (output.shape, self.batchnorm.affine))
        binarization_flops = FlopsCounter.count('binarize', (output.shape, None))

        flops = batchnorm_flops + binarization_flops
        bops = conv_ops
        
        ops = flops + bops/64
        #return (ops,flops), output 
        return ops

    def max_memory(self, x):
        with torch.no_grad():
            output = self(x[0,:,:,:].unsqueeze(0))
        conv_memory, output_mem = MemoryCounter.count('conv', (x.shape, output.shape, self.conv.weight.shape, 1))
        batchnorm_memory, output_mem = MemoryCounter.count('batchnorm', (output_mem, output.shape, self.batchnorm.affine))
        binarization_memory, output_mem = MemoryCounter.count('binarize', (output_mem, output.shape))

        max_memory = max(conv_memory, batchnorm_memory, binarization_memory)
        #return (max_memory, output_mem),output
        return max_memory
    
    def params(self, x):
        #with torch.no_grad():
        #    output = self(x[0,:,:,:])
        bin_params = sum([p.numel() for p in self.conv.parameters()])*1/8
        fp_params = sum([p.numel() for p in self.batchnorm.parameters()]) *32/8
        params = fp_params + bin_params
        #return (params, fp_params), output
        return params
    
    def latency(self, x):
        if self.latency_table.get(str(x.shape[-3:]), None) is None:
            with torch.no_grad():
                l = get_latency(self.model(self.get_config()),x[0,:,:,:].unsqueeze(0))
                self.latency_table[str(x.shape[-3:])] = l
        else:
            l = self.latency_table.get(str(x.shape[-3:]))
        #with torch.no_grad():
        #    output = self(x)
        #return l, output
        return l

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