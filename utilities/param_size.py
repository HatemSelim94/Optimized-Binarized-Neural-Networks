import torch
import torch.nn as nn
import numpy as np 



def layer_params_forward_hook(mod, input, output):
    params = 0 
    params +=  sum(p.numel() for p in mod.parameters() if not hasattr(p, 'bin'))*32/8 
    params +=  sum(p.numel() for p in mod.parameters() if hasattr(p, 'bin'))*1/8 
    mod.__params = params

def get_total_param_size(net):
    total_params = [0]
    def recur(net):
        for mod in net.children():
            if hasattr(mod, '__params'):
                total_params[0] +=  mod.__params
            recur(mod)
    recur(net)
    total_params = total_params[0]
    return total_params

def params_size_counter(net, dummy_input_shape, device='cuda'):
    dummy = torch.randn(dummy_input_shape, device=device)
    handles = []
    def attach_hooks(net):
        leaf_layers = 0
        for mod in net.children():
            leaf_layers +=1
            attach_hooks(mod)
        if leaf_layers ==0:
            handles.append(net.register_forward_hook(layer_params_forward_hook))
    attach_hooks(net)
    net(dummy) 
    total_params = get_total_param_size(net)
    for handle in handles:
        handle.remove()
    return total_params # bytes

class Layer(nn.Module):
    def __init__(self) -> None:
        super(Layer, self).__init__()
        self.layer = Conv(6, 6)
    def forward(self, x):
        x = self.layer(x)
        return x

class Conv(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size=1, stride = 1, padding = 0, dilation= 1, groups: int = 1, bias: bool = False, padding_mode: str = 'zeros', device=None, dtype=None) -> None:
        super(Conv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        self.weight.bin = True

class Cat(nn.Module):
    def __init__(self):
        super(Cat, self).__init__()
    def forward(self, x):
        return torch.cat(x, 1)

class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv = nn.Conv2d(3,3,1, stride=1, padding=0, bias=True)
        self.bn = nn.BatchNorm2d(3)
        self.cat = Cat()
        self.con = Conv(12,6)
        self.layer = Layer()
    def forward(self, x):
        x=self.conv(x)
        x = self.bn(x)
        #print(x.shape)
        x = self.cat([x,x,x,x])
        x = self.con(x)
        x = self.layer(x)
        return x


if __name__ == '__main__':
    dummy_input_shape = (1,3, 420,420)
    net = Net()
    total_params = params_size_counter(net, dummy_input_shape)
    print(total_params)
    dummy = torch.randn(dummy_input_shape)
    net(dummy)
    def re(net):
        for mod in net.children():
            if hasattr(mod, '__params'):
                print(mod.__params, mod)
            re(mod)
    re(net)
    params_num_fp = sum(p.numel() for p in net.parameters() if not hasattr(p, 'bin'))*32/8 
    params_num_bin = sum(p.numel() for p in net.parameters() if hasattr(p, 'bin'))*1/8 
    print(params_num_fp+params_num_bin)