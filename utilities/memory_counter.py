import torch
import torch.nn as nn
import numpy as np 


def memory_forward_hook(mod, input, output):
    input_mem = 0
    output_mem = 0
    weight_mem = 0
    #if hasattr(mod, 'scale_factor'):
    #    print(mod)
    #    print(len(input[0]))
    #    print(input[0].shape)
    #    print(output[0].shape)
    # input
    #if torch.is_tensor(input[0]):
    #    print(mod, input[0].shape, output.shape)
    if isinstance(input[0], list) or isinstance(input[0], tuple):
        list_len = len(input[0])
        for i in range(list_len):
            unique_vals = torch.unique(input[0][i]).shape[0]
            if unique_vals <= 3:
                if unique_vals == 3:
                    print(mod)
                    print('Warning: Ternary Input ', mod)
                input_mem += np.prod(input[0][i].shape)*1/8
            else:
                print(input[0][i].shape)
                input_mem += np.prod(input[0][i].shape)*32/8
    else:
        unique_vals = torch.unique(input[0]).shape[0]
        if unique_vals <= 3:
            if unique_vals == 3:
                    print(mod)
                    print('Warning: Ternary Input ', mod)
            input_mem += np.prod(input[0].shape)*1/8
        else:
            input_mem += np.prod(input[0].shape)*32/8
    # output     
    unique_vals = torch.unique(output).shape[0]
    if unique_vals <= 3:
        if unique_vals == 3:
                print('Warning: Ternary Output ', mod)
        output_mem += np.prod(output[0].shape)*1/8
    else:
        output_mem += np.prod(output[0].shape)*32/8
    # weight
    if hasattr(mod, 'weight'):
        if mod.weight is not None:
            if hasattr(mod.weight,'bin'):
                weight_mem += np.prod(mod.weight.shape)*1/8
            else:
                weight_mem += np.prod(mod.weight.shape)*32/8
        if hasattr(mod, 'bias'):
            if mod.bias is not None:
                if hasattr(mod.bias,'bin'):
                    weight_mem += np.prod(mod.weight.shape)*1/8
                else:
                    weight_mem += np.prod(mod.weight.shape)*32/8
    layer_mem = input_mem + output_mem + weight_mem
    mod.__memory =  round(layer_mem*1e-6)



def get_max_mem(net):
    max_mem = [-1]
    max_layer = [None]
    def recur(net):
        for mod in net.children():
            if hasattr(mod, '__memory'):
                if mod.__memory > max_mem[0]:
                    #print(round(mod.__memory, 2), mod)
                    max_layer[0] = mod
                max_mem[0] = max(max_mem[0], mod.__memory)

            recur(mod)
    recur(net)
    max_mem = max_mem[0]
    print('mem max layer', max_layer)
    return max_mem

def max_mem_counter(net, dummy_input_shape, device='cuda'):
    dummy = torch.randn(dummy_input_shape, device=device)
    handles = []
    def attach_hooks(net):
        leaf_layers = 0
        for mod in net.children():
            leaf_layers += 1
            attach_hooks(mod)
        if leaf_layers == 0:
            handles.append(net.register_forward_hook(memory_forward_hook))
    attach_hooks(net)
    net(dummy) 
    max_mem_bytes = get_max_mem(net)
    for handle in handles:
        handle.remove()
    return round(max_mem_bytes, 2)

def remove_mem_vars(net):
    def recur(net):
        for mod in net.children():
            if hasattr(mod, '__memory'):
                del mod.__memory
            recur(mod)
    recur(net)


if __name__ == '__main__':
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
            x = self.con(x.sign())
            x = self.layer(x)
            return x
    dummy_input_shape = (1,3, 420,420)
    net = Net()
    max_mem = max_mem_counter(net, dummy_input_shape)
    #print(max_mem)
    dummy = torch.randn(dummy_input_shape)
    net(dummy)
    def re(net):
        for mod in net.children():
            if hasattr(mod, '__memory'):
                print(mod.__memory, mod)
            re(mod)
    re(net)