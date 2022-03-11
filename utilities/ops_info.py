import types
import torch
import torch.nn as nn
import numpy as np 
import sys
sys.path.append('search/darts/')
from architecture.networks.cell.operations.binarization.binarized_layers import BinConvTranspose2d, BinConv2d, BinActivation
from architecture.networks.cell.operations.search_operations import Cat, Sum, Bilinear, Identity

def identity_ops_forward_hook(mod, input, output):
    mod.__ops = 0


def binconv_ops_forward_hook(mod, input, output): # weight binarization is not included
    #input_binarization_ops = np.prod(input[0].shape)
    input_binarization_ops = 0
    ops = np.prod(mod.weight.shape)
    ops *= np.prod(output.shape[2:])
    ops *= output.shape[0]
    ops/=64
    mod.__ops = (ops+input_binarization_ops)*1e-6

def bintconv_ops_forward_hook(mod, input, output): # same as binconv. weight binarization is not included
    #input_binarization_ops = np.prod(input[0].shape)
    input_binarization_ops = 0
    ops = np.prod(mod.weight.shape)
    ops *= np.prod(output.shape[2:])
    ops *= output.shape[0]
    ops/=64
    mod.__ops = (ops+input_binarization_ops)*1e-6

def pool_ops_forward_hook(mod, input, output):
    mod.__ops = np.prod(output.shape) * 1e-6

def cat_ops_forward_hook(mod, input, output):
    mod.__ops = 0

def bilinear_ops_forward_hook(mod, input, output):
    mod.__ops = np.prod(output.shape) * 11 *1e-6
    
def sum_ops_forward_hook(mod, input, output):
    bin_inputs = 0
    if isinstance(input[0], list) or isinstance(input[0], tuple):
        list_len = len(input[0])
        for i in range(list_len):
            unique_vals = torch.unique(input[0][i]).shape[0]
            if unique_vals <= 3:
                if unique_vals == 3:
                    print('Warning: Ternary')
                bin_inputs += 1
        if bin_inputs == 0:
            ops = np.prod(output.shape)*(list_len-1)
        else:
            ops = np.prod(output.shape)*(bin_inputs - 1)/64 + np.prod(output.shape)*(list_len-bin_inputs)
    else:
        #wrong shortcut
        #print(len(list(input[0])))
        #print('Sum ops Warning!: line 55 in ops_info')
        ops = np.prod(output.shape)*(5)

    mod.__ops = ops*1e-6 

def binarization_ops_forward_hook(mod, input, output):
    mod.__ops = np.prod(output.shape)*1e-6

def tanh_ops_forward_hook(mod, input, output):
    mod.__ops = np.prod(output.shape)* 1e-6

def bn_ops_forward_hook(mod, input, output):
    ops = np.prod(output.shape)
    if mod.affine:
        ops *= 2
    mod.__ops = ops*1e-6

def conv_ops_forward_hook(mod, input, output):
    ops = np.prod(mod.weight.shape)
    ops *= np.prod(output.shape[2:])
    ops *= output.shape[0]
    mod.__ops = ops*1e-6




def get_total_ops(net):
    total_ops = [0]
    def recur(net):
        for mod in net.children():
            if hasattr(mod, '__ops'):
                total_ops[0] +=  mod.__ops
            recur(mod)
    recur(net)
    total_ops = total_ops[0]
    return total_ops

def ops_counter(net, dummy_input_shape, device='cuda'):
    dummy = torch.randn(dummy_input_shape).to(device)
    handles = []
    def attach_hooks(net):
        for mod in net.children():
            if isinstance(mod, BinConv2d):
                handles.append(mod.register_forward_hook(binconv_ops_forward_hook))
            elif isinstance(mod, BinConvTranspose2d):
                handles.append(mod.register_forward_hook(bintconv_ops_forward_hook))
            elif isinstance(mod, Sum):
                handles.append(mod.register_forward_hook(sum_ops_forward_hook))
            elif isinstance(mod, Bilinear):
                handles.append(mod.register_forward_hook(bilinear_ops_forward_hook))
            elif isinstance(mod, nn.MaxPool2d) or isinstance(mod, nn.AvgPool2d):
                handles.append(mod.register_forward_hook(pool_ops_forward_hook))
            elif isinstance(mod, Identity):
                handles.append(mod.register_forward_hook(identity_ops_forward_hook))
            elif isinstance(mod, nn.BatchNorm2d):
                handles.append(mod.register_forward_hook(bn_ops_forward_hook))
            elif isinstance(mod, nn.Hardtanh):
                handles.append(mod.register_forward_hook(tanh_ops_forward_hook))
            elif isinstance(mod, Cat):
                handles.append(mod.register_forward_hook(cat_ops_forward_hook))
            elif isinstance(mod, nn.Conv2d):
                handles.append(mod.register_forward_hook(conv_ops_forward_hook))
            elif isinstance(mod, nn.ConvTranspose2d):
                handles.append(mod.register_forward_hook(conv_ops_forward_hook))
            elif isinstance(mod, BinActivation):
                handles.append(mod.register_forward_hook(binarization_ops_forward_hook))
            else:
                #print(f'Warning: {type(mod)} ops is not defined')
                pass
            attach_hooks(mod)
    attach_hooks(net)
    with torch.no_grad():
        net(dummy) 
    total_ops = get_total_ops(net)
    for handle in handles:
        handle.remove()
    return round(total_ops) # giga ops