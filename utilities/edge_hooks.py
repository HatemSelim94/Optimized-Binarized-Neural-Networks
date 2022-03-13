import torch
import torch.nn as nn
import numpy as np 
import sys
sys.path.append('search/darts/')
from architecture.networks.cell.edge import Edge


def total_metrics_hook(mod, input, output):
    total_ops = [0]
    total_params = [0]
    def recures(main_layer):
        for child in main_layer.children():
            if hasattr(child, '__ops'):
                total_ops[0] += child.__ops
                total_params[0] += child.__params
            recures(child)
    recures(mod)
    mod.__total_ops = total_ops[0]
    mod.__total_params = total_params[0]*1e-6

def calculate_ops_metrics(net, input_shape):
    handles = []
    def recurs(net):
        for mod in net.children():
            if hasattr(mod, 'edge_layer'):
                handles.append(mod.register_forward_hook(total_metrics_hook))
            else:
                recurs(mod)
    recurs(net)
    dummy_input = torch.randn(input_shape, device='cuda')
    with torch.no_grad():
        net(dummy_input)
    for handle in handles:
        handle.remove()

def calculate_min(net):
    def recurs(net):
        for mod in net.children():
            if isinstance(mod, Edge):
                mod.__min_ops = min([layer.__total_ops for layer in mod.ops])
                mod.__min_params = min([layer.__total_params for layer in mod.ops])
                mod.__min_latency = min([layer.__total_latency for layer in mod.ops])
            else:
                recurs(mod)
    recurs(net)
            