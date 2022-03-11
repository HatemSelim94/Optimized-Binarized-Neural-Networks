from pip import main
import torch
import torch.nn as nn
import numpy as np 
import sys
sys.path.append('search/darts/')
from architecture.networks.edge import Edge


def total_metrics_hook(mod, input, output):
    total_ops = [0]
    def recures(main_layer):
        for child in main_layer.children():
            if hasattr(child, '__ops'):
                total_ops[0] += child.__ops
            recures(child)
    recures(mod)
    mod.__total_ops = total_ops[0]

def calculate_layers_metrics(net):
    handles = []
    def recurs(net):
        for mod in net.children():
            if hasattr(mod, 'edge_layer'):
                handles.append(mod.register_forward_hook(mod, total_metrics_hook))
        recurs(mod)
            