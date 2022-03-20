from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from .operations import OperationsConstructor, FpOperationsConstructor
from .operations import EvalSum
from operator import itemgetter


class EvalEdge(nn.Module):
    def __init__(self, C, stride, ops_idx, cell_type, affine, binary=True, padding_mode='zeros',jit=False,dropout2d=0.1):
        super(EvalEdge, self).__init__()
        ops_constructor = OperationsConstructor if binary else FpOperationsConstructor
        self.ops = nn.ModuleList()
        self.ops_idx=ops_idx
        primitives = ops_constructor.get_primitives(cell_type)
        operations = ops_constructor.get_ops(cell_type)
        for i, primitive in enumerate(primitives):
            if i in ops_idx:
                self.ops.append(operations[primitive](C=C, stride=stride, affine=affine, padding_mode=padding_mode, jit=jit,dropout2d=dropout2d))
                self.ops[-1].edge_layer=True
                #print(self.ops[-1])
        self.C = C
        self.stride = stride
        self.sum = EvalSum()

    def forward(self, x):
        if len(self.ops_idx)>1:
            return self.sum([op(x) for op in self.ops])
        return self.ops[0](x)