from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from .operations import OperationsConstructor, FpOperationsConstructor
from .operations import Sum
from operator import itemgetter


class Edge(nn.Module):
    def __init__(self, C, stride, ops_num, cell_type, affine, binary=True, objs=None, padding_mode='zeros',jit=False,dropout2d=0.1,binarization=1, activation='htanh'):
        super(Edge, self).__init__()
        ops_constructor = OperationsConstructor if binary else FpOperationsConstructor
        self.ops = nn.ModuleList()
        primitives = ops_constructor.get_primitives(cell_type)
        operations = ops_constructor.get_ops(cell_type)
        for i, primitive in enumerate(primitives):
            if i < ops_num:
                self.ops.append(operations[primitive](C, stride, affine=affine, padding_mode=padding_mode, jit=jit,dropout2d=dropout2d,binarization=binarization, activation=activation))
                self.ops[-1].edge_layer=True
                #print(self.ops[-1])
        self.C = C
        self.stride = stride
        self.sum = Sum()
        self.objs = objs
        #self.set_mins(dummy_input, objs)

    def forward(self, x, weights=None, idx=None):
        if weights is None:
            if len(idx) ==1:
                return self.op[idx[0]](x)
            return self.sum([op(x) for id, op in enumerate(self.ops) if id in idx])
        else:
            if self.objs is None:
                if idx is None:
                    return self.sum([w*op(x) for w, op in zip(weights, self.ops)])
                else:
                    return self.sum([w*op(x) for id, (w, op) in enumerate(zip(weights, self.ops)) if id in idx])

    def forward_ops(self, weights, idx=None):
        if self.objs is None:
            if idx is None:
                return self.sum(w*(getattr(op,'__total_ops') - getattr(self, '__min_ops'))for w, op in zip(weights, self.ops))
            else:
                return self.sum(w*(getattr(op,'__total_ops') - getattr(self, '__min_ops')) for id, (w, op) in enumerate(zip(weights, self.ops)) if id in idx)

    def forward_params(self, weights, idx=None):
        if self.objs is None:
            if idx is None:
                return self.sum(w*(getattr(op,'__total_params') - getattr(self, '__min_params')) for w, op in zip(weights, self.ops))
            else:
                return self.sum(w*(getattr(op,'__total_params') - getattr(self, '__min_params')) for id, (w, op) in enumerate(zip(weights, self.ops)) if id in idx)
    
    def forward_latency(self, weights, idx=None):
        if self.objs is None:
            if idx is None:
                return self.sum(w*(getattr(op,'__total_latency') - getattr(self, '__min_latency')) for w, op in zip(weights, self.ops))
            else:
                return self.sum(w*(getattr(op,'__total_latency') - getattr(self, '__min_latency')) for id, (w, op) in enumerate(zip(weights, self.ops)) if id in idx)