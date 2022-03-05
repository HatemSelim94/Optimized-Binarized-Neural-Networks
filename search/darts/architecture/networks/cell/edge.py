from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from .operations import OperationsConstructor
from .operations import Sum
from operator import itemgetter


class Edge(nn.Module):
    def __init__(self, C, stride, ops_num, cell_type, objs=None) -> None:
        super(Edge, self).__init__()
        self.ops = nn.ModuleList()
        primitives = OperationsConstructor.get_primitives(cell_type)
        operations = OperationsConstructor.get_ops(cell_type)
        for i, primitive in enumerate(primitives):
            if i < ops_num:
                self.ops.append(operations[primitive](C, stride, affine=False))
                #print(self.ops[-1])
        self.C = C
        self.stride = stride
        self.sum = Sum()
        self.objs = objs
        #self.set_mins(dummy_input, objs)

    def forward(self, x, weights, idx=None):
        if self.objs is None:
            if idx is None:
                return self.sum(w*op(x) for w, op in zip(weights, self.ops))
            else:
                return self.sum(w*op(x) for id, (w, op) in enumerate(zip(weights, self.ops)) if id in idx)
        else: 
            output = {}
            if idx is None:
                output['images'] = self.sum(w*op(x['images']) for w, op in zip(weights, self.ops))
                if 'latency' in self.objs:
                    output['latency']= x['latency']+ self.sum(w*op.latency(x['images']) for w, op in zip(weights, self.ops))
                if 'params' in self.objs:
                    output['params']= x['params']+ self.sum(w*op.params(x['images']) for w, op in zip(weights, self.ops))
                if 'memory' in self.objs:
                    output['memory']= x['memory']+ self.sum(w*op.max_memory(x['images']) for w, op in zip(weights, self.ops))
                if 'ops' in self.objs:
                    output['ops']= x['ops']+ self.sum(w*op.ops(x['images']) for w, op in zip(weights, self.ops))
            else:
                output['images'] = self.sum(w*op(x) for id, (w, op) in enumerate(zip(weights, self.ops)) if id in idx)
            return output

    def forward_latency(self, x, weights, idx=None):
        if idx is None:
            return self.sum(w*op.latency(x) for w, op in zip(weights, self.ops))
        else:
            return self.sum(w*op.latency(x) for id, (w, op) in enumerate(zip(weights, self.ops)) if id in idx)

    def forward_memory(self, x, weights, idx=None):
        if idx is None:
            return self.sum(w*op.max_memory(x,output_only=True) for w, op in zip(weights, self.ops))
        else:
            return self.sum(w*op.max_memory(x,output_only=True) for id, (w, op) in enumerate(zip(weights, self.ops)) if id in idx)
    
    def forward_ops(self, x, weights, idx=None):
        if idx is None:
            return self.sum(w*op.ops(x,output_only=True) for w, op in zip(weights, self.ops))
        else:
            return self.sum(w*op.ops(x,output_only=True) for id, (w, op) in enumerate(zip(weights, self.ops)) if id in idx)
    
    def forward_params(self, x, weights, idx=None):
        if idx is None:
            return self.sum(w*op.params(x,output_only=True) for w, op in zip(weights, self.ops))
        else:
            return self.sum(w*op.params(x,output_only=True) for id, (w, op) in enumerate(zip(weights, self.ops)) if id in idx)
    
    def get_min_mem(self, x, idx=None):
        with torch.no_grad():
            if self.min_mem == None:
                if idx is None:
                    mems = []
                    outputs = []
                    for op in self.ops():
                        (max_mem,_) ,output = op.max_mem(x)
                        mems.append(max_mem)
                        outputs.append(output)
                    min_id, min_mem = min(enumerate(mems), key=itemgetter(1))
                    self.min_mem = min_mem
                    return outputs[min_id]
                else:
                    mems = {}
                    outputs = {}
                    for id, op in enumerate(self.ops):
                        if id in idx:
                            (max_mem,_) ,output = op.max_mem(x)
                            mems[id] = max_mem
                            outputs[id] = output
                    min_id = min(mems, key=mems.get)
                    self.min_mem = mems[min_id]
                self.mins['memroy'] = self.min_mem
            return outputs[min_id]
    
    def get_min_flops(self, x, idx= None):
        with torch.no_grad():
            if self.min_flops == None:
                if idx is None:
                    flops = []
                    outputs = []
                    for op in self.ops():
                        (ops,_) ,output = op.ops(x)
                        flops.append(ops)
                        outputs.append(output)
                    min_id, min_flops = min(enumerate(flops), key=itemgetter(1))
                    self.min_flops = min_flops
                    return outputs[min_id]
                else:
                    flops = {}
                    outputs = {}
                    for id, op in enumerate(self.ops):
                        if id in idx:
                            (max_flops,_) ,output = op.ops(x)
                            flops[id] = max_flops
                            outputs[id] = output
                    min_id = min(flops, key=flops.get)
                    self.min_flops = flops[min_id]
                self.mins['flops'] = self.min_flops
            return outputs[min_id]

    def get_min_params(self, x, idx= None):
        with torch.no_grad():
            if self.min_params == None:
                if idx is None:
                    params = []
                    outputs = []
                    for op in self.ops():
                        (param,_) ,output = op.params(x)
                        params.append(param)
                        outputs.append(output)
                    min_id, min_params = min(enumerate(params), key=itemgetter(1))
                    self.min_flops = min_params
                    return outputs[min_id]
                else:
                    params = {}
                    outputs = {}
                    for id, op in enumerate(self.ops):
                        if id in idx:
                            (max_params,_) ,output = op.params(x)
                            params[id] = max_params
                            outputs[id] = output
                    min_id = min(params, key=params.get)
                    self.min_params = params[min_id]
                self.mins['params'] = self.min_params
            return outputs[min_id]


    def get_min_latency(self, x, idx= None):
        with torch.no_grad():
            if self.min_latency == None:
                if idx is None:
                    latencies = []
                    outputs = []
                    for op in self.ops():
                        latency, output = op.latency(x)
                        latencies.append(latency)
                        outputs.append(output)
                    min_id, min_latency = min(enumerate(latencies), key=itemgetter(1))
                    self.min_latency = min_latency
                    return outputs[min_id]
                else:
                    latencies = {}
                    outputs = {}
                    for id, op in enumerate(self.ops):
                        if id in idx:
                            latency, output = op.latency(x)
                            latencies[id] = latency
                            outputs[id] = output
                    min_id = min(latencies, key=latencies.get)
                    self.min_latency = latencies[min_id]
                self.mins['params'] = self.min_latency
            return outputs[min_id]

    def set_mins(self, dummy_input):
        with torch.no_grad():
            if self.objs is not None:
                self.mins = {}
                for obj in self.objs:
                    if obj == 'params':
                        self.get_min_params(dummy_input)
                    elif obj == 'latency':
                        self.mins['latency'] = self.get_min_latency(dummy_input)
                    elif obj == 'flops':
                        self.mins['flops'] = self.get_min_flops(dummy_input)
                    elif obj == 'memory':
                        self.mins['memory'] = self.get_min_mem(dummy_input)
        



def mm(w,x, op, mins,output_dict,opts:Dict):
    output = op(x)
    output =  w*output
    if opts.get('flops', None)is not None:
        output_dict['flops'] += w*(op.flops(x.shape, output.shape)-mins['flops'])
    if opts.get('flops', None)is not None:
        output_dict['params'] += w*(op.params(x.shape, output.shape)-mins['params'])
    if opts.get('flops', None)is not None:
        output_dict['memory'] += w*(op.memory(x.shape, output.shape)-mins['memory'])
    if opts.get('flops', None)is not None:
        output_dict['latency'] += w*(op.latency(x.shape, output.shape)-mins['latency'])
    return output
    

if __name__ == '__main__':
    import torch
    e = Edge(10, 1, 5, 'n', objs=['latency','memory', 'params','ops']).cuda()
    #e = Edge(10, 1, 5, 'n')
    inp = torch.randn((1,10, 32, 32), device='cuda')
    w = torch.ones((5), requires_grad=True, device='cuda')
    torch.nn.init.constant_(w, 1/5)

    dict_inp  = {}
    dict_inp['latency'] = 0
    dict_inp['memory'] = 0
    dict_inp['params'] = 0
    dict_inp['ops'] = 0
    dict_inp['images'] = inp
    out = e(dict_inp, torch.nn.functional.softmax(w,-1))
    #out = e(inp, torch.nn.functional.softmax(w,-1), [0])
    print(out)
