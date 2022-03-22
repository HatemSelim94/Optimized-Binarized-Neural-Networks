import torch
import torch.nn as nn
import torch.nn.functional as F

from .operations.search_operations import BinConvbn1x1, BinConvT1x1, ConvBn, BasicBinConv1x1, BinDilConv3x3

from .operations import Sum, Preprocess, FpPreprocess,Cat, Bilinear
from .edge import Edge


class NCell(nn.Module):
    def __init__(self,C, C_prev_prev, C_prev, prev_cell_type,init_alphas=None,stride=1, edges_num=2, ops_num=5, node_num=4,binary=True, affine=False,padding_mode='zeros',jit=False,dropout2d=0.1, binarization=1, activation='htanh'):
        super(NCell, self).__init__()
        self.edges_num = edges_num
        self.ops_num = ops_num
        self.node_num = node_num
        self.binary = binary
        preprocess = Preprocess if binary else FpPreprocess
        self.preprocess0 = preprocess.operations(prev_cell_type,0,C_prev_prev, C, affine, padding_mode=padding_mode,jit= jit,dropout2d=dropout2d, binarization=binarization,activation=activation)
        self.preprocess1 = preprocess.operations(prev_cell_type,1,C_prev, C, affine, padding_mode=padding_mode,jit= jit,dropout2d=dropout2d, binarization=binarization,activation=activation)
        self.edges = nn.ModuleList()

        for _ in range(node_num):
            for _ in range(edges_num):
                self.edges.append(Edge(C, stride, ops_num, 'n', affine, self.binary))
        
        self._init_alphas(init_alphas)
        self.sum = Sum()
        self.cat = Cat()

    def forward(self, input0, input1,  skip_input):
        s0 = self.preprocess0(input0)
        s1 = self.preprocess1(input1)

        states = [s0, s1]
        #print(states[-2].shape)
        offset = 0
        for i in range(self.node_num):
            s = self.sum([self.edges[offset+j](h, F.softmax(self.alphas[offset+j], dim=-1)) for j, h in enumerate(states[-2:])]) #offset +j = 2
            offset += 2
            states.append(s)
        states.append(skip_input)
        return self.cat(states[-(self.node_num+1):]), skip_input # channels number is multiplier "4" * C "C_curr"
    
    def _init_alphas(self, alphas=None):
        if alphas is None:
            self.alphas = torch.empty((self.node_num*self.edges_num, self.ops_num), requires_grad=True, device='cuda')
            nn.init.constant_(self.alphas, 1/self.ops_num)
        else:
            self.alphas = alphas
    
    def forward_ops(self):
        offset = 0
        loss = 0
        for _ in range(self.node_num):
            loss += sum(self.edges[offset+j].forward_ops(F.softmax(self.alphas[offset+j], dim=-1)) for j in range(2)) #offset +j = 2
            offset += 2
        return loss
    
    def forward_latency(self):
        offset = 0
        loss = 0
        for _ in range(self.node_num):
            loss += sum(self.edges[offset+j].forward_latency(F.softmax(self.alphas[offset+j], dim=-1)) for j in range(2)) #offset +j = 2
            offset += 2
        return loss
    
    def forward_params(self):
        offset = 0
        loss = 0
        for _ in range(self.node_num):
            loss += sum(self.edges[offset+j].forward_params(F.softmax(self.alphas[offset+j], dim=-1)) for j in range(2)) #offset +j = 2
            offset += 2
        return loss
    
    def binarize_weight(self, state=True):
        def recurs(net,state):
            for mod in net.children():
                if hasattr(mod,'binarized_weight'):
                    setattr(mod,'binarized_weight',state)
                else:
                    recurs(mod, state)
        recurs(self, state)
    
    def binarize_input(self, state=True):
        def recurs(net,state):
            for mod in net.children():
                if hasattr(mod,'binarized_input'):
                    setattr(mod,'binarized_input',state)
                else:
                    recurs(mod, state)
        recurs(self, state)
        

class RCell(nn.Module):
    def __init__(self,C, C_prev_prev, C_prev, prev_cell_type,init_alphas=None,skip_channels=64,edges_num=2, ops_num=5, node_num=4,binary=True, affine=False, padding_mode='zeros',jit=False,dropout2d=0.1, binarization=1,activation='htanh'):
        super(RCell, self).__init__()
        self.edges_num = edges_num
        self.ops_num = ops_num
        self.node_num = node_num
        self.binary = binary
        preprocess = Preprocess if binary else FpPreprocess
        self.preprocess0 = preprocess.operations(prev_cell_type,0,C_prev_prev, C, affine)
        self.preprocess1 = preprocess.operations(prev_cell_type,1,C_prev, C, affine)
        self.edges = nn.ModuleList()
        self.C = C

        self.preprocess_skip = preprocess.skip('r',(skip_channels, affine, 2))

        for n in range(node_num):
            for e in range(edges_num):
                stride = 2 if n+e <= 1 else 1
                self.edges.append(Edge(C, stride, ops_num,'r', affine, binary, padding_mode=padding_mode,jit=jit,dropout2d=dropout2d, binarization=binarization,activation=activation))
        
        self._init_alphas(init_alphas)
        self.sum = Sum()
        self.cat = Cat()

    def forward(self, input0, input1, skip_input):
        #_, c,_,_ = input1.shape
        s0 = self.preprocess0(input0)
        s1 = self.preprocess1(input1)
        #print(s0.shape, s1.shape)

        states = [s0, s1]
        offset = 0
        for i in range(self.node_num):
            s = self.sum([self.edges[offset+j](h, F.softmax(self.alphas[offset+j], dim=-1)) for j, h in enumerate(states[-2:])]) #offset +j = 2
            offset += 2
            states.append(s)
        prp = self.preprocess_skip(skip_input)
        #print(prp.shape)
        states.append(prp)
        return self.cat(states[-(self.node_num+1):]), prp # channels number is multiplier "4" * C "C_curr"
    
    def _init_alphas(self, alphas=None):
        if alphas is None:
            self.alphas = torch.empty((self.node_num*self.edges_num, self.ops_num), requires_grad=True, device='cuda')
            nn.init.constant_(self.alphas, 1/self.ops_num)
        else:
            self.alphas = alphas

    def forward_ops(self):
        offset = 0
        loss = 0
        for _ in range(self.node_num):
            loss += sum(self.edges[offset+j].forward_ops(F.softmax(self.alphas[offset+j], dim=-1)) for j in range(2)) #offset +j = 2
            offset += 2
        return loss
    
    def forward_latency(self):
        offset = 0
        loss = 0
        for _ in range(self.node_num):
            loss += sum(self.edges[offset+j].forward_latency(F.softmax(self.alphas[offset+j], dim=-1)) for j in range(2)) #offset +j = 2
            offset += 2
        return loss
    
    def forward_params(self):
        offset = 0
        loss = 0
        for _ in range(self.node_num):
            loss += sum(self.edges[offset+j].forward_params(F.softmax(self.alphas[offset+j], dim=-1)) for j in range(2)) #offset +j = 2
            offset += 2
        return loss
    
    def binarize_weight(self, state=True):
        def recurs(net,state):
            for mod in net.children():
                if hasattr(mod,'binarized_weight'):
                    setattr(mod,'binarized_weight',state)
                else:
                    recurs(mod, state)
        recurs(self, state)
    
    def binarize_input(self, state=True):
        def recurs(net,state):
            for mod in net.children():
                if hasattr(mod,'binarized_input'):
                    setattr(mod,'binarized_input',state)
                else:
                    recurs(mod, state)
        recurs(self, state)

class UCell(nn.Module):
    def __init__(self,C, C_prev_prev, C_prev, prev_cell_type,init_alphas=None,edges_num=2, ops_num=5, node_num=4,binary=True, affine=False,padding_mode='zeros', jit=False,dropout2d=0.1, binarization=1,activation='htanh') :
        super(UCell, self).__init__()
        self.edges_num = edges_num
        self.ops_num = ops_num
        self.node_num = node_num
        prprocess = Preprocess if binary else FpPreprocess
        self.preprocess0 = prprocess.operations(prev_cell_type,0,C_prev_prev, C, affine, padding_mode=padding_mode, jit=jit,dropout2d=dropout2d, binarization=binarization,activation=activation)
        self.preprocess1 = prprocess.operations(prev_cell_type,1,C_prev, C, affine, padding_mode=padding_mode, jit=jit,dropout2d=dropout2d, binarization=binarization,activation=activation)
        self.edges = nn.ModuleList()
        self.binary = binary
        self.preprocess_skip = prprocess.skip('u',(2,))

        for n in range(node_num):
            for e in range(edges_num):
                stride = 2 if n+e <= 1 else 1
                self.edges.append(Edge(C, stride, ops_num, 'u', affine, binary,padding_mode=padding_mode,jit=jit, binarization=binarization,activation=activation))
        
        self._init_alphas(init_alphas)
        self.sum = Sum()
        self.cat = Cat()

    def forward(self, input0, input1, skip_input):
        s0 = self.preprocess0(input0)
        s1 = self.preprocess1(input1)
        #print(s0.shape, s1.shape)
        states = [s0, s1]
        offset = 0
        for i in range(self.node_num):
            s = self.sum([self.edges[offset+j](h, F.softmax(self.alphas[offset+j], dim=-1)) for j, h in enumerate(states[-2:])]) #offset +j = 2
            offset += 2
            states.append(s)
        prp = self.preprocess_skip(skip_input)
        #print(prp.shape)
        states.append(prp)
        #states.append(input1)
        return self.cat(states[-(self.node_num+1):]), prp # channels number is multiplier "4" * C "C_curr"
    
    def _init_alphas(self, alphas=None):
        if alphas is None:
            self.alphas = torch.empty((self.node_num*self.edges_num, self.ops_num), requires_grad=True, device='cuda')
            nn.init.constant_(self.alphas, 0.005)
        else:
            self.alphas = alphas
    
    def forward_ops(self):
        offset = 0
        loss = 0
        for _ in range(self.node_num):
            loss += sum(self.edges[offset+j].forward_ops(F.softmax(self.alphas[offset+j], dim=-1)) for j in range(2)) #offset +j = 2
            offset += 2
        return loss
    
    def forward_latency(self):
        offset = 0
        loss = 0
        for _ in range(self.node_num):
            loss += sum(self.edges[offset+j].forward_latency(F.softmax(self.alphas[offset+j], dim=-1)) for j in range(2)) #offset +j = 2
            offset += 2
        return loss
    
    def forward_params(self):
        offset = 0
        loss = 0
        for _ in range(self.node_num):
            loss += sum(self.edges[offset+j].forward_params(F.softmax(self.alphas[offset+j], dim=-1)) for j in range(2)) #offset +j = 2
            offset += 2
        return loss
    
    def binarize_weight(self, state=True):
        def recurs(net,state):
            for mod in net.children():
                if hasattr(mod,'binarized_weight'):
                    setattr(mod,'binarized_weight',state)
                else:
                    recurs(mod, state)
        recurs(self, state)
    
    def binarize_input(self, state=True):
        def recurs(net,state):
            for mod in net.children():
                if hasattr(mod,'binarized_input'):
                    setattr(mod,'binarized_input',state)
                else:
                    recurs(mod, state)
        recurs(self, state)

###################################################################
# no skip
class NCellNSkip(nn.Module):
    def __init__(self,C, C_prev_prev, C_prev, prev_cell_type,init_alphas=None,stride=1, edges_num=2, ops_num=5, node_num=4,binary=True, affine=False,padding_mode='zeros',jit=False,dropout2d=0.1, binarization=1, activation='htanh'):
        super(NCellNSkip, self).__init__()
        self.edges_num = edges_num
        self.ops_num = ops_num
        self.node_num = node_num
        self.binary = binary
        preprocess = Preprocess if binary else FpPreprocess
        self.preprocess0 = preprocess.operations(prev_cell_type,0,C_prev_prev, C, affine, padding_mode=padding_mode,jit= jit,dropout2d=dropout2d, binarization=binarization,activation=activation)
        self.preprocess1 = preprocess.operations(prev_cell_type,1,C_prev, C, affine, padding_mode=padding_mode,jit= jit,dropout2d=dropout2d, binarization=binarization,activation=activation)
        self.edges = nn.ModuleList()

        for _ in range(node_num):
            for _ in range(edges_num):
                self.edges.append(Edge(C, stride, ops_num, 'n', affine, self.binary, padding_mode=padding_mode, jit=jit,dropout2d=dropout2d, binarization=binarization,activation=activation))
        
        self._init_alphas(init_alphas)
        self.sum = Sum()
        self.cat = Cat()

    def forward(self, input0, input1):
        s0 = self.preprocess0(input0)
        s1 = self.preprocess1(input1)

        states = [s0, s1]
        #print(states[-2].shape)
        offset = 0
        for i in range(self.node_num):
            s = self.sum([self.edges[offset+j](h, F.softmax(self.alphas[offset+j], dim=-1)) for j, h in enumerate(states[-2:])]) #offset +j = 2
            offset += 2
            states.append(s)
        return self.cat(states[-(self.node_num):]) # channels number is multiplier "4" * C "C_curr"
    
    def _init_alphas(self, alphas=None):
        if alphas is None:
            self.alphas = torch.empty((self.node_num*self.edges_num, self.ops_num), requires_grad=True, device='cuda')
            nn.init.constant_(self.alphas, 1/self.ops_num)
        else:
            self.alphas = alphas
    
    def forward_ops(self):
        offset = 0
        loss = 0
        for _ in range(self.node_num):
            loss += sum(self.edges[offset+j].forward_ops(F.softmax(self.alphas[offset+j], dim=-1)) for j in range(2)) #offset +j = 2
            offset += 2
        return loss
    
    def forward_latency(self):
        offset = 0
        loss = 0
        for _ in range(self.node_num):
            loss += sum(self.edges[offset+j].forward_latency(F.softmax(self.alphas[offset+j], dim=-1)) for j in range(2)) #offset +j = 2
            offset += 2
        return loss
    
    def forward_params(self):
        offset = 0
        loss = 0
        for _ in range(self.node_num):
            loss += sum(self.edges[offset+j].forward_params(F.softmax(self.alphas[offset+j], dim=-1)) for j in range(2)) #offset +j = 2
            offset += 2
        return loss
    
    def binarize_weight(self, state=True):
        def recurs(net,state):
            for mod in net.children():
                if hasattr(mod,'binarized_weight'):
                    setattr(mod,'binarized_weight',state)
                else:
                    recurs(mod, state)
        recurs(self, state)
    
    def binarize_input(self, state=True):
        def recurs(net,state):
            for mod in net.children():
                if hasattr(mod,'binarized_input'):
                    setattr(mod,'binarized_input',state)
                else:
                    recurs(mod, state)
        recurs(self, state)
        

class RCellNSkip(nn.Module):
    def __init__(self,C, C_prev_prev, C_prev, prev_cell_type,init_alphas=None, skip_channels=0,edges_num=2, ops_num=5, node_num=4,binary=True, affine=False, padding_mode='zeros',jit=False,dropout2d=0.1, binarization=1,activation='htanh'):
        super(RCellNSkip, self).__init__()
        self.edges_num = edges_num
        self.ops_num = ops_num
        self.node_num = node_num
        self.binary = binary
        preprocess = Preprocess if binary else FpPreprocess
        self.preprocess0 = preprocess.operations(prev_cell_type,0,C_prev_prev, C, affine,padding_mode=padding_mode, jit=jit,dropout2d=dropout2d, binarization=binarization,activation=activation)
        self.preprocess1 = preprocess.operations(prev_cell_type,1,C_prev, C, affine, padding_mode=padding_mode, jit=jit,dropout2d=dropout2d, binarization=binarization,activation=activation)
        self.edges = nn.ModuleList()
        self.C = C


        for n in range(node_num):
            for e in range(edges_num):
                stride = 2 if n+e <= 1 else 1
                self.edges.append(Edge(C, stride, ops_num,'r', affine, binary, padding_mode=padding_mode,jit=jit,dropout2d=dropout2d, binarization=binarization,activation=activation))
        
        self._init_alphas(init_alphas)
        self.sum = Sum()
        self.cat = Cat()

    def forward(self, input0, input1):
        #_, c,_,_ = input1.shape
        s0 = self.preprocess0(input0)
        s1 = self.preprocess1(input1)
        #print(s0.shape, s1.shape)

        states = [s0, s1]
        offset = 0
        for i in range(self.node_num):
            s = self.sum([self.edges[offset+j](h, F.softmax(self.alphas[offset+j], dim=-1)) for j, h in enumerate(states[-2:])]) #offset +j = 2
            offset += 2
            states.append(s)
        #print(prp.shape)
        return self.cat(states[-(self.node_num):]) # channels number is multiplier "4" * C "C_curr"
    
    def _init_alphas(self, alphas=None):
        if alphas is None:
            self.alphas = torch.empty((self.node_num*self.edges_num, self.ops_num), requires_grad=True, device='cuda')
            nn.init.constant_(self.alphas, 1/self.ops_num)
        else:
            self.alphas = alphas

    def forward_ops(self):
        offset = 0
        loss = 0
        for _ in range(self.node_num):
            loss += sum(self.edges[offset+j].forward_ops(F.softmax(self.alphas[offset+j], dim=-1)) for j in range(2)) #offset +j = 2
            offset += 2
        return loss
    
    def forward_latency(self):
        offset = 0
        loss = 0
        for _ in range(self.node_num):
            loss += sum(self.edges[offset+j].forward_latency(F.softmax(self.alphas[offset+j], dim=-1)) for j in range(2)) #offset +j = 2
            offset += 2
        return loss
    
    def forward_params(self):
        offset = 0
        loss = 0
        for _ in range(self.node_num):
            loss += sum(self.edges[offset+j].forward_params(F.softmax(self.alphas[offset+j], dim=-1)) for j in range(2)) #offset +j = 2
            offset += 2
        return loss
    
    def binarize_weight(self, state=True):
        def recurs(net,state):
            for mod in net.children():
                if hasattr(mod,'binarized_weight'):
                    setattr(mod,'binarized_weight',state)
                else:
                    recurs(mod, state)
        recurs(self, state)
    
    def binarize_input(self, state=True):
        def recurs(net,state):
            for mod in net.children():
                if hasattr(mod,'binarized_input'):
                    setattr(mod,'binarized_input',state)
                else:
                    recurs(mod, state)
        recurs(self, state)


class UCellNSkip(nn.Module):
    def __init__(self,C, C_prev_prev, C_prev, prev_cell_type,init_alphas=None,edges_num=2, ops_num=5, node_num=4,binary=True, affine=False,padding_mode='zeros', jit=False,dropout2d=0.1, binarization=1,activation='htanh') :
        super(UCellNSkip, self).__init__()
        self.edges_num = edges_num
        self.ops_num = ops_num
        self.node_num = node_num
        prprocess = Preprocess if binary else FpPreprocess
        self.preprocess0 = prprocess.operations(prev_cell_type,0,C_prev_prev, C, affine, padding_mode=padding_mode, jit=jit,dropout2d=dropout2d, binarization=binarization,activation=activation)
        self.preprocess1 = prprocess.operations(prev_cell_type,1,C_prev, C, affine, padding_mode=padding_mode, jit=jit,dropout2d=dropout2d, binarization=binarization,activation=activation)
        self.edges = nn.ModuleList()
        self.binary = binary

        for n in range(node_num):
            for e in range(edges_num):
                stride = 2 if n+e <= 1 else 1
                self.edges.append(Edge(C, stride, ops_num, 'u', affine, binary,padding_mode=padding_mode,jit=jit, binarization=binarization,activation=activation))
        
        self._init_alphas(init_alphas)
        self.sum = Sum()
        self.cat = Cat()

    def forward(self, input0, input1):
        s0 = self.preprocess0(input0)
        s1 = self.preprocess1(input1)
        #print(s0.shape, s1.shape)
        states = [s0, s1]
        offset = 0
        for i in range(self.node_num):
            s = self.sum([self.edges[offset+j](h, F.softmax(self.alphas[offset+j], dim=-1)) for j, h in enumerate(states[-2:])]) #offset +j = 2
            offset += 2
            states.append(s)
        #print(prp.shape)
        #states.append(input1)
        return self.cat(states[-(self.node_num):]) # channels number is multiplier "4" * C "C_curr"
    
    def _init_alphas(self, alphas=None):
        if alphas is None:
            self.alphas = torch.empty((self.node_num*self.edges_num, self.ops_num), requires_grad=True, device='cuda')
            nn.init.constant_(self.alphas, 0.005)
        else:
            self.alphas = alphas
    
    def forward_ops(self):
        offset = 0
        loss = 0
        for _ in range(self.node_num):
            loss += sum(self.edges[offset+j].forward_ops(F.softmax(self.alphas[offset+j], dim=-1)) for j in range(2)) #offset +j = 2
            offset += 2
        return loss
    
    def forward_latency(self):
        offset = 0
        loss = 0
        for _ in range(self.node_num):
            loss += sum(self.edges[offset+j].forward_latency(F.softmax(self.alphas[offset+j], dim=-1)) for j in range(2)) #offset +j = 2
            offset += 2
        return loss
    
    def forward_params(self):
        offset = 0
        loss = 0
        for _ in range(self.node_num):
            loss += sum(self.edges[offset+j].forward_params(F.softmax(self.alphas[offset+j], dim=-1)) for j in range(2)) #offset +j = 2
            offset += 2
        return loss
    
    def binarize_weight(self, state=True):
        def recurs(net,state):
            for mod in net.children():
                if hasattr(mod,'binarized_weight'):
                    setattr(mod,'binarized_weight',state)
                else:
                    recurs(mod, state)
        recurs(self, state)
    
    def binarize_input(self, state=True):
        def recurs(net,state):
            for mod in net.children():
                if hasattr(mod,'binarized_input'):
                    setattr(mod,'binarized_input',state)
                else:
                    recurs(mod, state)
        recurs(self, state)

###################################################################
# old ver Darts
class NCellNSkipOld(nn.Module):
    def __init__(self,C, C_prev_prev, C_prev, prev_cell_type,init_alphas=None,stride=1, edges_num=2, ops_num=5, node_num=4,binary=True, affine=False,padding_mode='zeros',jit=False,dropout2d=0.1, binarization=1, activation='htanh'):
        super(NCellNSkipOld, self).__init__()
        self.edges_num = edges_num
        self.ops_num = ops_num
        self.node_num = node_num
        self.binary = binary
        self.total_edge_num = sum([edges_num+i for i in range(self.node_num)])
        preprocess = Preprocess if binary else FpPreprocess
        self.preprocess0 = preprocess.operations(prev_cell_type,0,C_prev_prev, C, affine, padding_mode=padding_mode,jit= jit,dropout2d=dropout2d, binarization=binarization,activation=activation)
        self.preprocess1 = preprocess.operations(prev_cell_type,1,C_prev, C, affine, padding_mode=padding_mode,jit= jit,dropout2d=dropout2d, binarization=binarization,activation=activation)
        self.edges = nn.ModuleList()

        for n in range(node_num):
            for _ in range(edges_num+n):
                self.edges.append(Edge(C, stride, ops_num, 'n', affine, self.binary, padding_mode=padding_mode, jit=jit,dropout2d=dropout2d, binarization=binarization,activation=activation))
        
        self._init_alphas(init_alphas)
        self.sum = Sum()
        self.cat = Cat()

    def forward(self, input0, input1):
        s0 = self.preprocess0(input0)
        s1 = self.preprocess1(input1)

        states = [s0, s1]
        #print(states[-2].shape)
        offset = 0
        for _ in range(self.node_num):
            s = self.sum([self.edges[offset+j](h, F.softmax(self.alphas[offset+j], dim=-1)) for j, h in enumerate(states)]) #offset +j = 2
            offset += len(states)
            states.append(s)
        return self.cat(states[-(self.node_num):]) # channels number is multiplier "4" * C "C_curr"
    
    def _init_alphas(self, alphas=None):
        if alphas is None:
            self.alphas = torch.empty((self.total_edge_num, self.ops_num), requires_grad=True, device='cuda')
            nn.init.constant_(self.alphas, 1/self.ops_num)
        else:
            self.alphas = alphas
    
    def forward_ops(self):
        offset = 0
        loss = 0
        states = 2
        for _ in range(self.node_num):
            loss += sum(self.edges[offset+j].forward_ops(F.softmax(self.alphas[offset+j], dim=-1)) for j in range(states)) #offset +j = 2
            offset += states
            states += 1
        return loss
    
    def forward_latency(self):
        offset = 0
        loss = 0
        states = 2
        for i in range(self.node_num):
            loss += sum(self.edges[offset+j].forward_latency(F.softmax(self.alphas[offset+j], dim=-1)) for j in range(states)) #offset +j = 2
            offset += states
            states += 1
        return loss
    
    def forward_params(self):
        offset = 0
        loss = 0
        states = 2
        for i in range(self.node_num):
            loss += sum(self.edges[offset+j].forward_params(F.softmax(self.alphas[offset+j], dim=-1)) for j in range(states)) #offset +j = 2
            offset += states
            states += 1
        return loss
    
    def binarize_weight(self, state=True):
        def recurs(net,state):
            for mod in net.children():
                if hasattr(mod,'binarized_weight'):
                    setattr(mod,'binarized_weight',state)
                else:
                    recurs(mod, state)
        recurs(self, state)
    
    def binarize_input(self, state=True):
        def recurs(net,state):
            for mod in net.children():
                if hasattr(mod,'binarized_input'):
                    setattr(mod,'binarized_input',state)
                else:
                    recurs(mod, state)
        recurs(self, state)
        

class RCellNSkipOld(nn.Module):
    def __init__(self,C, C_prev_prev, C_prev, prev_cell_type,init_alphas=None, skip_channels=0,edges_num=2, ops_num=5, node_num=4,binary=True, affine=False, padding_mode='zeros',jit=False,dropout2d=0.1, binarization=1,activation='htanh'):
        super(RCellNSkipOld, self).__init__()
        self.edges_num = edges_num
        self.ops_num = ops_num
        self.node_num = node_num
        self.binary = binary
        self.total_edge_num = sum([edges_num+i for i in range(self.node_num)])
        preprocess = Preprocess if binary else FpPreprocess
        self.preprocess0 = preprocess.operations(prev_cell_type,0,C_prev_prev, C, affine,padding_mode=padding_mode, jit=jit,dropout2d=dropout2d, binarization=binarization,activation=activation)
        self.preprocess1 = preprocess.operations(prev_cell_type,1,C_prev, C, affine, padding_mode=padding_mode, jit=jit,dropout2d=dropout2d, binarization=binarization,activation=activation)
        self.edges = nn.ModuleList()
        self.C = C


        for n in range(node_num):
            for e in range(edges_num+n):
                stride = 2 if e < 2 else 1
                self.edges.append(Edge(C, stride, ops_num,'r', affine, binary, padding_mode=padding_mode,jit=jit,dropout2d=dropout2d, binarization=binarization,activation=activation))
        
        self._init_alphas(init_alphas)
        self.sum = Sum()
        self.cat = Cat()

    def forward(self, input0, input1):
        #_, c,_,_ = input1.shape
        s0 = self.preprocess0(input0)
        s1 = self.preprocess1(input1)
        #print(s0.shape, s1.shape)

        states = [s0, s1]
        offset = 0
        for _ in range(self.node_num):
            s = self.sum([self.edges[offset+j](h, F.softmax(self.alphas[offset+j], dim=-1)) for j, h in enumerate(states)]) #offset +j = 2
            offset += len(states)
            states.append(s)
        #print(prp.shape)
        return self.cat(states[-(self.node_num):]) # channels number is multiplier "4" * C "C_curr"
    
    def _init_alphas(self, alphas=None):
        if alphas is None:
            self.alphas = torch.empty((self.total_edge_num, self.ops_num), requires_grad=True, device='cuda')
            nn.init.constant_(self.alphas, 1/self.ops_num)
        else:
            self.alphas = alphas
    
    def forward_ops(self):
        offset = 0
        loss = 0
        states = 2
        for _ in range(self.node_num):
            loss += sum(self.edges[offset+j].forward_ops(F.softmax(self.alphas[offset+j], dim=-1)) for j in range(states)) #offset +j = 2
            offset += states
            states += 1
        return loss
    
    def forward_latency(self):
        offset = 0
        loss = 0
        states = 2
        for _ in range(self.node_num):
            loss += sum(self.edges[offset+j].forward_latency(F.softmax(self.alphas[offset+j], dim=-1)) for j in range(states)) #offset +j = 2
            offset += states
            states += 1
        return loss
    
    def forward_params(self):
        offset = 0
        loss = 0
        states = 2
        for _ in range(self.node_num):
            loss += sum(self.edges[offset+j].forward_params(F.softmax(self.alphas[offset+j], dim=-1)) for j in range(states)) #offset +j = 2
            offset += states
            states += 1
        return loss
    
    def binarize_weight(self, state=True):
        def recurs(net,state):
            for mod in net.children():
                if hasattr(mod,'binarized_weight'):
                    setattr(mod,'binarized_weight',state)
                else:
                    recurs(mod, state)
        recurs(self, state)
    
    def binarize_input(self, state=True):
        def recurs(net,state):
            for mod in net.children():
                if hasattr(mod,'binarized_input'):
                    setattr(mod,'binarized_input',state)
                else:
                    recurs(mod, state)
        recurs(self, state)


class UCellNSkipOld(nn.Module):
    def __init__(self,C, C_prev_prev, C_prev, prev_cell_type,init_alphas=None,edges_num=2, ops_num=5, node_num=4,binary=True, affine=False,padding_mode='zeros', jit=False,dropout2d=0.1, binarization=1,activation='htanh') :
        super(UCellNSkipOld, self).__init__()
        self.edges_num = edges_num
        self.ops_num = ops_num
        self.node_num = node_num
        self.total_edge_num = sum([edges_num+i for i in range(self.node_num)])
        prprocess = Preprocess if binary else FpPreprocess
        self.preprocess0 = prprocess.operations(prev_cell_type,0,C_prev_prev, C, affine, padding_mode=padding_mode, jit=jit,dropout2d=dropout2d, binarization=binarization,activation=activation)
        self.preprocess1 = prprocess.operations(prev_cell_type,1,C_prev, C, affine, padding_mode=padding_mode, jit=jit,dropout2d=dropout2d, binarization=binarization,activation=activation)
        self.edges = nn.ModuleList()
        self.binary = binary

        for n in range(node_num):
            for e in range(edges_num+n):
                stride = 2 if e < 2 else 1
                self.edges.append(Edge(C, stride, ops_num, 'u', affine, binary,padding_mode=padding_mode,jit=jit, binarization=binarization,activation=activation))
        
        self._init_alphas(init_alphas)
        self.sum = Sum()
        self.cat = Cat()

    def forward(self, input0, input1):
        s0 = self.preprocess0(input0)
        s1 = self.preprocess1(input1)
        #print(s0.shape, s1.shape)
        states = [s0, s1]
        offset = 0
        for _ in range(self.node_num):
            s = self.sum([self.edges[offset+j](h, F.softmax(self.alphas[offset+j], dim=-1)) for j, h in enumerate(states)]) #offset +j = 2
            offset += len(states)
            states.append(s)
        #print(prp.shape)
        #states.append(input1)
        return self.cat(states[-(self.node_num):]) # channels number is multiplier "4" * C "C_curr"
    
    def _init_alphas(self, alphas=None):
        if alphas is None:
            self.alphas = torch.empty((self.total_edge_num, self.ops_num), requires_grad=True, device='cuda')
            nn.init.constant_(self.alphas, 1/self.ops_num)
        else:
            self.alphas = alphas
    
    def forward_ops(self):
        offset = 0
        loss = 0
        states = 2
        for i in range(self.node_num):
            loss += sum(self.edges[offset+j].forward_ops(F.softmax(self.alphas[offset+j], dim=-1)) for j in range(states)) #offset +j = 2
            offset += states
            states += 1
        return loss
    
    def forward_latency(self):
        offset = 0
        loss = 0
        states = 2
        for i in range(self.node_num):
            loss += sum(self.edges[offset+j].forward_latency(F.softmax(self.alphas[offset+j], dim=-1)) for j in range(states)) #offset +j = 2
            offset += states
            states += 1
        return loss
    
    def forward_params(self):
        offset = 0
        loss = 0
        states = 2
        for i in range(self.node_num):
            loss += sum(self.edges[offset+j].forward_params(F.softmax(self.alphas[offset+j], dim=-1)) for j in range(states)) #offset +j = 2
            offset += states
            states += 1
        return loss
    
    def binarize_weight(self, state=True):
        def recurs(net,state):
            for mod in net.children():
                if hasattr(mod,'binarized_weight'):
                    setattr(mod,'binarized_weight',state)
                else:
                    recurs(mod, state)
        recurs(self, state)
    
    def binarize_input(self, state=True):
        def recurs(net,state):
            for mod in net.children():
                if hasattr(mod,'binarized_input'):
                    setattr(mod,'binarized_input',state)
                else:
                    recurs(mod, state)
        recurs(self, state)



###################################################################
class LastLayer(nn.Module):
    def __init__(self, in_channels, classes_num=3, binary=True, affine=False, kernel_size=3, jit = False, binarization=1):
        super(LastLayer, self).__init__() 
        if binary:
            self.layers = nn.Sequential(
                BinConvbn1x1(in_channels, classes_num, kernel_size, jit=jit, binarization=binarization)
                )
        else:
            self.layers = nn.Sequential(
                nn.Conv2d(in_channels, classes_num, kernel_size),
                nn.BatchNorm2d(classes_num, affine=affine)
            )
    def forward(self, x):
        x = self.layers(x)
        return x


class Pooling(nn.Module):
    def __init__(self, in_channels, out_channels,padding_mode,jit=False, dropout2d=0.0, binarization=1):
        super(Pooling,self).__init__()
        self.adaptive_pooling = nn.AdaptiveAvgPool2d(1)
        self.conv1= BasicBinConv1x1(in_channels, out_channels,1,padding_mode=padding_mode ,jit=jit, binarization=binarization)
        #self.upsample = EvalBilinear()
        self.batchnorm = nn.BatchNorm2d(out_channels, affine=True)
    
    def forward(self, x):
        img_size = x.shape[-2:]
        x = self.adaptive_pooling(x)
        x = self.conv1(x)
        x = F.interpolate(x, size=img_size, mode='bilinear', align_corners=False)
        x = self.batchnorm(x)
        return x


class BinASPP(nn.Module):
    def __init__(self, in_channels, out_channels,padding_mode,jit, dropout2d, rates=[1,4,8,12], binarization=1,activation='htanh'):
        super(BinASPP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(Pooling(in_channels, out_channels, padding_mode,jit, dropout2d,binarization=binarization))
        self.layers.append(BinConvbn1x1(in_channels, out_channels,padding_mode=padding_mode, jit=jit, dropout2d=dropout2d, binarization=binarization, activation=activation))
        for r in rates:
            self.layers.append(BinDilConv3x3(in_channels, out_channels,stride=1, padding=r,dilation=r, padding_mode=padding_mode,jit=jit, dropout2d=dropout2d, binarization=binarization,activation=activation))
        self.sum = Sum()

    def forward(self, x):
        output = []
        for mod in self.layers:
            output.append(mod(x))
        s = torch.sum(torch.stack(output, dim=1), dim=1)
        return s

    def binarize_weight(self, state=True):
        def recurs(net,state):
            for mod in net.children():
                if hasattr(mod,'binarized_weight'):
                    setattr(mod,'binarized_weight',state)
                else:
                    recurs(mod, state)
        recurs(self, state)
    
    def binarize_input(self, state=True):
        def recurs(net,state):
            for mod in net.children():
                if hasattr(mod,'binarized_input'):
                    setattr(mod,'binarized_input',state)
                else:
                    recurs(mod, state)
        recurs(self, state)      

if __name__ == '__main__':
    c = NCell(20, 10, 20, 1, 'r')

  

            
