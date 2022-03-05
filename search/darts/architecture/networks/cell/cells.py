import torch
import torch.nn as nn
import torch.nn.functional as F

from .operations.search_operations import BinConv1x1

from .operations import Sum, Preprocess, Cat
from .edge import Edge


class NCell(nn.Module):
    def __init__(self,C, C_prev_prev, C_prev, prev_cell_type,init_alphas=None,stride=1, edges_num=2, ops_num=5, node_num=4,binary=True, affine=False):
        super(NCell, self).__init__()
        self.edges_num = edges_num
        self.ops_num = ops_num
        self.node_num = node_num
        self.preprocess0 = Preprocess.operations(prev_cell_type,0,C_prev_prev, C, affine)
        self.preprocess1 = Preprocess.operations(prev_cell_type,1,C_prev, C, affine)
        self.edges = nn.ModuleList()

        for _ in range(node_num):
            for _ in range(edges_num):
                self.edges.append(Edge(C, stride, ops_num, cell_type='n'))
        
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
            s = self.sum([self.edges[offset+j](h, F.softmax(self.alphas[i+j], dim=-1)) for j, h in enumerate(states[-2:])]) #offset +j = 2
            offset += 2
            states.append(s)
        states.append(input1)
        return self.cat(states[-(self.node_num+1):]), skip_input # channels number is multiplier "4" * C "C_curr"
    
    def _init_alphas(self, alphas=None):
        if alphas is None:
            self.alphas = torch.empty((self.node_num*self.edges_num, self.ops_num), requires_grad=True, device='cuda')
            nn.init.constant_(self.alphas, 1/self.ops_num)
        else:
            self.alphas = alphas
    
    def forward_memory(self, input0, input1):
        s0 = self.preprocess0(input0)
        s1 = self.preprocess0(input1)

        states = [s0, s1]
        offset = 0

        for i in range(self.node_num):
            s = self.sum(self.edges[offset+j].forward_memory(h, F.softmax(self.alphas[i+j], dim=-1)) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)
        states.append(input1)
        return torch.cat(states[-(self.node_num+1):], dim=1) # channels number is multiplier "4" * C "C_curr"
    
    def forward_ops(self, input0, input1):
        s0 = self.preprocess0(input0)
        s1 = self.preprocess0(input1)

        states = [s0, s1]
        offset = 0

        for i in range(self.node_num):
            s = self.sum(self.edges[offset+j].forward_ops(h, F.softmax(self.alphas[i+j], dim=-1)) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)
        states.append(input1)
        return torch.cat(states[-(self.node_num+1):], dim=1) # channels number is multiplier "4" * C "C_curr"


class RCell(nn.Module):
    def __init__(self,C, C_prev_prev, C_prev, prev_cell_type,init_alphas=None,skip_channels=64,edges_num=2, ops_num=5, node_num=4,binary=True, affine=False) -> None:
        super(RCell, self).__init__()
        self.edges_num = edges_num
        self.ops_num = ops_num
        self.node_num = node_num
        self.preprocess0 = Preprocess.operations(prev_cell_type,0,C_prev_prev, C, affine)
        self.preprocess1 = Preprocess.operations(prev_cell_type,1,C_prev, C, affine)
        self.edges = nn.ModuleList()
        self.C = C

        self.preprocess_skip = Preprocess.skip('r',(skip_channels, affine, 2))

        for n in range(node_num):
            for e in range(edges_num):
                stride = 2 if n+e <= 1 else 1
                self.edges.append(Edge(C, stride, ops_num,'r'))
        
        self._init_alphas(init_alphas)
        self.sum = Sum()
        self.cat = Cat()

    def forward(self, input0, input1, skip_input):
        _, c,_,_ = input1.shape
        s0 = self.preprocess0(input0)
        s1 = self.preprocess1(input1)
        #print(s0.shape, s1.shape)

        states = [s0, s1]
        offset = 0
        for i in range(self.node_num):
            s = self.sum([self.edges[offset+j](h, F.softmax(self.alphas[i+j], dim=-1)) for j, h in enumerate(states[-2:])]) #offset +j = 2
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
    
    def forward_memory(self, input0, input1):
        s0 = self.preprocess0(input0)
        s1 = self.preprocess0(input1)

        states = [s0, s1]
        offset = 0

        for i in range(self.node_num):
            s = self.sum(self.edges[offset+j].forward_memory(h, F.softmax(self.alphas[i+j], dim=-1)) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)
        states.append(input1)
        return torch.cat(states[-(self.node_num+1):], dim=1) # channels number is multiplier "4" * C "C_curr"
    
    def forward_ops(self, input0, input1):
        s0 = self.preprocess0(input0)
        s1 = self.preprocess0(input1)

        states = [s0, s1]
        offset = 0

        for i in range(self.node_num):
            s = self.sum(self.edges[offset+j].forward_ops(h, F.softmax(self.alphas[i+j], dim=-1)) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)
        states.append(input1)
        return torch.cat(states[-(self.node_num+1):], dim=1) # channels number is multiplier "4" * C "C_curr"


class UCell(nn.Module):
    def __init__(self,C, C_prev_prev, C_prev, prev_cell_type,init_alphas=None,edges_num=2, ops_num=5, node_num=4,binary=True, affine=False) :
        super(UCell, self).__init__()
        self.edges_num = edges_num
        self.ops_num = ops_num
        self.node_num = node_num
        self.preprocess0 = Preprocess.operations(prev_cell_type,0,C_prev_prev, C, affine)
        self.preprocess1 = Preprocess.operations(prev_cell_type,1,C_prev, C, affine)
        self.edges = nn.ModuleList()

        self.preprocess_skip = Preprocess.skip('u',(2,))

        for n in range(node_num):
            for e in range(edges_num):
                stride = 2 if n+e <= 1 else 1
                self.edges.append(Edge(C, stride, ops_num, 'u'))
        
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
            s = self.sum([self.edges[offset+j](h, F.softmax(self.alphas[i+j], dim=-1)) for j, h in enumerate(states[-2:])]) #offset +j = 2
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
    
    def forward_memory(self, input0, input1):
        s0 = self.preprocess0(input0)
        s1 = self.preprocess0(input1)

        states = [s0, s1]
        offset = 0

        for i in range(self.node_num):
            s = self.sum(self.edges[offset+j].forward_memory(h, F.softmax(self.alphas[i+j], dim=-1)) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)
        states.append(self.preprocess_skip(input1))
        return torch.cat(states[-(self.node_num+1):], dim=1) # channels number is multiplier "4" * C "C_curr"
    
    def forward_ops(self, input0, input1):
        s0 = self.preprocess0(input0)
        s1 = self.preprocess0(input1)

        states = [s0, s1]
        offset = 0

        for i in range(self.node_num):
            s = self.sum(self.edges[offset+j].forward_ops(h, F.softmax(self.alphas[i+j], dim=-1)) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)
        states.append(input1)
        return torch.cat(states[-(self.node_num+1):], dim=1) # channels number is multiplier "4" * C "C_curr"


class LastLayer(nn.Module):
    def __init__(self, in_channels, classes_num=3, binary=True):
        super(LastLayer, self).__init__()
        conv = BinConv1x1 if binary else nn.Conv2d
        self.layers = nn.Sequential(
            conv(in_channels, classes_num),
            nn.BatchNorm2d(classes_num),
            )
    def forward(self, x):
        x = self.layers(x)
        return x

        

if __name__ == '__main__':
    c = NCell(20, 10, 20, 1, 'r')

  

            
