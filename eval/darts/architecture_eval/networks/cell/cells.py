import torch
import torch.nn as nn
import torch.nn.functional as F

from .operations.search_operations import BinConvbn1x1, BinConvT1x1, ConvBn

from .operations import Sum, Preprocess, FpPreprocess,Cat
from .edge import Edge


class NCell(nn.Module):
    def __init__(self,C, C_prev_prev, C_prev, prev_cell_type,genotype,stride=1, edges_num=2, node_num=4,binary=True, affine=True):
        super(NCell, self).__init__()
        self.edges_num = edges_num
        self.node_num = node_num
        self.binary = binary
        preprocess = Preprocess if binary else FpPreprocess
        self.preprocess0 = preprocess.operations(prev_cell_type,0,C_prev_prev, C, affine)
        self.preprocess1 = preprocess.operations(prev_cell_type,1,C_prev, C, affine)
        self.edges = nn.ModuleList()
        i = 0 
        for _ in range(node_num):
            for _ in range(edges_num):
                self.edges.append(Edge(C, stride, genotype[i], 'n', affine, self.binary))
                i+=1
        self.sum = Sum()
        self.cat = Cat()

    def forward(self, input0, input1,  skip_input):
        s0 = self.preprocess0(input0)
        s1 = self.preprocess1(input1)

        states = [s0, s1]
        #print(states[-2].shape)
        offset = 0
        for i in range(self.node_num):
            s = self.sum([self.edges[offset+j](h) for j, h in enumerate(states[-2:])]) #offset +j = 2
            offset += 2
            states.append(s)
        states.append(skip_input)
        return self.cat(states[-(self.node_num+1):]), skip_input # channels number is multiplier "4" * C "C_curr"
        

class RCell(nn.Module):
    def __init__(self,C, C_prev_prev, C_prev, prev_cell_type, genotype, skip_channels=64,edges_num=2, node_num=4,binary=True, affine=True):
        super(RCell, self).__init__()
        self.edges_num = edges_num
        self.node_num = node_num
        self.binary = binary
        preprocess = Preprocess if binary else FpPreprocess
        self.preprocess0 = preprocess.operations(prev_cell_type,0,C_prev_prev, C, affine)
        self.preprocess1 = preprocess.operations(prev_cell_type,1,C_prev, C, affine)
        self.edges = nn.ModuleList()
        self.C = C

        self.preprocess_skip = preprocess.skip('r',(skip_channels, affine, 2))
        i=0
        for n in range(node_num):
            for e in range(edges_num):
                stride = 2 if n+e <= 1 else 1
                self.edges.append(Edge(C, stride, genotype[i],'r', affine, binary))
                i+=1
        
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
            s = self.sum([self.edges[offset+j](h) for j, h in enumerate(states[-2:])]) #offset +j = 2
            offset += 2
            states.append(s)
        prp = self.preprocess_skip(skip_input)
        #print(prp.shape)
        states.append(prp)
        return self.cat(states[-(self.node_num+1):]), prp # channels number is multiplier "4" * C "C_curr"


class UCell(nn.Module):
    def __init__(self,C, C_prev_prev, C_prev, prev_cell_type, genotype,edges_num=2, node_num=4,binary=True, affine=True) :
        super(UCell, self).__init__()
        self.edges_num = edges_num
        self.node_num = node_num
        prprocess = Preprocess if binary else FpPreprocess
        self.preprocess0 = prprocess.operations(prev_cell_type,0,C_prev_prev, C, affine)
        self.preprocess1 = prprocess.operations(prev_cell_type,1,C_prev, C, affine)
        self.edges = nn.ModuleList()
        self.binary = binary
        self.preprocess_skip = prprocess.skip('u',(2,))
        i=0

        for n in range(node_num):
            for e in range(edges_num):
                stride = 2 if n+e <= 1 else 1
                self.edges.append(Edge(C, stride, genotype[i], 'u', affine, binary))
                i+=1
        self.sum = Sum()
        self.cat = Cat()

    def forward(self, input0, input1, skip_input):
        s0 = self.preprocess0(input0)
        s1 = self.preprocess1(input1)
        #print(s0.shape, s1.shape)
        states = [s0, s1]
        offset = 0
        for i in range(self.node_num):
            s = self.sum([self.edges[offset+j](h) for j, h in enumerate(states[-2:])]) #offset +j = 2
            offset += 2
            states.append(s)
        prp = self.preprocess_skip(skip_input)
        #print(prp.shape)
        states.append(prp)
        #states.append(input1)
        return self.cat(states[-(self.node_num+1):]), prp # channels number is multiplier "4" * C "C_curr"


class LastLayer(nn.Module):
    def __init__(self, in_channels, classes_num=3, binary=True, affine=True, kernel_size=3):
        super(LastLayer, self).__init__() 
        conv = BinConvbn1x1 if binary else nn.Conv2d
        #conv = ConvBn if binary else nn.Conv2d
        self.layers = nn.Sequential(
            #BinConvT1x1(in_channels, 30, affine=False),
            conv(in_channels, classes_num, kernel_size)
            )
        if not binary:
            self.layers.add_module(
                nn.BatchNorm2d(classes_num, affine=affine)
            )
    def forward(self, x):
        x = self.layers(x)
        return x

        

if __name__ == '__main__':
    c = NCell(20, 10, 20, 1, 'r')

  

            
