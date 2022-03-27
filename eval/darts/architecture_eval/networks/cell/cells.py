import torch
import torch.nn as nn
import torch.nn.functional as F

from .operations.search_operations import BinConvbn1x1, BinConvT1x1, ConvBn, BasicBinConv1x1, BinDilConv3x3,BinConv1x1

from .operations import EvalSum, Preprocess, FpPreprocess,EvalCat, EvalBilinear
from .edge import EvalEdge


class NCell(nn.Module):
    def __init__(self,C, C_prev_prev, C_prev, prev_cell_type,genotype,stride=1, edges_num=2, node_num=4,binary=True, affine=True,padding_mode='zeros',jit=False,dropout2d=0.1, binarization=1, activation='htanh'):
        super(NCell, self).__init__()
        self.edges_num = edges_num
        self.node_num = node_num
        self.binary = binary
        preprocess = Preprocess if binary else FpPreprocess
        self.preprocess0 = preprocess.operations(prev_cell_type,0,C_prev_prev, C, affine, padding_mode=padding_mode,jit= jit,dropout2d=dropout2d, binarization=binarization,activation=activation)
        self.preprocess1 = preprocess.operations(prev_cell_type,1,C_prev, C, affine, padding_mode=padding_mode, jit=jit,dropout2d=dropout2d, binarization=binarization,activation=activation)
        self.edges = nn.ModuleList()
        i = 0 
        for _ in range(node_num):
            for _ in range(edges_num):
                self.edges.append(EvalEdge(C, stride, genotype[i], 'n', affine, self.binary, padding_mode=padding_mode, jit=jit,dropout2d=dropout2d, binarization=binarization, activation=activation))
                i+=1
        self.sum = EvalSum()
        self.cat = EvalCat()

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
    
    def binarize_weight(self, state=True):
        def recurs(net,state):
            for mod in net.children():
                if hasattr(mod,'binarized_weight'):
                    setattr(mod,'binarized_weight',state)
                else:
                    recurs(mod, state)
        recurs(self, state)
    BinConv1x1


class RCell(nn.Module):
    def __init__(self,C, C_prev_prev, C_prev, prev_cell_type, genotype, skip_channels=64,edges_num=2, node_num=4,binary=True, affine=True, padding_mode='zeros',jit=False,dropout2d=0.1, binarization=1,activation='htanh'):
        super(RCell, self).__init__()
        self.edges_num = edges_num
        self.node_num = node_num
        self.binary = binary
        preprocess = Preprocess if binary else FpPreprocess
        self.preprocess0 = preprocess.operations(prev_cell_type,0,C_prev_prev, C, affine, padding_mode=padding_mode, jit=jit,dropout2d=dropout2d,binarization=binarization,activation=activation)
        self.preprocess1 = preprocess.operations(prev_cell_type,1,C_prev, C, affine, padding_mode=padding_mode,jit= jit,dropout2d=dropout2d, binarization=binarization,activation=activation)
        self.edges = nn.ModuleList()
        self.C = C

        self.preprocess_skip = preprocess.skip('r',(skip_channels, affine, 2))
        i=0
        for n in range(node_num):
            for e in range(edges_num):
                stride = 2 if n+e <= 1 else 1
                self.edges.append(EvalEdge(C, stride, genotype[i],'r', affine, binary, padding_mode=padding_mode,jit=jit,dropout2d=dropout2d, binarization=binarization,activation=activation))
                i+=1
        
        self.sum = EvalSum()
        self.cat = EvalCat()

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
    def __init__(self,C, C_prev_prev, C_prev, prev_cell_type, genotype,edges_num=2, node_num=4,binary=True, affine=True,padding_mode='zeros', jit=False,dropout2d=0.1, binarization=1,activation = 'htanh') :
        super(UCell, self).__init__()
        self.edges_num = edges_num
        self.node_num = node_num
        prprocess = Preprocess if binary else FpPreprocess
        self.preprocess0 = prprocess.operations(prev_cell_type,0,C_prev_prev, C, affine, padding_mode=padding_mode, jit=jit,dropout2d=dropout2d, binarization=binarization,activation=activation)
        self.preprocess1 = prprocess.operations(prev_cell_type,1,C_prev, C, affine, padding_mode=padding_mode, jit=jit,dropout2d=dropout2d, binarization=binarization,activation=activation)
        self.edges = nn.ModuleList()
        self.binary = binary
        self.preprocess_skip = prprocess.skip('u',(2,))
        i=0

        for n in range(node_num):
            for e in range(edges_num):
                stride = 2 if n+e <= 1 else 1
                self.edges.append(EvalEdge(C, stride, genotype[i], 'u', affine, binary,padding_mode=padding_mode,jit=jit, binarization=binarization,activation=activation))
                i+=1
        self.sum = EvalSum()
        self.cat = EvalCat()

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
    def __init__(self,C, C_prev_prev, C_prev, prev_cell_type,genotype,stride=1, edges_num=2, node_num=4,binary=True, affine=True,padding_mode='zeros',jit=False,dropout2d=0.1, binarization=1,activation='htanh'):
        super(NCellNSkip, self).__init__()
        self.edges_num = edges_num
        self.node_num = node_num
        self.binary = binary
        preprocess = Preprocess if binary else FpPreprocess
        self.preprocess0 = preprocess.operations(prev_cell_type,0,C_prev_prev, C, affine, padding_mode=padding_mode,jit= jit,dropout2d=dropout2d, binarization=binarization,activation=activation)
        self.preprocess1 = preprocess.operations(prev_cell_type,1,C_prev, C, affine, padding_mode=padding_mode, jit=jit,dropout2d=dropout2d, binarization=binarization,activation=activation)
        self.edges = nn.ModuleList()
        i = 0 
        for _ in range(node_num):
            for _ in range(edges_num):
                self.edges.append(EvalEdge(C, stride, genotype[i], 'n', affine, self.binary, padding_mode=padding_mode, jit=jit,dropout2d=dropout2d, binarization=binarization,activation=activation))
                i+=1
        self.sum = EvalSum()
        self.cat = EvalCat()

    def forward(self, input0, input1):
        s0 = self.preprocess0(input0)
        s1 = self.preprocess1(input1)

        states = [s0, s1]
        #print(states[-2].shape)
        offset = 0
        for i in range(self.node_num):
            s = self.sum([self.edges[offset+j](h) for j, h in enumerate(states[-2:])]) #offset +j = 2
            offset += 2
            states.append(s)
        return self.cat(states[-(self.node_num):])# channels number is multiplier "4" * C "C_curr"
    
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
    def __init__(self,C, C_prev_prev, C_prev, prev_cell_type, genotype, skip_channels,edges_num=2, node_num=4,binary=True, affine=True, padding_mode='zeros',jit=False,dropout2d=0.1, binarization=1,activation='htanh'):
        super(RCellNSkip, self).__init__()
        self.edges_num = edges_num
        self.node_num = node_num
        self.binary = binary
        preprocess = Preprocess if binary else FpPreprocess
        self.preprocess0 = preprocess.operations(prev_cell_type,0,C_prev_prev, C, affine, padding_mode=padding_mode, jit=jit,dropout2d=dropout2d,binarization=binarization,activation=activation)
        self.preprocess1 = preprocess.operations(prev_cell_type,1,C_prev, C, affine, padding_mode=padding_mode,jit= jit,dropout2d=dropout2d, binarization=binarization,activation=activation)
        self.edges = nn.ModuleList()
        self.C = C

        i=0
        for n in range(node_num):
            for e in range(edges_num):
                stride = 2 if n+e <= 1 else 1
                self.edges.append(EvalEdge(C, stride, genotype[i],'r', affine, binary, padding_mode=padding_mode,jit=jit,dropout2d=dropout2d, binarization=binarization,activation=activation))
                i+=1
        
        self.sum = EvalSum()
        self.cat = EvalCat()

    def forward(self, input0, input1):
        s0 = self.preprocess0(input0)
        s1 = self.preprocess1(input1)

        states = [s0, s1]
        offset = 0
        for i in range(self.node_num):
            s = self.sum([self.edges[offset+j](h) for j, h in enumerate(states[-2:])]) #offset +j = 2
            offset += 2
            states.append(s)
        return self.cat(states[-(self.node_num):]) # channels number is multiplier "4" * C "C_curr"
    
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
    def __init__(self,C, C_prev_prev, C_prev, prev_cell_type, genotype,edges_num=2, node_num=4,binary=True, affine=True,padding_mode='zeros', jit=False,dropout2d=0.1, binarization=1,activation='htanh') :
        super(UCellNSkip, self).__init__()
        self.edges_num = edges_num
        self.node_num = node_num
        prprocess = Preprocess if binary else FpPreprocess
        self.preprocess0 = prprocess.operations(prev_cell_type,0,C_prev_prev, C, affine, padding_mode=padding_mode, jit=jit,dropout2d=dropout2d, binarization=binarization,activation=activation)
        self.preprocess1 = prprocess.operations(prev_cell_type,1,C_prev, C, affine, padding_mode=padding_mode, jit=jit,dropout2d=dropout2d, binarization=binarization,activation=activation)
        self.edges = nn.ModuleList()
        self.binary = binary
        i=0

        for n in range(node_num):
            for e in range(edges_num):
                stride = 2 if n+e <= 1 else 1
                self.edges.append(EvalEdge(C, stride, genotype[i], 'u', affine, binary,padding_mode=padding_mode,jit=jit, binarization=binarization,activation=activation))
                i+=1
        self.sum = EvalSum()
        self.cat = EvalCat()

    def forward(self, input0, input1):
        s0 = self.preprocess0(input0)
        s1 = self.preprocess1(input1)
        states = [s0, s1]
        offset = 0
        for i in range(self.node_num):
            s = self.sum([self.edges[offset+j](h) for j, h in enumerate(states[-2:])]) #offset +j = 2
            offset += 2
            states.append(s)
        return self.cat(states[-(self.node_num):]) # channels number is multiplier "4" * C "C_curr"
    
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
    def __init__(self,C, C_prev_prev, C_prev, prev_cell_type, genotype,skip_ch,edges_num=2, node_num=4,binary=True, affine=True,padding_mode='zeros', jit=False,dropout2d=0.1, binarization=1,activation='htanh') :
        super(NCellNSkipOld, self).__init__()
        self.edges_num = edges_num
        self.node_num = node_num
        prprocess = Preprocess if binary else FpPreprocess
        self.preprocess0 = prprocess.operations(prev_cell_type,0,C_prev_prev, C, affine, padding_mode=padding_mode, jit=jit,dropout2d=dropout2d, binarization=binarization,activation=activation)
        self.preprocess1 = prprocess.operations(prev_cell_type,1,C_prev, C, affine, padding_mode=padding_mode, jit=jit,dropout2d=dropout2d, binarization=binarization,activation=activation)
        self.edges = nn.ModuleList()
        self.binary = binary
        i=0
        self.genotype_states_idx = genotype['states']
        for n in range(node_num):
            for e in range(edges_num):
                self.edges.append(EvalEdge(C, 1, [genotype['ops'][i]], 'n', affine, binary,padding_mode=padding_mode,jit=jit, binarization=binarization,activation=activation))
                i+=1
        self.sum = EvalSum()
        self.cat = EvalCat()

    def forward(self, input0, input1):
        s0 = self.preprocess0(input0)
        s1 = self.preprocess1(input1)
        states = [s0, s1]
        offset = 0
        for n in range(self.node_num):
            s = self.sum([self.edges[offset+j](h) for j,h in  enumerate([states[self.genotype_states_idx[n][0]], states[self.genotype_states_idx[n][1]]])]) #offset +j = 2
            offset += 2
            states.append(s)
        return self.cat(states[-(self.node_num):]) # channels number is multiplier "4" * C "C_curr"
    
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
    def __init__(self,C, C_prev_prev, C_prev, prev_cell_type, genotype,skip_channels,edges_num=2, node_num=4,binary=True, affine=True,padding_mode='zeros', jit=False,dropout2d=0.1, binarization=1,activation='htanh') :
        super(RCellNSkipOld, self).__init__()
        self.edges_num = edges_num
        self.node_num = node_num
        prprocess = Preprocess if binary else FpPreprocess
        self.preprocess0 = prprocess.operations(prev_cell_type,0,C_prev_prev, C, affine, padding_mode=padding_mode, jit=jit,dropout2d=dropout2d, binarization=binarization,activation=activation)
        self.preprocess1 = prprocess.operations(prev_cell_type,1,C_prev, C, affine, padding_mode=padding_mode, jit=jit,dropout2d=dropout2d, binarization=binarization,activation=activation)
        self.edges = nn.ModuleList()
        self.binary = binary
        i=0
        self.genotype_states_idx = genotype['states']
        for n in range(node_num):
            for e in range(edges_num):
                stride = 2 if self.genotype_states_idx[n][e] < 2 else 1
                self.edges.append(EvalEdge(C, stride, [genotype['ops'][i]], 'r', affine, binary,padding_mode=padding_mode,jit=jit, binarization=binarization,activation=activation))
                i+=1
        self.sum = EvalSum()
        self.cat = EvalCat()

    def forward(self, input0, input1):
        s0 = self.preprocess0(input0)
        s1 = self.preprocess1(input1)
        states = [s0, s1]
        offset = 0
        for n in range(self.node_num):
            s = self.sum([self.edges[offset+j](h) for j,h in  enumerate([states[self.genotype_states_idx[n][0]], states[self.genotype_states_idx[n][1]]])]) #offset +j = 2
            offset += 2
            states.append(s)
        return self.cat(states[-(self.node_num):]) # channels number is multiplier "4" * C "C_curr"
    
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
    def __init__(self,C, C_prev_prev, C_prev, prev_cell_type, genotype,edges_num=2, node_num=4,binary=True, affine=True,padding_mode='zeros', jit=False,dropout2d=0.1, binarization=1,activation='htanh') :
        super(UCellNSkipOld, self).__init__()
        self.edges_num = edges_num
        self.node_num = node_num
        prprocess = Preprocess if binary else FpPreprocess
        self.preprocess0 = prprocess.operations(prev_cell_type,0,C_prev_prev, C, affine, padding_mode=padding_mode, jit=jit,dropout2d=dropout2d, binarization=binarization,activation=activation)
        self.preprocess1 = prprocess.operations(prev_cell_type,1,C_prev, C, affine, padding_mode=padding_mode, jit=jit,dropout2d=dropout2d, binarization=binarization,activation=activation)
        self.edges = nn.ModuleList()
        self.binary = binary
        i=0
        self.genotype_states_idx = genotype['states']
        for n in range(node_num):
            for e in range(edges_num):
                stride = 2 if self.genotype_states_idx[n][e] < 2 else 1
                self.edges.append(EvalEdge(C, stride, [genotype['ops'][i]], 'u', affine, binary,padding_mode=padding_mode,jit=jit, binarization=binarization,activation=activation))
                i+=1
        self.sum = EvalSum()
        self.cat = EvalCat()

    def forward(self, input0, input1):
        s0 = self.preprocess0(input0)
        s1 = self.preprocess1(input1)
        states = [s0, s1]
        offset = 0
        for n in range(self.node_num):
            s = self.sum([self.edges[offset+j](h) for j,h in  enumerate([states[self.genotype_states_idx[n][0]], states[self.genotype_states_idx[n][1]]])]) #offset +j = 2
            offset += 2
            states.append(s)
        return self.cat(states[-(self.node_num):]) # channels number is multiplier "4" * C "C_curr"
    
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
    def __init__(self, in_channels, classes_num=3, binary=True, affine=True, kernel_size=3, jit = False, binarization=1):
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
    def __init__(self, in_channels, out_channels,padding_mode,jit, dropout2d, rates=[12,24, 36], binarization=1,activation='htanh', binary=False):
        super(BinASPP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(Pooling(in_channels, out_channels, padding_mode,jit, dropout2d,binarization=binarization))
        self.layers.append(BinConv1x1(in_channels, out_channels,padding_mode=padding_mode, jit=jit, dropout2d=dropout2d, binarization=binarization, activation=activation))
        for r in rates:
            self.layers.append(BinDilConv3x3(in_channels, out_channels,stride=1, padding=r,dilation=r, padding_mode=padding_mode,jit=jit, dropout2d=dropout2d, binarization=binarization,activation=activation))
        #self.sum = EvalSum()
        if binary:
            self.project= BinConv1x1(len(self.layers) * out_channels, out_channels, 1, jit=jit, binarization=binarization, dropout2d=dropout2d, activation=activation)
            
        else:    
            self.project= nn.Sequential(
                nn.Conv2d(len(self.layers) * out_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
                )

    def forward(self, x):
        output = []
        for mod in self.layers:
            output.append(mod(x))
        #s = torch.sum(torch.cat(output, dim=1), dim=1)
        #return s
        return self.project(torch.cat(output, dim=1))

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

  

            
