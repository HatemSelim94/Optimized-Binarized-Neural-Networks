from collections import OrderedDict

import torch
import torch.nn as nn

from .cell.operations.search_operations import Bilinear

from .cell.cells import LastLayer

from .constructor import NetConstructor
from .utilities.genotype import save_genotype, load_genotype

class Network(nn.Module):
    def __init__(self, args):
        super(Network, self).__init__()
        self.binary=args.binary
        self.affine = args.affine
        self.nodes_num = args.nodes_num
        self.edge_num = args.edge_num
        self.ops_num = args.ops_num
        self.cells_sequence = args.network_sequence
        self.initial_channels = args.stem_channels
        self.cells = nn.ModuleList()
        self.alphas = [torch.empty((self.nodes_num*self.edge_num ,self.ops_num), requires_grad=True, device=args.device) for _ in range(3)]
        for i in range(3):
            nn.init.constant_(self.alphas[i], 1/self.ops_num)
        self.first_layer = Stem(out_channels=self.initial_channels)
        
        c_prev_prev = self.initial_channels
        c_prev = self.initial_channels
        c = self.initial_channels
        prev_cell = 'n'

        for i, cell_type in enumerate(self.cells_sequence): 
            if cell_type == 'r':
                c = c_prev*2
            elif cell_type == 'u':
                c = c_prev//2
            elif cell_type == 'n':
                c = c_prev
            else:
                continue
            if cell_type =='r':
                cell_specs=(c, c_prev_prev, c_prev, prev_cell, self.alphas[1],self.initial_channels, 2,self.ops_num, self.nodes_num, self.binary, self.affine)
            elif cell_type == 'n':
                cell_specs=(c, c_prev_prev, c_prev, prev_cell, self.alphas[0],1,2,self.ops_num, self.nodes_num, self.binary, self.affine)
            else:
                cell_specs=(c, c_prev_prev, c_prev, prev_cell, self.alphas[2], 2, self.ops_num, self.nodes_num,self.binary, self.affine)
            self.cells.append(NetConstructor.construct(cell_type, cell_specs))
            prev_cell = cell_type
            c_prev_prev = c_prev
            c_prev = (self.nodes_num * c) + self.initial_channels
     
        self.upsample = Bilinear(scale_factor=2)
        self.last_layer = LastLayer(c_prev)
    
    def forward(self, x):
        s0 = s1=skip_input= self.first_layer(x)
        for cell in self.cells:
            s0, (s1, skip_input) = s1, cell(s0, s1, skip_input)
            #print(s0.shape, s1.shape, skip_input.shape)
        x = self.upsample(s1)
        x = self.last_layer(x)
        return x
    '''
    def forward_latency(self, x):
        output = 0
        for cell in self.cells:
            output += cell(x)
        return output
    '''
    def load_genotype(self, dir=None):
        self.idx = load_genotype(dir)

    def save_genotype(self, dir=None, epoch=0, nodes=4):
        save_genotype(self.model.alphas,dir, epoch,nodes=nodes)
    
    def _loss(self, inputs, targets):
        predictions = self(inputs)
        loss = self.criterion(predictions, targets)
        return loss
    
    def _set_criterion(self,criterion):
        self.criterion = criterion


class Stem(nn.Module):
    def __init__(self, out_channels=64, in_channels = 3, kernel_size=3, layers_num = 1):
        super(Stem, self).__init__()
        self.layers = nn.Sequential(OrderedDict([
            ('stem_conv', nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=False, padding=1, stride=2)),
            ('stem_bn', nn.BatchNorm2d(out_channels)),
            ('stem_tanh', nn.Tanh())
        ]))
        
    def forward(self, x):
        x = self.layers(x)
        return x

