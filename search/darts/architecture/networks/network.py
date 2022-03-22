from collections import OrderedDict

import torch
import torch.nn as nn

from .cell.operations.search_operations import Bilinear

from .cell.cells import LastLayer,BinASPP
import torchvision.transforms.transforms as T

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
        self.unique_cells = list(set([i  for i in self.cells_sequence if i in['u','r','n']]))
        self.unique_cells_len = len(self.unique_cells)
        self.initial_channels = args.stem_channels
        self.jit = args.jit
        self.onnx = args.onnx
        self.network_type = args.network_type
        self.dropout2d = args.dropout2d_prob
        self.padding_mode=args.padding_mode
        self.binarization = args.binarization
        self.activation = args.activation
        self.first_layer_activation = args.first_layer_activation
        self.use_skip = args.use_skip
        self.kd = args.use_kd
        self.cells = nn.ModuleList()
        self.alphas = [torch.empty((self.nodes_num*self.edge_num ,self.ops_num), requires_grad=True, device=args.device) for _ in range(self.unique_cells_len)]
        for i in range(self.unique_cells_len):
            nn.init.constant_(self.alphas[i], 1/self.ops_num)
                # first layer (fp)
        self.first_layer = Stem(out_channels=self.initial_channels, affine=self.affine, activation=self.first_layer_activation)
        # cells
        self.cells = nn.ModuleList()
        c_prev_prev = self.initial_channels
        c_prev = self.initial_channels
        c = self.initial_channels
        prev_cell = 'n'
        idx = {cell_type: i  for i, cell_type in enumerate(self.unique_cells)} 
        for i, cell_type in enumerate(self.cells_sequence): 
            if cell_type == 'r':
                c = c_prev*2
            elif cell_type == 'u':
                c = c_prev//14
            elif cell_type == 'n':
                c = c_prev
            else:
                continue
            if cell_type =='r':
                cell_specs=(c, c_prev_prev, c_prev, prev_cell, self.alphas[idx[cell_type]] ,self.initial_channels, 2, self.ops_num,self.nodes_num, self.binary, self.affine,self.padding_mode, self.jit, self.dropout2d,self.binarization, self.activation)
            elif cell_type == 'n':
                cell_specs=(c, c_prev_prev, c_prev, prev_cell, self.alphas[idx[cell_type]],1,2,self.ops_num, self.nodes_num, self.binary, self.affine,self.padding_mode, self.jit,self.dropout2d, self.binarization, self.activation)
            else:
                cell_specs=(c, c_prev_prev, c_prev, prev_cell, self.alphas[idx[cell_type]], 2, self.ops_num,self.nodes_num,self.binary, self.affine, self.padding_mode,self.jit,self.dropout2d, self.binarization, self.activation)
            self.cells.append(NetConstructor.construct(cell_type, cell_specs, self.use_skip))
            prev_cell = cell_type
            c_prev_prev = c_prev
            if self.use_skip:
                c_prev = (self.nodes_num * c) + self.initial_channels
            else:
                c_prev = self.nodes_num*c
        
        if self.network_type == 'cells':
            scale = 2
            last_layer_ch = c_prev
        elif self.network_type == 'aspp':
            scale = 4
            last_layer_ch = 64
            self.binaspp = BinASPP(c_prev, 64, self.padding_mode, self.jit, self.dropout2d, rates=[4,8,12,18], binarization =self.binarization)
        self.upsample = Bilinear(scale_factor=scale)
        # last layer (default:bin)
        self.last_layer = LastLayer(last_layer_ch, classes_num=args.num_of_classes,affine=self.affine, binary=args.last_layer_binary, kernel_size=args.last_layer_kernel_size,jit=args.jit, binarization=self.binarization) #no activation
        if self.jit or self.onnx:
            self.transforms = nn.Sequential(T.Resize([args.image_size, args.image_size]), T.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    def forward(self, x):
        if self.jit or self.onnx:
            x = self.transforms(x)
        if self.kd:
            intermediate_outputs = []
            if self.use_skip:
                s0 = s1=skip_input= self.first_layer(x)
                for cell in self.cells:
                    s0, (s1, skip_input) = s1, cell(s0, s1, skip_input)
                    intermediate_outputs.append(s1.clone()) # clone -> inplace ops protection
            else:
                s0 = s1=self.first_layer(x)
                for cell in self.cells:
                    s0, s1 = s1, cell(s0, s1)
                    intermediate_outputs.append(s1.clone()) # clone -> inplace ops protection
            if self.network_type == 'aspp':
                x = self.binaspp(x)
            x = self.upsample(s1)
            x = self.last_layer(x)
            return x, intermediate_outputs
        else:
            if self.use_skip:
                s0 = s1=skip_input= self.first_layer(x)
                for cell in self.cells:
                    s0, (s1, skip_input) = s1, cell(s0, s1, skip_input)
            else:
                s0 = s1=self.first_layer(x)
                for cell in self.cells:
                    s0, s1 = s1, cell(s0, s1)
            if self.network_type == 'aspp':
                x = self.binaspp(x)
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
        torch.use_deterministic_algorithms(False)
        loss = self.criterion(predictions, targets)
        return loss
    
    def _set_criterion(self,criterion):
        self.criterion = criterion


class Stem(nn.Module):
    def __init__(self, out_channels=64, in_channels = 3, kernel_size=3, layers_num = 1, affine=True, activation='tanh'):
        super(Stem, self).__init__()
        activation_func = {'htanh':nn.Tanh, 'relu': nn.ReLU}
        self.layers = nn.Sequential()
        self.layers.add_module('stem_conv', nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=False, padding=1, stride=2))
        if activation =='relu':
            self.layers.add_module('stem_activation', activation_func[activation]())
        self.layers.add_module('stem_bn', nn.BatchNorm2d(out_channels, affine=affine))
        if activation =='htanh':
            self.layers.add_module('stem_activation', activation_func[activation]())            
        
    def forward(self, x):
        x = self.layers(x)
        return x
