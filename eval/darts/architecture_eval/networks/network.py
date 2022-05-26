from collections import OrderedDict
import json
import torch
import torch.nn as nn

from .cell.operations.search_operations import EvalBilinear
import torchvision.transforms.transforms as T
from .cell.cells import LastLayer, BinASPP

from .constructor import NetConstructor
from .utilities.genotype import save_genotype
import os

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
        self.genotype_path = args.genotype_path
        self.use_old_ver = args.use_old_ver
        self.genotypes = self.load_genotype(os.path.join(self.genotype_path, args.search_exp_name),args.teacher)
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
        self.binary_aspp = args.binary_aspp
        self.use_maxpool = args.use_maxpool

        # first layer (fp)
        self.first_layer = Stem(out_channels=self.initial_channels, affine=self.affine, activation=self.first_layer_activation, use_maxpool=self.use_maxpool)
        # cells
        self.cells = nn.ModuleList()
        c_prev_prev = self.initial_channels
        c_prev = self.initial_channels
        c = self.initial_channels
        prev_cell = 'n'
        idx = {cell_type: i  for i, cell_type in enumerate(self.unique_cells)} 
        for i, cell_type in enumerate(self.cells_sequence): 
            if cell_type == 'r':
                c = int(c_prev*args.channel_expansion_ratio_r)
            elif cell_type == 'u':
                c = int(c_prev//args.channel_reduction_ratio_u)
            elif cell_type == 'n':
                c = int(c_prev*args.channel_normal_ratio_n)
            else:
                continue
            if cell_type =='r':
                cell_specs=(c, c_prev_prev, c_prev, prev_cell, self.genotypes[cell_type] ,self.initial_channels, 2, self.nodes_num, self.binary, self.affine,self.padding_mode, self.jit, self.dropout2d,self.binarization, self.activation)
            elif cell_type == 'n':
                cell_specs=(c, c_prev_prev, c_prev, prev_cell, self.genotypes[cell_type],1,2, self.nodes_num, self.binary, self.affine,self.padding_mode, self.jit,self.dropout2d, self.binarization, self.activation)
            else:
                cell_specs=(c, c_prev_prev, c_prev, prev_cell, self.genotypes[cell_type], 2, self.nodes_num,self.binary, self.affine, self.padding_mode,self.jit,self.dropout2d, self.binarization, self.activation)
            self.cells.append(NetConstructor.construct(cell_type, cell_specs, self.use_skip, old_ver=self.use_old_ver))
            prev_cell = cell_type
            c_prev_prev = c_prev
            if self.use_skip:
                c_prev = (self.nodes_num * c) + self.initial_channels
            else:
                c_prev = self.nodes_num*c
        if self.use_maxpool:
            scale = 4
        else:
            scale = 2 # stem i.e. first layer
        for cell in self.cells_sequence:
            if cell == 'r':
                scale*=2
            if cell == 'u':
                scale /=2
        if self.network_type == 'aspp':
            self.binaspp = BinASPP(c_prev, c_prev, self.padding_mode, self.jit, self.dropout2d, rates=[12,24,32], binarization =self.binarization, binary=args.binary_aspp)
        self.upsample = EvalBilinear(scale_factor=scale, mode= args.upsample_mode)
        # last layer (default:bin)
        self.last_layer = LastLayer(c_prev, classes_num=args.num_of_classes,affine=self.affine, binary=args.last_layer_binary, kernel_size=args.last_layer_kernel_size,jit=args.jit, binarization=self.binarization) #no activation
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
                x = self.binaspp(s1)
                x = self.last_layer(x)
            else:
                x = self.last_layer(s1)
            x = self.upsample(x)
            
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
                x = self.binaspp(s1)
                x = self.last_layer(x)
            else:
                x = self.last_layer(s1)
            x = self.upsample(x)
            return x

    def load_genotype(self, dir=None,teacher=False):
        indices = {}
        if teacher:
            if self.use_old_ver:
                genotype_folder_generic = 'teacher_darts_relaxed_cell_'
            else:    
                genotype_folder_generic = 'teacher_darts_relaxed_cell_modified_'
        else:
            if self.use_old_ver:
                genotype_folder_generic = 'darts_relaxed_cell_'
            else:    
                genotype_folder_generic = 'darts_relaxed_cell_modified_'

        for i, cell_type in enumerate(self.unique_cells):
            genotype_folder = genotype_folder_generic+cell_type
            gentyoes_path = os.path.join(dir, genotype_folder)
            with open(os.path.join(gentyoes_path,'genotype_best.json'), 'r') as f:
                indices[cell_type] = json.load(f)
        return indices
    
    def _loss(self, inputs, targets):
        predictions = self(inputs)
        loss = self.criterion(predictions, targets)
        return loss
    
    def _set_criterion(self,criterion):
        self.criterion = criterion


class Stem(nn.Module):
    def __init__(self, out_channels=64, in_channels = 3, kernel_size=3, layers_num = 1, affine=True, activation='tanh', use_maxpool=False):
        super(Stem, self).__init__()
        activation_func = {'htanh':nn.Hardtanh, 'relu': nn.ReLU, 'prelu':nn.PReLU}
        self.layers = nn.Sequential()
        self.layers.add_module('stem_conv', nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=False, padding=1, stride=2))
        #self.layers.add_module('max_pool', nn.MaxPool2d(3,stride=2, padding=1))
        if use_maxpool:
            self.layers.add_module('max_pool', nn.MaxPool2d(2,2, ceil_mode=True))
        if activation =='relu':
            self.layers.add_module('stem_activation', activation_func[activation]())
        self.layers.add_module('stem_bn', nn.BatchNorm2d(out_channels, affine=affine))
        if activation =='htanh':
            self.layers.add_module('stem_activation', activation_func[activation]()) 
        if activation == 'prelu':
            self.layers.add_module('stem_activation', activation_func[activation](out_channels))
        
    def forward(self, x):
        x = self.layers(x)
        return x

