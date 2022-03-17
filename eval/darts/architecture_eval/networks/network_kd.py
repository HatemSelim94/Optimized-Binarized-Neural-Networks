from collections import OrderedDict
import os 
import json
import torch
import torch.nn as nn

from .cell.operations.search_operations import EvalBilinear

from .cell.cells import LastLayer

from .constructor import NetConstructor
from .utilities.genotype import save_genotype, load_genotype

class Network_kd(nn.Module):
    def __init__(self, args):
        super(Network_kd, self).__init__()
        self.affine = args.affine
        self.nodes_num = args.nodes_num
        self.teacher_nodes_num = args.teacher_nodes_num
        self.edge_num = args.edge_num
        self.ops_num = args.ops_num
        self.cells_sequence = args.network_sequence
        self.unique_cells = list(set([i  for i in self.cells_sequence if i in['u','r','n']]))
        self.unique_cells_len = len(self.unique_cells)
        self.initial_channels = args.stem_channels
        self.cells = nn.ModuleList()
        self.fp_cells = nn.ModuleList()
        self.genotype_path = args.genotype_path
        self.genotypes = self.load_genotype(os.path.join(self.genotype_path, args.search_exp_name))
        self.first_layer = Stem(out_channels=self.initial_channels)
        self.jit = args.jit
        self.dropout2d = args.dropout2d_prob
        self.padding_mode=args.padding_mode
        c_prev_prev = self.initial_channels
        c_prev = self.initial_channels
        c = self.initial_channels
        prev_cell = 'n'
        idx = {cell_type: i  for i, cell_type in enumerate(self.unique_cells)} 
        binary = True
        # student
        for cell_type in self.cells_sequence: 
            if cell_type == 'r':
                c = c_prev*2
            elif cell_type == 'u':
                c = c_prev//8
            elif cell_type == 'n':
                c = c_prev
            else:
                continue
            if cell_type =='r':
                cell_specs=(c, c_prev_prev, c_prev, prev_cell, self.genotypes[cell_type],self.initial_channels, 2,self.ops_num, self.nodes_num, binary, self.affine)
            elif cell_type == 'n':
                cell_specs=(c, c_prev_prev, c_prev, prev_cell, self.genotypes[cell_type],1,2,self.ops_num, self.nodes_num, binary, self.affine)
            else:
                cell_specs=(c, c_prev_prev, c_prev, prev_cell,  self.genotypes[cell_type],2, self.ops_num, self.nodes_num,binary, self.affine)
            #print(cell_specs)
            self.cells.append(NetConstructor.construct(cell_type, cell_specs))
            prev_cell = cell_type
            c_prev_prev = c_prev
            c_prev = (self.nodes_num * c) + self.initial_channels
        #self.upsample = EvalBilinear(scale_factor=2)
        binary = False
        # teacher
        for cell_type in self.cells_sequence: 
            if cell_type == 'r':
                c = c_prev*2
            elif cell_type == 'u':
                c = c_prev//8
            elif cell_type == 'n':
                c = c_prev
            else:
                continue
            if cell_type =='r':
                cell_specs=(c, c_prev_prev, c_prev, prev_cell, self.fp_alphas[idx[cell_type]],self.initial_channels, 2,self.ops_num, self.teacher_nodes_num, binary, self.affine)
            elif cell_type == 'n':
                cell_specs=(c, c_prev_prev, c_prev, prev_cell, self.fp_alphas[idx[cell_type]],1,2,self.ops_num, self.teacher_nodes_num, binary, self.affine)
            else:
                cell_specs=(c, c_prev_prev, c_prev, prev_cell, self.fp_alphas[idx[cell_type]], 2, self.ops_num, self.teacher_nodes_num,binary, self.affine)
            #print(cell_specs)
            self.fp_cells.append(NetConstructor.construct(cell_type, cell_specs))
            prev_cell = cell_type
            c_prev_prev = c_prev
            c_prev = (self.nodes_num * c) + self.initial_channels

        # last layer
        self.last_layer = LastLayer(c_prev)
        self.fp_last_layer = LastLayer(c_prev)
        self.kd_criterion = nn.KLDivLoss()
    def forward(self, x):
        kd_loss = []
        t_s0=t_s1=t_skip_input=s0 = s1=skip_input= self.first_layer(x)
        for cell in zip(self.cells, self.fp_cells):
            s0, (s1, skip_input) = s1, cell(s0, s1, skip_input)
            t_s0, (t_s1, t_skip_input) = s1, cell(t_s0, t_s1, t_skip_input)
            kd_loss.append(self.kd_criterion(t_s1, s1))
            #print(s0.shape, s1.shape, skip_input.shape)
        #x = self.upsample(s1)
        x = self.last_layer(s1)
        t_x = self.last_layer(t_s1)
        return x, t_x, kd_loss
    '''
    def forward_latency(self, x):
        output = 0
        for cell in self.cells:
            output += cell(x)
        return output
    '''
    def load_genotype(self, dir=None):
        indices = {}
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
    def __init__(self, out_channels=64, in_channels = 3, kernel_size=3):
        super(Stem, self).__init__()
        self.layers = nn.Sequential(OrderedDict([
            ('stem_conv', nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=False, padding=1, stride=2)),
            ('stem_bn', nn.BatchNorm2d(out_channels)),
            ('stem_tanh', nn.Tanh())
        ]))
        
    def forward(self, x):
        x = self.layers(x)
        return x

