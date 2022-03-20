from collections import OrderedDict
import math
import torch
import torch.nn as nn
import copy
from .cell.operations.search_operations import Bilinear

from .cell.cells import LastLayer

from .constructor import NetConstructor
from .utilities.genotype import save_genotype, load_genotype
from bnas_optimization import Cells
from .sample_network import SampleNetwork


NR_PRIMITIVES = [
    'max_pool_3x3',
    'avg_pool_3x3',
    'conv_1x1',
    'conv_3x3',
    'dil_conv_3x3_r4',
    'skip_connect'
]

UP_PRIMITIVES = [
    'tconv_1x1',
    'tconv_3x3',
    'tconv_5x5',
    'dil_tconv_3x3_r4',
    'dil_tconv_3x3_r6',
    'dil_tconv_3x3_r8'
]


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
        self.cells = nn.ModuleList()
        self.alphas = [torch.empty((self.nodes_num*self.edge_num ,self.ops_num), requires_grad=True, device=args.device) for _ in range(self.unique_cells_len)]
        for i in range(self.unique_cells_len):
            nn.init.constant_(self.alphas[i], 1/self.ops_num)
        self.first_layer = Stem(out_channels=self.initial_channels, affine=self.affine)
        primitives = {}
        for type in self.unique_cells:
            if type =='r' or type =='n': 
                primitives[type] = NR_PRIMITIVES
            if type =='u':
                 primitives[type] = UP_PRIMITIVES
        self.bnas = Cells(cell_types=self.unique_cells,primitives=primitives, nodes=self.nodes_num, ops_num = self.ops_num, binary=True)

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
                cell_specs=(c, c_prev_prev, c_prev, prev_cell, self.alphas[idx[cell_type]],self.initial_channels, 2,self.ops_num, self.nodes_num, self.binary, self.affine)
            elif cell_type == 'n':
                cell_specs=(c, c_prev_prev, c_prev, prev_cell, self.alphas[idx[cell_type]],1,2,self.ops_num, self.nodes_num, self.binary, self.affine)
            else:
                cell_specs=(c, c_prev_prev, c_prev, prev_cell, self.alphas[idx[cell_type]], 2, self.ops_num, self.nodes_num,self.binary, self.affine)
            #print(cell_specs)
            self.cells.append(NetConstructor.construct(cell_type, cell_specs))
            prev_cell = cell_type
            c_prev_prev = c_prev
            c_prev = (self.nodes_num * c) + self.initial_channels
     
        self.upsample = Bilinear(scale_factor=2)
        self.last_layer = LastLayer(c_prev, classes_num=args.num_of_classes,affine=self.affine, binary=args.last_layer_binary, kernel_size=args.last_layer_kernel_size)
    
    def forward(self, x):
        training_idx = self.cells_primitives.get_training_idx()
        s0 = s1=skip_input= self.first_layer(x)
        for cell, type in zip(self.cells, self.cells_sequence):
            s0, (s1, skip_input) = s1, cell(s0, s1, skip_input, training_idx[type])
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
    
      ###################################################################
    def select_worst_primitives(self):
        with torch.no_grad():
            #_, ops = self._arch_parameters[0].shape
            training_idx = self.bnas.get_training_idx()
            ops_num = len(training_idx['r'][0]) # k in the paper
            worst_ops = math.ceil(ops_num/2)
        
            worst_idx = {}
            for i, cell_type in enumerate(self.unique_cells):
                cell_weights = self.alphas[i].detach().clone().cpu() 
                cell_training_idx = training_idx[cell_type]
                _, idx = torch.sort(cell_weights,dim=-1, descending=False, stable=True) # ascending, ids
                idx = idx.numpy().tolist()
                idx_modified = copy.deepcopy(idx)
                j=0
                for edge_idx, edge_training_idx in zip(idx, cell_training_idx):
                    for op_id in edge_idx:
                        if op_id not in edge_training_idx:
                            idx_modified[j].remove(op_id)
                    j += 1  
                idx_modified = torch.tensor(idx_modified)
                worst_idx[cell_type] = idx_modified[:,:worst_ops]
        return worst_idx
    
    def set_worst_primitives(self):
        worst_idx = self.select_worst_primitives()
        self.cells_primitives.set_worst_primitives(worst_idx)
        return len(worst_idx['r'][0])
    
    def get_k(self):
        training_idx = self.cells_primitives.get_training_idx()
        k = len(training_idx['r'][0]) # k in the paper
        return k

    def sample_network(self):
        s = self.bnas.sample()
        self.sampled_primitives = s
        return self.new_full_model(s)


    def new_full_model(self, sample_primitives_idx):
        model_new = SampleNetwork(self._C, self._num_classes, self._layers, self._criterion, binary= self.binary, primitive_idx=sample_primitives_idx).cuda()
        model_new.load_state_dict(self.state_dict(), strict=False)
        return model_new
    
    def update_score(self, t, score):
        self.bnas.update_score(self.sampled_primitives, t, score)
    
    def reudce_space(self):
        self.bnas.reduce_space()



class Stem(nn.Module):
    def __init__(self, out_channels=64, in_channels = 3, kernel_size=3, layers_num = 1, affine=False):
        super(Stem, self).__init__()
        self.layers = nn.Sequential(OrderedDict([
            ('stem_conv', nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=False, padding=1, stride=2)),
            ('stem_bn', nn.BatchNorm2d(out_channels, affine=affine)),
            ('stem_tanh', nn.Tanh())
        ]))
        
    def forward(self, x):
        x = self.layers(x)
        return x

