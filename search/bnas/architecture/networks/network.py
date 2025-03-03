from collections import OrderedDict

import torch
import torch.nn as nn

from .cell.operations.search_operations import Bilinear

from .cell.cells import LastLayer,BinASPP
import torchvision.transforms.transforms as T

from .constructor import NetConstructor
from .utilities.genotype import save_genotype, load_genotype
import math 
import copy
from .sampled_network import SampledNetwork
from .bnas import Bnas


class Network(nn.Module):
    def __init__(self, args):
        super(Network, self).__init__()
        self.args = args
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
        self.use_old_ver = args.use_old_ver
        if self.use_old_ver:
            self.total_edge_num = sum([self.edge_num+i for i in range(self.nodes_num)])
            self.alphas = [torch.empty((self.total_edge_num ,self.ops_num), requires_grad=True, device=args.device) for _ in range(self.unique_cells_len)]
        else:
            self.alphas = [torch.empty((self.nodes_num*self.edge_num ,self.ops_num), requires_grad=True, device=args.device) for _ in range(self.unique_cells_len)]
        for i in range(self.unique_cells_len):
            nn.init.constant_(self.alphas[i], 1/self.ops_num)
                # first layer (fp)
        self.bnas = Bnas(nodes=args.nodes_num, t=args.arch_t_epochs, edges=self.total_edge_num, unique_cell_types=self.unique_cells)
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
                c = int(c_prev*args.channel_expansion_ratio_r)
            elif cell_type == 'u':
                c = int(c_prev//args.channel_reduction_ratio_u)
            elif cell_type == 'n':
                c = int(c_prev*args.channel_normal_ratio_n)
            else:
                continue
            if cell_type =='r':
                cell_specs=(c, c_prev_prev, c_prev, prev_cell, self.alphas[idx[cell_type]] ,self.initial_channels, 2, self.ops_num,self.nodes_num, self.binary, self.affine,self.padding_mode, self.jit, self.dropout2d,self.binarization, self.activation)
            elif cell_type == 'n':
                cell_specs=(c, c_prev_prev, c_prev, prev_cell, self.alphas[idx[cell_type]],1,2,self.ops_num, self.nodes_num, self.binary, self.affine,self.padding_mode, self.jit,self.dropout2d, self.binarization, self.activation)
            else:
                cell_specs=(c, c_prev_prev, c_prev, prev_cell, self.alphas[idx[cell_type]], 2, self.ops_num,self.nodes_num,self.binary, self.affine, self.padding_mode,self.jit,self.dropout2d, self.binarization, self.activation)
            self.cells.append(NetConstructor.construct(cell_type, cell_specs, None, self.use_old_ver))
            prev_cell = cell_type
            c_prev_prev = c_prev
            c_prev = self.nodes_num*c
        
        if self.network_type == 'cells':
            scale = 2 # stem i.e. first layer
            for cell in self.cells_sequence:
                if cell == 'r':
                    scale*=2
                if cell == 'u':
                    scale /=2
            last_layer_ch = c_prev
        elif self.network_type == 'aspp':
            self.binaspp = BinASPP(c_prev, c_prev, self.padding_mode, self.jit, self.dropout2d, rates=[4,8,12,18], binarization =self.binarization)
        self.upsample = Bilinear(scale_factor=scale)
        # last layer (default:bin)
        self.last_layer = LastLayer(last_layer_ch, classes_num=args.num_of_classes,affine=self.affine, binary=args.last_layer_binary, kernel_size=args.last_layer_kernel_size,jit=args.jit, binarization=self.binarization) #no activation
        if self.jit or self.onnx:
            self.transforms = nn.Sequential(T.Resize([args.image_size, args.image_size]), T.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    def forward(self, x):
        training_idx = self.bnas.get_training_idx()
        s0 = s1=self.first_layer(x)
        for cell in self.cells:
            s0, s1 = s1, cell(s0, s1,training_idx[cell.cell_type])
        if self.network_type == 'aspp':
            x = self.binaspp(s1)
            x = self.last_layer(x)
        else:
            x = self.last_layer(s1)
        x = self.upsample(x)
        return x
        

    def save_genotype(self, dir=None, epoch=0, nodes=4):
        save_genotype(self.model.alphas,dir, epoch,nodes=nodes,types=self.unique_cells,use_old_ver=self.use_old_ver)
    
    def _loss(self, inputs, targets):
        predictions = self(inputs)
        torch.use_deterministic_algorithms(False)
        loss = self.criterion(predictions, targets)
        return loss
    
    def _set_criterion(self,criterion):
        self.criterion = criterion
    
    ###################################################################
    #BNAS
    def select_worst_primitives(self):
        with torch.no_grad():
            training_idx = self.bnas.get_training_idx()
            k = self.get_k() # ops number (k in the paper) 
            worst_ops = math.ceil(k/2)
        
            worst_idx = {}
            for i, cell_type in enumerate(self.unique_cells):
                cell_weights = self.alphas[i].detach().clone().cpu() 
                cell_training_idx = training_idx[cell_type]
                # sort accodring to alpha
                _, worst_training_idx = torch.sort(cell_weights,dim=-1, descending=False, stable=True) # ascending, ids
                #_, worst_training_idx =torch.topk(cell_weights, worst_ops,dim=-1,largest=False, sorted=False)
                worst_training_idx = worst_training_idx.tolist()
                idx_modified = copy.deepcopy(worst_training_idx)
                j=0 
                # remove all but training ops
                for edge_idx, edge_training_idx in zip(worst_training_idx, cell_training_idx): 
                    for op_id in edge_idx:
                        if op_id not in edge_training_idx:
                            idx_modified[j].remove(op_id)
                    j += 1  
                idx_modified = torch.tensor(idx_modified) 
                # select the worst [k/2] ops for each edge
                worst_idx[cell_type] = idx_modified[:,:worst_ops]# contains the worst k training ops
                assert worst_idx[cell_type].shape[-1] == worst_ops
                assert worst_idx[cell_type].shape[0] == self.total_edge_num
        return worst_idx
    
    def set_worst_primitives(self):
        worst_idx = self.select_worst_primitives()
        self.bnas.set_worst_primitives(worst_idx)
        return len(worst_idx[self.unique_cells[0]][0])
    
    def get_k(self):
        training_idx = self.bnas.get_training_idx()
        k = len(training_idx[self.unique_cells[0]][0]) # k in the paper
        return k

    def sample_network(self):
        s = self.bnas.sample()
        self.sampled_primitives = s
        return self.new_full_model(s)
    
    def save_gen(self):
        self.bnas.save_genotype(self.args)
        


    def new_full_model(self, sampled_primitives_idx):
        self.args.affine=True
        model_new = SampledNetwork(self.args, sampled_primitives_idx).cuda()
        self.args.affine=False
        model_new.load_state_dict(self.state_dict(), strict=False)
        return model_new
    
    def update_score(self, t, score):
        self.bnas.update_score(self.sampled_primitives, t, score)
    
    def reudce_space(self):
        self.bnas.reduce_space()
    #############################################################################

class Stem(nn.Module):
    def __init__(self, out_channels=64, in_channels = 3, kernel_size=3, layers_num = 1, affine=True, activation='tanh'):
        super(Stem, self).__init__()
        activation_func = {'htanh':nn.Hardtanh, 'relu': nn.ReLU}
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