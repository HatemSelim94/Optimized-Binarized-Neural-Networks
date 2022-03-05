import torch
import torch.nn as nn

from .cell.cells import LastLayer

from .constructor import NetConstructor
from .utilities.genotype import save_genotype, load_genotype

class Network(nn.Module):
    def __init__(self, args):
        super(Network, self).__init__()
        self.nodes_num = args.nodes_num
        self.edge_num = args.edge_num
        self.ops_num = args.ops_num
        self.cells_sequence = args.cell_sequence
        self.initial_channels = args.initial_channels
        self.cells = nn.ModuleList()
        self.alphas = torch.empty((3, self.nodes_num*self.edge_num ,self.ops_num), requires_grad=True, device=args.device)
        nn.init.constant_(self.alphas, 1/self.ops_num)
        self.first_layer = Stem(out_channels=self.initial_channels)
        
        c_prev_prev = self.initial_channels
        c_prev = self.initial_channels
        c = self.initial_channels
        prev_cell = 'n'
        specs = []

        for i, cell_type in enumerate(self.cells_sequence): 
            if cell_type == 'r':
                c = c_prev*2
            elif cell_type == 'u':
                c = c_prev//2
            else:
                c = c_prev
            specs.append((c, c_prev_prev, c_prev, prev_cell))
            prev_cell = cell_type
            c_prev_prev = c_prev
            c_prev = 4 * c

        for cell_type, cell_specs in zip(self.cells_sequence, specs):
            self.cells.append(NetConstructor.construct(cell_type, cell_specs))
        
        self.last_layer = LastLayer(c_prev)
    
    def forward(self, x):
        x = self.first_layer(x)
        for cell in self.cells:
            x = cell(x)
        x = self.last_layer(x)
        return x
    
    def forward_latency(self, x):
        output = 0
        for cell in self.cells:
            output += cell(x)
        return output

    def forward_flops(self, x):
        output = 0
        for cell in self.cells:
            output += cell.forward_flops(x)
        return output

    def forward_memory(self, x):
        output = -1
        for cell in self.cells:
            output = max( output, cell.forward_memory(x))
        return output
    
    def forward_params(self, x):
        output = 0
        for cell in self.cells:
            output += cell.forward_params(x)
        return output
    
    def load_genotype(self, dir=None):
        self.idx = load_genotype(dir)

    def save_genotype(self, dir=None, epoch=0):
        save_genotype(self.model.alphas,dir, epoch)
    
    def _loss(self, predictions, targets):
        loss = self.criterion(predictions, targets)
        return loss
    
    def _set_criterion(self,criterion):
        self.criterion = criterion


class Stem(nn.Module):
    def __init__(self, out_channels=64, in_channels = 3, kernel_size=3, layers_num = 1):
        super(Stem, self).__init__()
        self.layers = nn.Sequential(
            ('stem_conv', nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=False, padding=1, stride=2)),
            ('stem_bn', nn.BatchNorm2d(out_channels)),
            ('stem_tanh', nn.Tanh())
        )
        
    def forward(self, x):
        x = self.layers(x)
        return x

