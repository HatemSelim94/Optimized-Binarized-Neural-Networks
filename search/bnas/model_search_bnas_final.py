import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

from .binarized_operations import BnnConv, BnnTConv, ConvBnReLU, TConvBnReLU
from .genotypes import NORMAL_REDUCTION_PRIMITIVES, UPSAMPLING_PRIMITIVES
from .genotypes import SegGenotype
from .binarized_operations import Normal_Reduction_OPS, Upsampling_OPS, FP_Normal_Reduction_OPS, FP_Upsampling_OPS 
from .model_search_bnas_sample import SampleNetwork
#from .bnas_optimization import Cells
from .bnas_optimization_multi_obj import Cells
from .utils import plot_tensor_dist
device = torch.device('cuda') # cuda

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups
    
    # reshape
    x = x.view(batchsize, groups, 
        channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class MixedOp(nn.Module): # each edge has mixed ops in the relaxed cell

  def __init__(self, C, stride, k=4,binary=True):
    super(MixedOp, self).__init__()
    self.k = k
    C = C//self.k
    self._ops = nn.ModuleList()
    ops = Normal_Reduction_OPS if binary else FP_Normal_Reduction_OPS
    for primitive in NORMAL_REDUCTION_PRIMITIVES:
      op = ops[primitive](C, stride, False, ) # returns an operation, C is the number of input channels and output channels
      if 'pool' in primitive: # if pool in an op name like max_pool 
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False)) # add batchnorm afterwards
        # Learnable affine parameters in all batch normalizations are disabled 
        # during the search process to avoid rescaling the outputs of the candidate operations.

      self._ops.append(op)
    self.maxpool = nn.MaxPool2d(2,2)
    #self.shuffle = nn.ChannelShuffle(self.k)

  def forward(self, x, weights, training_idx):
    channels_no = x.shape[1]
    # the first k channels are selected
    k_channels = x[:, :channels_no//self.k,:,:]
    rest_of_channels = x[:, channels_no//self.k:,:,:]
    # perform the operations on the first k-channels only
    op_output = sum(w * op(k_channels) for id, (w, op) in enumerate(zip(weights, self._ops)) if id in training_idx) # the sum is taken and weights here are the alphas
    # merge the channels and then chuffel the output of the merge
    # to prepare for the merge, apply any operation to the intact channels to match the shape (h, w)
    if op_output.shape[2:] == rest_of_channels.shape[2:]:
      output = torch.cat((op_output, rest_of_channels), dim=1)
    else: 
      output1 = self.maxpool(rest_of_channels)
      output = torch.cat((op_output, output1),dim=1)

    #output = self.shuffle(output)
    output = channel_shuffle(output, self.k)
    return output


class UpsampleMixedOp(nn.Module): # each edge has mixed ops in the relaxed cell

  def __init__(self, C, stride, k = 4,binary= True):
    super(UpsampleMixedOp, self).__init__()
    self.k = k
    C = C//self.k
    self._ops = nn.ModuleList()
    ops = Upsampling_OPS if binary else FP_Upsampling_OPS
    for primitive in UPSAMPLING_PRIMITIVES:
      op = ops[primitive](C, stride, False) # returns an operation, C is the number of input channels and output channels
      if 'pool' in primitive: # if pool in an op name like max_pool 
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False)) # add batchnorm afterwards
        # Learnable affine parameters in all batch normalizations are disabled 
        # during the search process to avoid rescaling the outputs of the candidate operations.

      self._ops.append(op)
    self.upsample = nn.Upsample(scale_factor=2, align_corners=False, mode='bilinear')
    #self.shuffle = nn.ChannelShuffle(self.k)

  def forward(self, x, weights, training_idx):
    channels_no = x.shape[1]
    # the first k channels are selected
    k_channels = x[:, :channels_no//self.k,:,:]
    rest_of_channels = x[:, channels_no//self.k:,:,:]
    # perform the operations on the first k-channels only
    op_output = sum(w * op(k_channels) for id, (w, op) in enumerate(zip(weights, self._ops)) if id in training_idx) # the sum is taken and weights here are the alphas
    # merge the channels and then chuffel the output of the merge
    # to prepare for the merge, apply any operation to the intact channels to match the shape (h, w)
    if op_output.shape[2:] == rest_of_channels.shape[2:]:
      output = torch.cat((op_output, rest_of_channels), dim=1)
    else: 
      output1 = self.upsample(rest_of_channels)
      output = torch.cat((op_output, output1), dim=1)
    #output = self.shuffle(output)
    output = channel_shuffle(output, self.k)
    return output



class Cell(nn.Module):
  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, upsample=False, upsample_prev=False, binary=True):
    super(Cell, self).__init__()
    '''
    inputs: 
          steps: number of intermediate nodes in the cell graph
          C_prev_prev: num of channels of the tensor coming from the prev prev cell
          C_prev: num of channels of the tensor coming from the prev prev cell
          C: num of channels of this (self) cell
          reduction: True or False if this cell is a reduction cell (to match the number of channels)
          reduction_prev: if the prev prev cell is a reduction cell (to match the number of channels)
    '''
    self.binary = binary
    self.reduction = reduction
    self.upsample = False
    preprocess0 = BnnConv if self.binary else ConvBnReLU
    preprocess1 = BnnTConv if self.binary else TConvBnReLU
    if reduction_prev:
      self.preprocess0 = preprocess0(C_prev_prev, C, 1,stride=2,affine=False)  # downsize the image dimension and match the output channels to C 
    elif upsample_prev:
      self.preprocess0 = preprocess1(C_prev_prev, C, 1, stride=2,padding=0, affine=False)
    else:
      self.preprocess0 = preprocess0(C_prev_prev, C, 1, 1,0, affine=False) # match number of channels
    self.preprocess1 = preprocess0(C_prev, C, 1, 1, 0, affine=False)  # match number of chnnels by 1*1 conv
    self._steps = steps
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    for i in range(self._steps):
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(C, stride, binary = self.binary)
        self._ops.append(op)

  def forward(self, s0, s1, weights, idx):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    offset = 0
    for i in range(self._steps):
      s = sum(self._ops[offset+j](h, weights[offset+j],idx[offset+j]) for j, h in enumerate(states))
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1) # channels number is multiplier "4" * C "C_curr"
  

class UpsampleCell(nn.Module):
  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, upsample, upsample_prev, binary=True):
    super(UpsampleCell, self).__init__()
    '''
    inputs: 
          steps: number of intermediate nodes in the cell graph
          C_prev_prev: num of channels of the tensor coming from the prev prev cell
          C_prev: num of channels of the tensor coming from the prev prev cell
          C: num of channels of this (self) cell
          reduction: True or False if this cell is a reduction cell (to match the number of channels)
          reduction_prev: if the prev prev cell is a reduction cell (to match the number of channels)
    '''
    self.binary = binary 
    self.upsample = upsample
    self.reduction = False
    preprocess0 = BnnConv if self.binary else ConvBnReLU
    preprocess1 = BnnTConv if self.binary else TConvBnReLU
    if upsample_prev:
      self.preprocess0 = preprocess1(C_prev_prev, C, 1,stride=2,affine=False)  # downsize the image dimension and match the output channels to C 
    else:
      self.preprocess0 = preprocess0(C_prev_prev, C, 1, 1,0, affine=False) # match number of channels # or BnnConv
    self.preprocess1 = preprocess0(C_prev, C, 1, 1, 0, affine=False)  # match number of chnnels by 1*1 conv
    self._steps = steps
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    for i in range(self._steps):
      for j in range(2+i):
        stride = 2 if upsample and j < 2 else 1
        op = UpsampleMixedOp(C, stride, binary = self.binary)
        self._ops.append(op)
  def forward(self, s0, s1, weights, idx):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    offset = 0
    for i in range(self._steps):
      s = sum(self._ops[offset+j](h, weights[offset+j], idx[offset+j]) for j, h in enumerate(states))
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1) # channels number is multiplier "4" * C "C_curr"
  

class Stem(nn.Module):
    def __init__(self, C=64):
        super(Stem, self).__init__()
        self.conv1 = nn.Conv2d(3, C//2, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(C//2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(C//2, C//2, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(C//2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(C//2, C, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(C)
        self.relu3 = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu2(output)
        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu3(output)
        #output = self.maxpool(output)
        return output

# basic classification network,  normal and downsampling convolutions and at the end a classifier(linear layer) is used
class Network(nn.Module):
  '''
    Classification network:
    This network is used to search for a good cell arch only. It is also can be seen as a proxy network. The alphas (edges learnable params)
    are shared between the all the cells  

  '''
  def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3,t=4,binary=True,latency=False, beta=None):
    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion 
    self._steps = steps # intermediate states 0,1,2,3 in DARTS Cells
    self._multiplier = multiplier # channels multiplier
    self.binary = binary
    #self.edge_history = PrimitivesHistory({'noraml':NORMAL_REDUCTION_PRIMITIVES,'reduction': NORMAL_REDUCTION_PRIMITIVES,'upsample':UPSAMPLING_PRIMITIVES})
    primitives = {'normal':NORMAL_REDUCTION_PRIMITIVES,'reduction': NORMAL_REDUCTION_PRIMITIVES,'upsample':UPSAMPLING_PRIMITIVES}
    self.cells_primitives = Cells(primitives, nodes=steps, t=t, binary=binary, latency=latency, beta=beta)
    #self.cells_primitives = Cells(primitives, nodes=steps, t=t)

    C_curr = stem_multiplier*C
    # downsize the image dimension by half and increase channels number
    #self.stem = nn.Sequential(
    #  nn.Conv2d(3, C_curr, 3,stride=2, padding=1, bias=False), #, stride=2
    #  nn.BatchNorm2d(C_curr),
    #  nn.LeakyReLU() # 50%
    #)
    self.stem = Stem(C_curr)
 
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
      if i in [layers//3, 2*layers//3]:  #at certain layers do reduction
        C_curr *= 2  # double channels
        reduction = True # downsample the image
        cell_type = 'reduction'  
      else:
        reduction = False
        cell_type = 'reduction'
      
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, binary=self.binary) # construct a cell
      
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr # In this i-th iteration the output of the cell is multiplier * C_curr 
      # because the outputs of the intermediate nodes (each of which has C_curr number of channels) are concatenated, therefore, 
      # the C_prev = multiplier * C_curr 
    
    upsample_prev = False
    for i in range(layers):
      if i in [layers//3, 2*layers//3]:
        C_curr = C_curr //2
        upsample = True
        cell = UpsampleCell(steps, multiplier, C_prev_prev, C_prev, C_curr, upsample, upsample_prev, binary=self.binary)
      else:
        upsample = False
        cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, False, False,upsample, upsample_prev, binary=self.binary)
      upsample_prev = upsample
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr    
    self._initialize_alphas()
    self.final_layer = nn.Conv2d(C_prev, num_classes, 1, padding=0)

  def new(self):
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion, binary= self.binary).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new
  
  def forward(self, input):
    training_idx = self.cells_primitives.get_training_idx()
    input_shape = input.shape
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      if cell.reduction:
        weights = F.softmax(self.alphas_reduce, dim=-1)
        idx = training_idx['reduction']
      elif cell.upsample:
        weights = F.softmax(self.alphas_upsample, dim=-1)
        idx = training_idx['upsample']
      else:
        weights = F.softmax(self.alphas_normal, dim=-1)
        idx = training_idx['normal']

      s0, s1 = s1, cell(s0, s1, weights, idx)
      #plot_tensor_dist(s1, save=True, file_name=f'{i}')
    output = self.final_layer(s1)
    if output.shape[-2:] != input_shape[-2:]:
      output = F.interpolate(output,size= input_shape[-2:], align_corners=False, mode='bilinear')
    return output

  def _loss(self, input, target):
    # returns the loss
    logits = self(input)
    return self._criterion(logits, target) 

  def _initialize_alphas(self):
    k = sum(1 for i in range(self._steps) for n in range(2+i)) # number of learnable edges (number of learnable edges is 14 in CIFAR DARTS CNN )
    num_ops = len(NORMAL_REDUCTION_PRIMITIVES)
    upsample_ops = len(UPSAMPLING_PRIMITIVES)

    self.alphas_normal = torch.randn((k, num_ops), requires_grad=True, device=device)
    self.alphas_reduce = torch.randn((k, num_ops), requires_grad=True, device=device)
    self.alphas_upsample = torch.randn((k, upsample_ops), requires_grad=True, device=device)
    self._arch_parameters = [
      self.alphas_normal,
      self.alphas_reduce,
      self.alphas_upsample
    ]
  ###################################################################
  def select_worst_primitives(self):
    with torch.no_grad():
      #_, ops = self._arch_parameters[0].shape
      training_idx = self.cells_primitives.get_training_idx()
      ops_num = len(training_idx['normal'][0]) # k in the paper
      worst_ops = math.ceil(ops_num/2)
      worst_idx = {}
      for i, cell_type in enumerate(['normal', 'reduction','upsample']):
          cell_weights = self._arch_parameters[i].detach().clone().cpu() 
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
    return len(worst_idx['normal'][0])
  
  def get_k(self):
    training_idx = self.cells_primitives.get_training_idx()
    k = len(training_idx['normal'][0]) # k in the paper
    return k

  def sample_network(self):
    s = self.cells_primitives.sample()
    self.sampled_primitives = s
    return self.new_full_model(s)


  def new_full_model(self, sample_primitives_idx):
    model_new = SampleNetwork(self._C, self._num_classes, self._layers, self._criterion, binary= self.binary, primitive_idx=sample_primitives_idx).cuda()
    model_new.load_state_dict(self.state_dict(), strict=False)
    return model_new
  
  def update_score(self, t, score):
    self.cells_primitives.update_score(self.sampled_primitives, t, score)
  
  def reudce_space(self):
    self.cells_primitives.reduce_space()

  #def update_edge_primitives(self):
  #################################################################
  def arch_parameters(self):
    '''
      return arch parameters (Alphas)
    '''
    return self._arch_parameters
  

  def genotype(self):
    # the idea is to sort edges since each edge contains only one operation
    def _parse(weights, cell_training_idx,reduction_normal= True):
      current_primitives = NORMAL_REDUCTION_PRIMITIVES if reduction_normal else  UPSAMPLING_PRIMITIVES
      gene = []
      n = 2 # number of predecessors (nodes) of the first intermediate node
      start = 0
      all_li = []
      for i in range(self._steps): # for each intermediate node
        end = start + n # weights indices that belong to the edges of the current intermediate node(_steps)
        W = weights[start:end].copy()
        # sort and get the best two edges for each node based on the best operation in each edge
        # each edge key is the max or the best operation in this edge
        # based on the best operation's alpha in each edge, the best two edges are selected
        w = {}
        best = None
        for x in range(len(W)):
          for k in range(len(W[x])):
            if (k != current_primitives.index('none')) and (k in cell_training_idx[x]):
              w[x] = W[x][k]
        edges = [k for k, v in sorted(w.items(), key=lambda item:item[1], reverse=True)][:2]
              #li.append(W[x][k])
        #all_li.append(li)
        #sort w based on val and get the key for the best two edges and thats it

        #edgs = [W[x][k] for k in range(len(W[x])) for x in range(W) if (k != current_primitives.index('none')) and (k in cell_training_idx[x])]
        #edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if (k != current_primitives.index('none')) and (k in cell_training_idx[x])))[:2]
        # get the best operation's name in each one of the selected edges
        ##for j in edges:
          #k_best = None
          #for k in range(len(W[j])):
          #  if k != current_primitives.index('none'):
          #    if k_best is None or W[j][k] > W[j][k_best]:
          #      k_best = k
        for j in edges:
          gene.append((current_primitives[cell_training_idx[j][0]], j)) # the gene is formated similar to the formatting in genotyps.py
        start = end
        n += 1
      #print(all_li)
      return gene
    
    # get normal/reduction cell genotype
    training_idx = self.cells_primitives.get_training_idx()
    gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy(), training_idx['normal'])
    gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy(), training_idx['reduction'])
    gene_upsample = _parse(F.softmax(self.alphas_upsample, dim=-1).data.cpu().numpy(), training_idx['upsample'], False)

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = SegGenotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat,
      upsample = gene_upsample, upsample_concat = concat
    )
    return genotype


if __name__ == '__main__':
  from visualize import GenotypePlotter
  cross_etropy_loss = nn.CrossEntropyLoss() 
  net = Network(5, 3, 6, cross_etropy_loss, binary = True)
  fp_net = Network(5, 3, 6, cross_etropy_loss, binary = False)
  #print(net)
  gene = net.genotype()
  #from visualize import plot_genotype
  #print(len(gene.upsample))
  print(gene.upsample)
  print(gene.upsample_concat)
  print(gene.reduce)
  print(gene.reduce_concat)
  print(gene.normal)
  print(gene.normal_concat)
  #plot(gene.upsample, 'upsample')
  #plot(gene.reduce, 'reduc')
  #input = torch.randn([1,3,512,512]).to(device)
  #out = net(input)
  #print(out.shape)
  #gene_plotter = GenotypePlotter('./')
  #gene_plotter.store_genome(gene,1)
  #cell = UpsampleCell(4,4, 20, 10, 10, True, False)
  #co = 0
  #for p in cell.parameters():
  #  if hasattr(p, 'bin'):
  #    co+=1
  #print(co)
