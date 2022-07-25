import os
import json
import random
import numpy as np
import copy
import torch

from .utilities.genotype import plot_cell
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

class Bnas: 
    def __init__(self, primitives=None, edges=14,nodes=4, t=4, unique_cell_types=['r','n','u'], edges_per_node=2):
        '''
        primitives(dict): dictionary of operations for each cell
        edges(int): number of edges in a cell
        nodes(int): number of nodes in a cell
        t(int): number of sampled nets to generate scores
        '''
        if primitives is None:
            primitives = {}
            primitives['r'] = NR_PRIMITIVES
            primitives['n'] = NR_PRIMITIVES
            primitives['u'] = UP_PRIMITIVES

        self.edges_per_node = edges_per_node
        self.nodes_no = nodes
        self.edges_no = edges
        self.unique_cell_types = unique_cell_types
        self.cell_edges = {}

        for cell_type in self.unique_cell_types:
            self.cell_edges[cell_type] = BnasEdges(primitives[cell_type], self.edges_no,t)
        
        
    def set_worst_primitives(self, worst_idx:torch.Tensor):
        for cell_type in self.unique_cell_types:
            self.cell_edges[cell_type].set_worst_primitives(worst_idx[cell_type].numpy().tolist())

    def sample(self):
        samples = {}
        for cell_type in self.unique_cell_types:
            samples[cell_type] = self.cell_edges[cell_type].sample()
        return samples
    
    def get_training_idx(self):
        training_idx = {}
        for cell_type in self.unique_cell_types:
            training_idx[cell_type] = self.cell_edges[cell_type].get_training_idx()
        return training_idx
    
    def reduce_space(self):
        for cell_type in self.unique_cell_types:
            self.cell_edges[cell_type].reduce_space()
    
    def update_score(self, sample, t, score):
        for cell_type in self.unique_cell_types:
            self.cell_edges[cell_type].update_scores(sample[cell_type], t, score)
    
    def save_genotype(self, args):
        genotypes = {}
        train_idx = self.get_training_idx()
        for type, cell_idx in train_idx.items():
            start = 0
            end = 2
            genotype_cell = {'ops':[], 'states':[]}
            for n in range(self.nodes_no):
                edges = self.cell_edges[type].edges[start:end]
                edges_idx = cell_idx[start:end]
                (op1,op2), states= max_two(edges, edges_idx)
                genotype_cell['ops'].append(op1)
                genotype_cell['ops'].append(op2)
                genotype_cell['states'].append(states)
                start = end 
                end+=n+3
            genotypes[type] = genotype_cell
            save_gene(genotype_cell, type, os.path.join(args.experiment_path, args.experiment_name))
            plot_cell(genotype_cell, type, os.path.join(args.experiment_path, args.experiment_name),nodes=self.nodes_no, epoch='final')


class BnasEdges:
    # two networks train network and sample network
    def __init__(self, primitives, edges=4, t=4):
        self.edges_no = edges
        self.edges = [BnasEdge(primitives,t) for _ in range(self.edges_no)]
    
    def get_training_idx(self):
        idx = []
        for edge in self.edges:
            idx.append(edge.get_training_primitives())
        return idx
    
    def update_scores(self, sample, t, score):
        for i in range(len(sample)):
            self.edges[i].update_score(sample[i], t, score)
    
    def reduce_space(self):
        for edge in self.edges:
            edge.reduce_space()
    
    def sample(self):
        samples = []
        for edge in self.edges:
            samples.append(edge.sample())
        return samples
    
    def set_worst_primitives(self, worst_idx):
        for i, edge in enumerate(self.edges):
            edge.set_worst_primitives(worst_idx[i])
            

class BnasEdge:
    def __init__(self, primitives, t=4):
        self.primitives = copy.deepcopy(primitives)
        self.original_primitives_idx = [i for i in range(len(primitives))]
        self.working_primitives_idx = copy.deepcopy(self.original_primitives_idx)
        self.t_no = t
        self._create_selection_likelihood()
    
    def set_worst_primitives(self, worst_primitives):
        #print(worst_primitives)
        self.worst_primitives_idx = worst_primitives
        self.modifed_worst_primitives_idx = copy.deepcopy(worst_primitives)
        self.create_scores()
    
    def create_scores(self):
        self.scores = {id:[0 for _ in range(self.t_no)] for id in self.worst_primitives_idx}
        self.latency = {id:[0 for _ in range(self.t_no)] for id in self.worst_primitives_idx}
        self.bops = {id:[0 for _ in range(self.t_no)] for id in self.worst_primitives_idx}
        self.size = {id:[0 for _ in range(self.t_no)] for id in self.worst_primitives_idx}
    def reduce_space(self):
        #print(self.scores)
        self.update_selection()
        self.abandon_worst_primitive()

    def update_score(self, sample, t, score, latency=0, bops=0,size=0):
        self.scores[sample[0]][t] = score 
        self.latency[sample[0][t]] = latency
        self.bops[sample[0][t]] = bops
        self.size[sample[0][t]] = size
    

    def update_selection(self):
        s_smaller_dict = self.calculate_s_smaller()
        s_larger = self.calculate_s_larger(s_smaller_dict) #scalar
        for id in self.selection.keys():
            if id in self.worst_primitives_idx:
                self.selection[id] = self.selection[id]/2 + s_smaller_dict[id]
            else:
                self.selection[id] = self.selection[id]/2 + s_larger 
    
    def abandon_worst_primitive(self):
        worst_primtive_id = min(self.selection, key=self.selection.get)
        self.working_primitives_idx.remove(worst_primtive_id)
        del self.selection[worst_primtive_id]


    def sample(self):
        # the sampling procedure is repeated k/2 * t times
        # generate a new one if its consumed  
        if len(self.modifed_worst_primitives_idx) == 0:
            self.modifed_worst_primitives_idx = copy.deepcopy(self.worst_primitives_idx)
        sample = random.sample(self.modifed_worst_primitives_idx, 1) 
        self.modifed_worst_primitives_idx.remove(sample[0])
        return sample

    def calculate_s_smaller(self):
        y_dash = {}
        for sample in self.worst_primitives_idx:
            y_dash[sample] = sum(self.scores[sample])/self.t_no
        s_smaller = softmax(list(y_dash.values()))
        s_smaller_dict = {id:v for id, v in zip(self.worst_primitives_idx, s_smaller)}
        return s_smaller_dict
    
    def calculate_s_larger(self, s_smaller_dict):
        s_smaller = list(s_smaller_dict.values())
        s_larger = 0.5 * (max(s_smaller) + (1/len(s_smaller)) * sum(s_smaller)) # scalar
        return s_larger
    
    def get_training_primitives(self):
        return self.working_primitives_idx
    
    def _create_selection_likelihood(self):
        self.selection = {id:0 for id in self.original_primitives_idx}


def sample_from_lists_(l):
    s = []
    for i, v in enumerate(l):
        sampled_num = random.sample(v, 1)
        s.append(sampled_num)
        v.remove(sampled_num[0])
    return s

def softmax(l):
    exp_output = np.exp(l)
    output = exp_output/sum(exp_output)
    return output

def in_range(num, min_limit=0, max_limit=2):
    return min(max(num, min_limit), max_limit)

def max_two(edges, idx):
    #print(idx)
    ops  = [0, 0 ]
    states = [0, 0]
    max_num_1 = -100
    max_num_2 = -100
    for i, (edge,id) in enumerate(zip(edges, idx)):
        max_candidate = edge.selection[id[0]]
        if max_num_1 < max_candidate:
            max_num_2 = max_num_1
            max_num_1 = max_candidate 
            ops[-1] = ops[0]
            ops[0] = id[0]
            states[-1] = states[0]
            states[0] = i
        elif max_num_2 < max_candidate:
            max_num_2 = max_candidate
            states[-1] = i
            ops[-1] = id[0]
    return tuple(ops), states

def save_gene(genotype, type, dir, epoch='final'):
    filefolder = f'darts_relaxed_cell_{type}' 
    filepath = os.path.join(dir, filefolder)
    filename = os.path.join(filepath,f'genotype_{epoch}')
    best_filename = os.path.join(filepath,f'genotype_best')
    with open(filename+'.json','w') as f:
        json.dump(genotype,f)
    with open(best_filename+'.json','w') as f:
        json.dump(genotype,f)
