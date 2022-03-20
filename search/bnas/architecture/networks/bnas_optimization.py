import json
import random
from typing import Dict
import numpy as np
import copy

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



class Cells: 
    def __init__(self, cell_types,primitives, nodes=4, t=4,binary=True, latency=False, beta=None):
        '''
        primitives(dict): dictionary of operations for each cell
        nodes(int): number of nodes in a cell
        t(int): number of samples
        latency(bool): optimize for latency
        beta(float): a coefficient for latency. The expected value should be in range [0, 2], zero represents
        an optimization for validation score and two for latency.
        '''
        self.types = cell_types
        self.nodes_no = nodes
        self._set_edge_no()
        self.edges = {}
        self.latency= latency
        if latency:
            self.betas = {}
            beta = 1 if beta is None else in_range(beta)
            for cell_type in self.types:
                self.betas[cell_type] = beta
            self.load_latency_table(binary=binary)

        for cell_type in self.types:
            latency_table =  self.latency_table[cell_type] if latency else None
            self.edges[cell_type] = Edges(primitives[cell_type], self.edges_no,t,latency, latency_table)
        
        
    def set_worst_primitives(self, worst_idx:Dict):
        for cell_type in self.types:
            self.edges[cell_type].set_worst_primitives(worst_idx[cell_type].numpy().tolist())

    def sample(self):
        samples = {}
        for cell_type in self.types:
            samples[cell_type] = self.edges[cell_type].sample()
        return samples
    
    def get_training_idx(self):
        training_idx = {}
        for cell_type in self.types:
            training_idx[cell_type] = self.edges[cell_type].get_training_idx()
        return training_idx
    
    def reduce_space(self, betas=None):
        if self.latency:
            assert((isinstance(betas, dict) and len(betas) == 3) or (betas is None))
            betas = self.betas if betas is None else betas 
            for cell_type in self.types:
                self.edges[cell_type].reduce_space(in_range(betas[cell_type]))
        else:
            for cell_type in self.types:
                self.edges[cell_type].reduce_space()
    
    def update_score(self, sample, t, score):
        for cell_type in self.types:
            self.edges[cell_type].update_scores(sample[cell_type], t, score)
    
    def _set_edge_no(self):
        self.edges_no= self.nodes_no*2

    def load_latency_table(self, binary =True ,filename="arch_search/bnas_latency_table"):
        filename = filename + '_bn' if binary else filename +'_fp'
        with open(filename+'.json') as f:
            self.latency_table = json.load(f) 

class Edges:
    # two networks, train network and sample network
    def __init__(self, primitives, edges=4, t=4, latency=False, latency_table = None):
        self.edges_no = edges
        #self.primitives = copy.deepcopy(primitives) # will be returened as indices each time the train network works
        self.edges = [Edge(primitives,t, latency, latency_table) for _ in range(self.edges_no)]
    
    def get_training_idx(self):
        idx = []
        for edge in self.edges:
            idx.append(edge.get_training_primitives())
        return idx
    
    def update_scores(self, sample, t, score):
        for i in range(len(sample)):
            self.edges[i].update_score(sample[i], t, score)
    
    def reduce_space(self, beta=None):
        for edge in self.edges:
            edge.reduce_space(beta)
    
    def sample(self):
        samples = []
        for edge in self.edges:
            samples.append(edge.sample())
        return samples
    
    def set_worst_primitives(self, worst_idx):
        for i, edge in enumerate(self.edges):
            edge.set_worst_primitives(worst_idx[i])
            

class Edge:
    def __init__(self, primitives, t=4, latency=False, latency_table=None):
        self.primitives = copy.deepcopy(primitives)
        self.original_primitives_idx = [i for i in range(len(primitives))]
        self.working_primitives_idx = copy.deepcopy(self.original_primitives_idx)
        self.t_no = t
        self.latency = latency
        self.latency_table = latency_table
        #print(self.latency_table)
        #print(self.latency_table[str(0)])
        #print(isinstance(self.latency_table[str(0)],float))
        #print(self.latency_table[0])
        self._create_selection()
    
    def set_worst_primitives(self, worst_primitives):
        #print(worst_primitives)
        self.worst_primitives_idx = worst_primitives
        self.modifed_worst_primitives_idx = copy.deepcopy(worst_primitives)
        self.create_scores()
    
    def create_scores(self):
        self.scores = {id:[0 for _ in range(self.t_no)] for id in self.worst_primitives_idx}
    
    def reduce_space(self, beta=None):
        #print(self.scores)
        self.update_selection(beta)
        self.abandon_worst_primitive()

    def update_score(self, sample, t, score):
        self.scores[sample[0]][t] = score 
    

    def update_selection(self, beta=None):
        s_smaller_dict = self.calculate_s_smaller(beta)
        s_larger = self.calculate_s_larger(s_smaller_dict)
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
        if len(self.modifed_worst_primitives_idx) == 0:
            self.modifed_worst_primitives_idx = copy.deepcopy(self.worst_primitives_idx)
        sample = random.sample(self.modifed_worst_primitives_idx, 1) 
        self.modifed_worst_primitives_idx.remove(sample[0])
        return sample

    def calculate_s_smaller(self, beta=None):
        y_dash = {}
        if self.latency:
            y_dash_l = {}
        for sample in self.worst_primitives_idx:
            y_dash[sample] = sum(self.scores[sample])/self.t_no
            if self.latency:
                y_dash_l[sample] = self.latency_table[str(sample)]
        s_smaller = softmax(list(y_dash.values()))
        s_smaller_multi_obj = s_smaller
        if self.latency:
            s_smaller_l = 1 - softmax(list(y_dash_l.values()))
            s_smaller_multi_obj = [((2-beta)* acc+ beta*l)/2 for acc, l in zip(s_smaller, s_smaller_l) ]
        s_smaller_dict = {id:v for id, v in zip(self.worst_primitives_idx, s_smaller_multi_obj)}
        return s_smaller_dict
    
    def calculate_s_larger(self, s_smaller_dict):
        s_smaller = list(s_smaller_dict.values())
        s_larger = 0.5 * (max(s_smaller) + (1/len(s_smaller)) * sum(s_smaller)) # scalar
        return s_larger
    
    def get_training_primitives(self):
        return self.working_primitives_idx
    
    def _create_selection(self):
        self.selection = {id:0 for id in self.original_primitives_idx}