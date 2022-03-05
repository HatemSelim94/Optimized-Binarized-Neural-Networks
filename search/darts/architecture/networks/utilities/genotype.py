import json
import torch
import torch.nn as nn
from graphviz import Digraph
import os

def plot_cell(idx,type, dir, view=False):
    g = Digraph(
      format='pdf',
      edge_attr=dict(fontsize='20', fontname="times"),
      node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5', penwidth='2', fontname="times"),
      engine='dot')
    g.body.extend(['rankdir=LR'])

    nodes = 4
    start = 1
    end = start+2
    counter = 1
    for i in range(start, nodes+2+1):
        if i <=2:
            g.node(f'c_{{k-{i}}}', fillcolor='darkseagreen2')
        else:
            g.node(f'n{i-2}', fillcolor='lightblue')
    
    for j in range(1, nodes+1):
        for node in range(start, end):
            if node <=2:
                if node == 1:
                    source = f'c_{{k-{2}}}'
                else:
                    source = f'c_{{k-{1}}}'
            else:    
                source = f'n{node-2}'
            destination = f'n{end-2}'
            g.edge(source, destination, label=f'{counter} op{idx[node+j-2][0]}', color='gray')
            counter +=1
        start +=1
        end += 1
    
    g.node("c_{k}", fillcolor='palegoldenrod')
    for i in range(1,nodes+1):
        g.edge(f'n{i}', "c_{k}", fillcolor="gray")
    filename = f'darts_relaxed_cell_modified_{type}'
    filepath = os.path.join(dir, filename)
    g.render(filepath,view=view)

def save_cell_idx(cell_idx:torch.Tensor, type,dir, epoch=0):
    filename = f'cell_{type}_genotype_{epoch}' 
    filepath = os.path.join(dir, filename)
    with open(filepath,'w') as f:
        json.dump(cell_idx.tolist(),f)
    

def save_genotype(alphas, dir, epoch=0):
    types = ['n','r','u']
    def best_pick(alphas):
        with torch.no_grad():
            best_alphas = torch.topk(alphas, 1, dim=-1)
        return best_alphas
    best_alphas, indices = best_pick(alphas)
    for cell_idx, type in zip(indices, types):
        plot_cell(cell_idx, type,dir)
        save_cell_idx(cell_idx,type,dir,epoch)
    return indices

def load_genotype(dir):
    with open(dir+'.json', 'r') as f:
        indices = json.load(f)
    return indices
    

if __name__ == '__main__':
    nodes_num = 4
    edges_num = 2
    ops_num = 5
    alphas = torch.randn((3, nodes_num*edges_num ,ops_num), requires_grad=True, device='cpu')
    #nn.init.constant_(alphas, 1/ops_num)
    idx = save_genotype(alphas)
    for id in idx:
        print(id)
