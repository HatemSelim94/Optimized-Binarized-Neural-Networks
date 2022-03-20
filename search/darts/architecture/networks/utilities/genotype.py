import json
import torch
import torch.nn as nn
from graphviz import Digraph
import os


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



def plot_cell(idx,cell_type, dir,epoch, view=False, nodes=4):
    ops = NR_PRIMITIVES if cell_type in ['n','r'] else UP_PRIMITIVES
    g = Digraph(
      format='pdf',
      edge_attr=dict(fontsize='20', fontname="times"),
      node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5', penwidth='2', fontname="times"),
      engine='dot')
    g.body.extend(['rankdir=LR'])

    nodes = nodes
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
            #g.edge(source, destination, label=f'{counter} op{idx[node+j-2][0]}', color='gray')
            g.edge(source, destination, label=f'{ops[idx[node+j-2][0]]}', color='gray')
            counter +=1
        start +=1
        end += 1
    
    g.node("c_{k}", fillcolor='palegoldenrod')
    for i in range(1,nodes+1):
        g.edge(f'n{i}', "c_{k}", fillcolor="gray")
    filefolder = f'darts_relaxed_cell_modified_{cell_type}'
    filepath = os.path.join(dir, filefolder)
    if not os.path.exists(filepath): 
        os.makedirs(filepath)
    filename = os.path.join(filepath, f'genotype_{epoch}')
    g.render(filename,view=view)

def save_cell_idx(cell_idx:torch.Tensor, type, dir, epoch=0):
    filefolder = f'darts_relaxed_cell_modified_{type}' 
    filepath = os.path.join(dir, filefolder)
    filename = os.path.join(filepath,f'genotype_{epoch}')
    best_filename = os.path.join(filepath,f'genotype_best')
    with open(filename+'.json','w') as f:
        json.dump(cell_idx.tolist(),f)
    with open(best_filename+'.json','w') as f:
        json.dump(cell_idx.tolist(),f)
    

def save_genotype(alphas, dir,epoch=0, nodes=4, types=['n','r','u'], best=1):
    def best_pick(alphas):
        best_alphas = []
        for alpha in alphas:
            with torch.no_grad():
                best_alphas.append(torch.topk(alpha, best, dim=-1)[-1])
        return best_alphas
    indices = best_pick(alphas)
    for cell_idx, type in zip(indices, types):
        plot_cell(cell_idx, type,dir, epoch, nodes=nodes)
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
