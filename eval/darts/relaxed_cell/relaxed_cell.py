from graphviz import Digraph

def plot_relaxed_cell():
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
                source = f'c_{{k-{node}}}'
            else:    
                source = f'n{node-2}'
            if counter == 3:
                 source = f'c_{{k-{node-1}}}'
            destination = f'n{end-2}'
            g.edge(source, destination, label=f'{counter}', color='gray')
            counter +=1
        start +=1
        end += 1
    
    g.node("c_{k}", fillcolor='palegoldenrod')
    for i in range(1,nodes+1):
        g.edge(f'n{i}', "c_{k}", fillcolor="gray")
    
    g.render('darts_relaxed_cell_modified',view=True)

if __name__ == '__main__':
    plot_relaxed_cell()