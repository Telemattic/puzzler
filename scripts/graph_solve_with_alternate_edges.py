import argparse
import collections
import itertools
import json
import networkx as nx
import operator
import os
import subprocess
import tempfile

def find_longest_cycles(G):

    l_max = 0
    longest_cycle = []
    for cycle in nx.simple_cycles(G):
        l = len(cycle)
        if l > l_max:
            l_max = l
            longest_cycle = cycle

    return longest_cycle

def make_graph(ofile, data, opt):

    G = nx.DiGraph()

    extra_edges = []
    for prev, choices in data['scores'].items():
        G.add_node(prev)
        for rank, (mse, _, _, succ) in enumerate(sorted(choices), start=1):
            if mse < opt['max_error']:
                if rank == 1:
                    G.add_edge(prev, succ, mse=mse)
                else:
                    extra_edges.append((mse, prev, succ))

    extra_edges.sort()

    l_max = 0
    longest_cycles = []
    for cycle in nx.simple_cycles(G):
        l = len(cycle)
        if l < l_max:
            continue

        cost = nx.path_weight(G, cycle + [cycle[0]], 'mse')
        s = ' '.join(cycle)
        print(f"Found cycle len={l}, {cost=:.3f}, path={s}")
        
        if l > l_max:
            l_max = l
            longest_cycles = [cycle]
        else:
            longest_cycles.append(cycle)

    for n in range(opt['extra_edges']):
        mse, prev, succ = extra_edges[n]
        G.add_edge(prev, succ, mse=mse)

        print(f"Adding extra edge {n}: {prev} -> {succ}")

        # we added an edge from prev -> succ, so now we're looking for
        # the possibility of a cycle that starts at succ and returns
        # to prev (which may not exist of course):
        for cycle in nx.all_simple_paths(G, succ, prev):
            l = len(cycle)
            if l < l_max:
                continue

            cost = nx.path_weight(G, cycle + [cycle[0]], 'mse')
            s = ' '.join(cycle)
            print(f"Found cycle len={l}, {cost=:.3f}, path={s}")
            
            if l > l_max:
                l_max = l
                longest_cycles = [l]
            else:
                longest_cycles.append(l)

    if opt['merge']:
        simple_nodes = set()
        for node in G.nodes:
            if G.in_degree(node) == 1 and G.out_degree(node) == 1:
                simple_nodes.add(node)

        for node in simple_nodes:
            pred = next(G.predecessors(node))
            succ = next(G.successors(node))
            if pred in simple_nodes and succ in simple_nodes:
                # print(f"Removing {node=}, {pred=} {succ=}")
                mse = G.edges[pred,node]['mse'] + G.edges[node,succ]['mse']
                G.remove_node(node)
                G.add_edge(pred, succ, style='dotted', mse=mse)

    print("digraph G {", file=ofile)
    
    for n in G.nodes:
        print(f"  {n}", file=ofile)
        
    for e in G.edges:
        src, dst = e
        attr = G.edges[e]
        props = []
        
        style = attr.get('style', 'solid')
        
        props.append(f'style={style}')
        
        if G.out_degree(src) > 1:
            mse = attr.get('mse',-1)
            props.append(f'label="{mse:.3f}"')

        props = '[' + ' '.join(props) + ']'
            
        print(f"  {src} -> {dst} {props}", file=ofile)

    print("}", file=ofile)

def make_pdf(dotpath, pdfpath):
    # "C:\Program Files\Graphviz\bin\dot.exe" -Tpdf -Gsize=8,10.5 -Gpage=8.5,11 -o <pdf_path> <dot_path>
    exe = r'C:\Program Files\Graphviz\bin\dot.exe'
    args = [exe, '-Tpdf', '-Gsize=8,10.5', '-Gpage=8.5,11', '-o', pdfpath, dotpath]
    print(' '.join(args))
    subprocess.run(args)

def main():

    parser = argparse.ArgumentParser(
        prog='GraphIt',
        description='Graph piece connectivity data')
    parser.add_argument('filename', help='input json file containing connectivity data')
    parser.add_argument('-n', '--extra-edges', help='add the lowest error edges in addition to the rank 1 edges', type=int, default=0)
    parser.add_argument('-e', '--max-error', help='maximum error, any edges with more error are always pruned', type=float, default=60.)
    parser.add_argument('-m', '--merge', help='merge runs of nodes with only good edges in and out',
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('-o', '--output', help='output filename', required=True)

    args = parser.parse_args()

    with open(args.filename, 'r') as f:
        s = f.read()
        data = json.loads(s)

    opt = {'merge': args.merge, 'max_error': args.max_error, 'extra_edges': args.extra_edges}
    
    if args.output.endswith('.pdf'):

        # delete_on_close depends on python 3.12
        with tempfile.NamedTemporaryFile(dir=r'C:\Temp', prefix='graphit_', suffix='.dot',
                                         mode='w', newline='\n', delete=False) as dotfile:
            make_graph(dotfile, data, opt)
            dotfile.close()
            make_pdf(dotfile.name, args.output)
            os.unlink(dotfile.name)

    elif args.output.endswith('.dot'):

        with open(args.output, 'w') as dotfile:
            make_graph(dotfile, data, opt)

if __name__ == '__main__':
    main()
