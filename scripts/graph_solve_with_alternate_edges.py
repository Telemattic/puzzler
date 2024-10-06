import argparse
import collections
import itertools
import json
import networkx as nx
import operator
import os
import subprocess
import tempfile

def make_graph(ofile, data, opt):

    G = nx.DiGraph()

    rank1_edges = []
    extra_edges = []
    for prev, choices in data['scores'].items():
        for rank, (mse, _, _, succ) in enumerate(sorted(choices), start=1):
            if mse < opt['max_error']:
                if rank == 1:
                    rank1_edges.append((prev, succ, rank, mse))
                else:
                    extra_edges.append((prev, succ, rank, mse))

    extra_edges.sort(key=operator.itemgetter(3))
    for prev, succ, rank, mse in rank1_edges + extra_edges[:opt['extra_edges']]:
        G.add_edge(prev, succ, rank=rank, mse=mse)

    n_cycles = 0
    l_max = 0
    longest_cycles = []
    for cycle in nx.simple_cycles(G):
        n_cycles += 1
        l = len(cycle)
        if l > l_max:
            l_max = l
            longest_cycles = [cycle]
        elif l == l_max:
            longest_cycles.append(cycle)

    n_longest = len(longest_cycles)
    s = ' '.join(longest_cycles[0])
    print(f"{n_cycles} cycles, {n_longest} of max length={l_max}")

    expected_pairs = set(data['expected_pairs'].items())

    scored_cycles = []
    for cycle in longest_cycles:
        edges = list(itertools.pairwise(cycle + [cycle[0]]))
        is_good_path = all(e in expected_pairs for e in edges)
        error = sum(G.edges[e]['mse'] for e in edges)
        scored_cycles.append((is_good_path, error, cycle))

    for is_good_path, error, cycle in sorted(scored_cycles, key=operator.itemgetter(1)):
        x = '*' if is_good_path else ' '
        s = ' '.join(cycle)
        print(f'{x} {error=:8.3f} {s}')

    if False:
        src = 'A1'
        dst = 'B1'
        print(f"all_simple_paths: {src=} {dst=}")
        for path in sorted(list(nx.all_simple_edge_paths(G, src, dst)), key=len):

            path.append((path[-1][1], path[0][0]))

            cost = 0.
            nodes = []
            is_good_path = True
            for uv in path:
                cost += G.edges[uv]['mse']
                nodes.append(uv[0])
                if uv not in expected_pairs:
                    is_good_path = False

            path = ' '.join(nodes)
            is_good_path = '*' if is_good_path else ' '
            print(f"{is_good_path} len={len(nodes):3d} {cost=:8.3f} {path=}")

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
    for e in G.edges:
        src, dst = e
        attr = G.edges[e]
        props = []
        
        style = attr.get('style', 'solid')
        
        props.append(f'style={style}')
        
        if G.out_degree(src) > 1:
            rank = attr.get('rank',-1)
            mse = attr.get('mse',-1)
            props.append(f'label="{rank}: {mse:.3f}"')

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
