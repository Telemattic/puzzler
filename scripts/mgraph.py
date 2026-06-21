import os, sys

# blech, fix up the path to find the project-specific modules
lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")
sys.path.insert(0, lib)

import argparse
import collections
import csv
import networkx as nx
from networkx.algorithms import isomorphism
import puzzler
import subprocess
import tempfile
import time
from msat import *

Feature = puzzler.raft.Feature

def make_dotty_graph(path, G):

    def format_prop(k, v):
        return f"{k}={v}"

    def format_props(props):
        if not props:
            return ''
        
        return '[' + ', '.join(format_prop(k,v) for k,v in props.items()) + ']'

    def output_component(component):
        print("", file=f)
        for i in component:
            node_props = {}
            if G.nodes[i].get('is_marooned'):
                node_props['fillcolor'] = 'pink'
                node_props['style'] = 'filled'
            if G.nodes[i].get('is_filtered'):
                node_props['peripheries'] = 2

            if node_props:
                print(f"  \"{i}\" {format_props(node_props)}", file=f)
                
            for j in G.succ[i]:
                edge_props = {}
                if G[i][j].get('is_good'):
                    edge_props['color'] = 'darkgreen'
                    edge_props['penwidth'] = '2.0'
                if G[i][j].get('is_fake'):
                    edge_props['style'] = 'dashed'
                if G[i][j].get('z3'):
                    edge_props['color'] = 'darkblue' if G[i][j].get('valid',True) else 'darkred'
                    edge_props['penwidth'] = '3.0'
                    
                print(f"  \"{i}\" -> \"{j}\" {format_props(edge_props)}", file=f)

    if path is None:
        with tempfile.NamedTemporaryFile(prefix='mosaic_', suffix='.dot') as f:
            path = f.name

    with open(path, 'w') as f:
        print("digraph G {", file=f)
        print("  node [fontname=\"Courier New\"]", file=f)
        for c in sorted(nx.weakly_connected_components(G), key=len, reverse=True):
            output_component(c)
        print("}", file=f)

    return path

def make_best_fits_from_tab_pairs(tab_pairs):
    best_fits = {}
    for dst in tab_pairs.id_to_tab:
        src = tab_pairs.get_ranked_fit(dst, 1)
        best_fits[dst] = src
    return best_fits

def make_graph_from_best_fits(best_fits, expected = None):
        
    G = nx.DiGraph()

    # each node represents a single tab and has a *single* outbound
    # edge to the tab it would most like to be matched with. A node *may*
    # have multiple inbound edges.
    for a, b in best_fits.items():
        if expected and expected.get(str(a),'') == str(b):
            G.add_edge(str(a), str(b), is_good=True)
        else:
            G.add_edge(str(a), str(b))

    if expected:
        for n in G.nodes:
            e = expected.get(n,'')
            if not (e in G.pred[n] or e in G.succ[n]):
                G.add_node(n, is_marooned=True)

    if expected and False:
        for a, b in expected.items():
            if a < b and not G.has_edge(a, b) and not G.has_edge(b, a):
                G.add_edge(a, b, is_good=True, is_fake=True)

    return G

def filter_to(G, filters):

    def keep_node(node):
        return any(node.startswith(f) for f in filters)

    def keep_component(component):
        return any(keep_node(i) for i in component)

    nodes_to_keep = set()
    
    for component in nx.weakly_connected_components(G):

        if keep_component(component):
            nodes_to_keep.update(component)
            for n in component:
                if keep_node(n):
                    G.add_node(n, is_filtered=True)

    return G.subgraph(nodes_to_keep)
        
def just_unique_examples(G):

    zoo = collections.defaultdict(list)

    def is_isomorphic(G1, G2):
        GM = isomorphism.DiGraphMatcher(
            G1, G2, node_match=operator.eq, edge_match=operator.eq)
        return GM.is_isomorphic()

    def is_known_graph(self, G2):
        return any(is_isomoprhic(G1, G2) for G1 in self.zoo[len(G2)])

    def add_unknown_graph(self, G2):
        self.zoo[len(G2)].append(G2)

    nodes_to_remove = set()
    
    for c in nx.weakly_connected_components(G):

        if is_known_graph(G.subgraph(c)):
            nodes_to_remove.update(c)
        
        add_unknown_graph(G.subgraph(c))

    G.remove_nodes_from(nodes_to_remove)
    return G

def page_graph(path, G, max_nodes_per_graph):

    graph_num = 0
    nodes = set()

    def output_subgraph():
        nonlocal graph_num, nodes
        opath = f"{path}_{graph_num}.dot"
        print(opath)
        make_dotty_graph(opath, G.subgraph(nodes))
        graph_num += 1
        nodes = set()

    for c in sorted(nx.weakly_connected_components(G), key=len, reverse=True):

        if nodes and len(nodes)+len(c) > max_nodes_per_graph:
            output_subgraph()

        nodes.update(c)

    if nodes:
        output_subgraph()

def read_expected(path):

    expected = {}
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dst = Feature(row['dst_piece'], 'tab', int(row['dst_tab_no']))
            src = Feature(row['src_piece'], 'tab', int(row['src_tab_no']))
            expected[dst] = src

    return {str(k):str(v) for k, v in expected.items()}

def output_graph(output_path, G):

    if output_path.endswith('.dot'):
        
        make_dotty_graph(output_path, G)
        
    elif output_path.endswith('.png'):
        
        dot_path = make_dotty_graph(None, G)
        run_dotty(['-Tpng', '-o', output_path, dot_path])
        os.unlink(dot_path)
        
    elif output_path.endswith('.pdf'):

        dot_path = make_dotty_graph(None, G)
        run_dotty(['-Tpdf', '-o', output_path, dot_path])
        os.unlink(dot_path)
        
    else:
        raise ValueError(f"don't know how to generate {output_path}")

def run_dotty(args):
    
    exe = r'C:\Program Files\Graphviz\bin\dot.exe'
    subprocess.run([exe] + args)
    
def show_graph(G):

    dot_path = make_dotty_graph(None, G)

    png = tempfile.NamedTemporaryFile(prefix='mosaic_', suffix='.png', delete=False)
    png.close()

    run_dotty(['-Tpng', '-o', png.name, dot_path])

    os.unlink(dot_path)
    
    os.startfile(png.name)

    time.sleep(1)
    os.unlink(png.name)

def main():

    parser = argparse.ArgumentParser("mosaic")
    parser.add_argument("-t", "--tab-pairs", required=True)
    parser.add_argument("-e", "--expected")
    parser.add_argument("-o", "--output")
    parser.add_argument("-f", "--filter")
    parser.add_argument("-p", "--page-size", type=int)
    parser.add_argument("-w", "--window", action='store_true')
    parser.add_argument("--z3")
    
    args = parser.parse_args()

    tab_pairs = puzzler.tabpairs.load_tab_pairs(args.tab_pairs)
    expected = read_expected(args.expected) if args.expected else None

    best_fits = make_best_fits_from_tab_pairs(tab_pairs)
    G = make_graph_from_best_fits(best_fits, expected)

    if args.z3:
        edges_to_add = []
        with open(args.z3, 'r') as f:
            for row in f:
                a, b = row.split()
                if G.has_edge(a, b):
                    edges_to_add.append((a,b))
                    if G.has_edge(b, a):
                        edges_to_add.append((b,a))
                else:
                    edges_to_add.append((b,a))

        for a, b in edges_to_add:
            if expected:
                G.add_edge(a, b, z3=True, valid=expected.get(a,'') == b)
            else:
                G.add_edge(a, b, z3=True)
                    
    if args.filter:
        G = filter_to(G, args.filter.split(','))

    if args.page_size:
        page_graph(args.output, G, args.page_size)
    elif args.output:
        output_graph(args.output, G)
    elif args.window:
        show_graph(G)

if __name__ == '__main__':
    main()
