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

Feature = puzzler.raft.Feature

class GraphBestiary:

    def __init__(self):
        self.zoo = collections.defaultdict(list)

    def is_known_graph(self, G2):

        edge_match = node_match = lambda a,b: a['is_good'] == b['is_good']
        
        for G1 in self.zoo[len(G2)]:
            GM = isomorphism.DiGraphMatcher(G1, G2, node_match=node_match, edge_match=edge_match)
            if GM.is_isomorphic():
                return True
        return False

    def add_unknown_graph(self, G2):

        self.zoo[len(G2)].append(G2)

def make_dotty_graph(path, G):

    def output_component(component):
        print("", file=f)
        for i in component:
            if not G.nodes[i]['is_good']:
                print(f"  \"{i}\" [fillcolor=pink,style=filled]", file=f)
                
            for j in G.succ[i]:
                props = ''
                if G[i][j]['is_good']:
                    props = '[color=darkgreen]'
                print(f"  \"{i}\" -> \"{j}\" {props}", file=f)

    with open(path, 'w') as f:
        print("digraph G {", file=f)
        for c in sorted(nx.weakly_connected_components(G), key=len, reverse=True):
            output_component(c)
        print("}", file=f)

def make_best_fits_from_tab_pairs(tab_pairs):
    best_fits = {}
    for dst in tab_pairs.id_to_tab:
        src = tab_pairs.get_ranked_fit(dst, 1)
        best_fits[dst] = src
    return best_fits

def make_graph_from_best_fits(best_fits, expected = None):
        
    if expected:
        expected = {str(k):str(v) for k, v in expected.items()}

    G = nx.DiGraph()

    # each node represents a single tab and has a *single* outbound
    # edge to the tab it would most like to be matched with. A node *may*
    # have multiple inbound edges.
    for a, b in best_fits.items():
        is_good = expected and expected.get(str(a),'') == str(b)
        G.add_edge(str(a), str(b), is_good=is_good)

    for n in G.nodes:
        is_good = False
        if expected:
            e = expected.get(n,'')
            is_good = e in G.pred[n] or e in G.succ[n]
        G.add_node(n, is_good=is_good)

    return G

def filter_to(G, filters):

    def keep_node(node):
        return any(node.startswith(f) for f in filters)

    def keep_component(component):
        return any(keep_node(i) for i in component)

    nodes_to_remove = set()
    
    for c in nx.weakly_connected_components(G):

        if not keep_component(c):
            nodes_to_remove.update(c)

    G.remove_nodes_from(nodes_to_remove)

    return G
        
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
            
    return expected

def show_graph(G):

    dot = tempfile.NamedTemporaryFile(prefix='mosaic_', suffix='.dot', delete=False)
    dot.close()

    make_dotty_graph(dot.name, G)

    png = tempfile.NamedTemporaryFile(prefix='mosaic_', suffix='.png', delete=False)
    png.close()

    args = [r'C:\Program Files\Graphviz\bin\dot.exe', '-Tpng', '-o', png.name, dot.name]
    print(' '.join(args))

    subprocess.run(args)
    os.unlink(dot.name)
    
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
    
    args = parser.parse_args()

    tab_pairs = puzzler.tabpairs.load_tab_pairs(args.tab_pairs)
    expected = read_expected(args.expected) if args.expected else None

    best_fits = make_best_fits_from_tab_pairs(tab_pairs)
    G = make_graph_from_best_fits(best_fits, expected)

    if args.filter:
        G = filter_to(G, args.filter.split(','))

    if args.page_size:
        page_graph(args.output, G, args.page_size)
    elif args.output:
        make_dotty_graph(args.output, G)
    elif args.window:
        show_graph(G)

if __name__ == '__main__':
    main()
