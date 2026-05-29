import os, sys

# blech, fix up the path to find the project-specific modules
lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")
sys.path.insert(0, lib)

import argparse
import collections
import csv
import networkx as nx
#from networkx.algorithms import bipartite
import puzzler

def load_quads(input_csv_path):

    quads = []
    with open(input_csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k in 'col_no', 'row_no', 'rank':
                row[k] = int(row[k])
            for k in 'mse',:
                row[k] = float(row[k])
            quads.append(row)

    return quads

def write_tabs(path, tabs):

    def parse_tab(s):
        a, b = s.split(':')
        return a, int(b)

    fieldnames = 'dst_piece dst_tab_no src_piece src_tab_no'.split()
    
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames)
        writer.writeheader()
        for a, b in sorted(tabs.items()):
            dst_piece, dst_tab_no = parse_tab(a)
            src_piece, src_tab_no = parse_tab(b)
            writer.writerow({'dst_piece':dst_piece,
                             'dst_tab_no':dst_tab_no,
                             'src_piece':src_piece,
                             'src_tab_no':src_tab_no})

def match_it(puzzle, quads):

    def parse_raft(s):
        return [tuple(f.split('=')) for f in s.split(',')]

    def parse_tab(s):
        a, b = s.split(':')
        return a, int(b)

    all_tabs = set()
    for quad in quads:
        
        for a, b in parse_raft(quad['raft']):
            all_tabs.add(a)
            all_tabs.add(b)

    matching_tabs = collections.defaultdict(set)
    for quad in quads:
        
        if quad['rank'] != 1: # or quad['mse'] > 10.:
            continue

        for a, b in parse_raft(quad['raft']):

            matching_tabs[a].add(b)
            matching_tabs[b].add(a)

    def get_source(dst):
        srcs = matching_tabs[dst]
        if len(srcs) != 1:
            return ''
        return next(iter(srcs))

    retval = dict()
    unknown = set()
    for dst in matching_tabs.keys():
        src = get_source(dst)
        if src and dst == get_source(src):
            retval[dst] = src
        else:
            unknown.add(dst)

    unknown |= all_tabs - set(retval.keys())

    for dst in sorted(unknown):
        srcs = ','.join(sorted(matching_tabs[dst]))
        print(f"{dst}: ({srcs})")

    return retval

def match_nx(puzzle, quads):

    def parse_raft(s):
        return [tuple(f.split('=')) for f in s.split(',')]

    def parse_tab(s):
        a, b = s.split(':')
        return a, int(b)

    G = nx.Graph()
    
    for piece in puzzle.pieces:
        for tab_no, tab in enumerate(piece.tabs):
            node = f"{piece.label}:{tab_no}"
            G.add_node(node, bipartite=int(tab.indent))

    for quad in quads:
        
        if quad['rank'] != 1:
            continue

        for dst, src in parse_raft(quad['raft']):
            G.add_edge(src, dst)

    # remove orphan nodes (tabs that were never matched)
    nodes_to_remove = set()
    for n in G.nodes:
        if len(G.edges(n)) == 0:
            nodes_to_remove.add(n)

    G.remove_nodes_from(nodes_to_remove)
    
    top_nodes = {n for n, d in G.nodes(data=True) if d["bipartite"] == 0}

    m = nx.bipartite.maximum_matching(G, top_nodes)
    unmatched = G.nodes - set(m.keys())
    print("unmatched:", ','.join(sorted(unmatched)))

    def is_special(n):
        return len(G.edges(n)) == 1

    nodes_to_remove = set()
    for a, b in G.edges:
        if is_special(a) and is_special(b):
            nodes_to_remove.add(a)
            nodes_to_remove.add(b)
    print(f"before: {len(nodes_to_remove)=} {len(G.nodes)=} {len(G.edges)=}")
    G.remove_nodes_from(nodes_to_remove)
    print(f"after: {len(G.nodes)=} {len(G.edges)=}")

    with open('dotty.dot', 'w',) as f:

        print("graph G {", file=f)
        for n in sorted(G.nodes):
            color = 'red' if n in top_nodes else 'green'
            print(f"  \"{n}\" [color={color}]", file=f)
            
        for a, b in G.edges:
            print(f"  \"{a}\" -- \"{b}\"", file=f)
        print("}", file=f)

    return m

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--puzzle', required=True,
                        help='input puzzle')
    parser.add_argument('-q', '--quads', required=True,
                        help='input CSV of quads data')
    parser.add_argument('-o', '--output', required=True,
                        help='output CSV of tab correspondence')

    args = parser.parse_args()

    puzzle = puzzler.file.load(args.puzzle)
    quads = load_quads(args.quads)
    tabs = match_nx(puzzle, quads)
    write_tabs(args.output, tabs)

if __name__ == '__main__':
    main()
