import os, sys

# blech, fix up the path to find the project-specific modules
lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")
sys.path.insert(0, lib)

import argparse
import collections
import csv
import networkx as nx
import operator
import os
import puzzler
import subprocess
import tempfile

from tqdm import tqdm

def read_scores(path):

    retval = collections.defaultdict(dict)
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            lhs = row['lhs']
            rhs = row['rhs']
            rank = int(row['rank'])
            score = float(row['score'])
            retval[lhs][rhs] = (rank, score)

    return retval

def compute_renames(scores, dotfile):

    from networkx.algorithms import bipartite

    reversed_scores = collections.defaultdict(list)
    edges = []
    for lhs in scores:
        for rhs, (rank, score) in scores[lhs].items():
            if rank == 1:
                edges.append((lhs, rhs, score))
            reversed_scores[rhs].append((score, lhs))

    for rhs in reversed_scores:
        score, lhs = sorted(reversed_scores[rhs])[0]
        edges.append((lhs, rhs, score))

    G = nx.Graph()

    lhs_nodes = set()
    rhs_nodes = set()
    for lhs, rhs, score in edges:
        lhs_nodes.add('l'+lhs)
        rhs_nodes.add('r'+rhs)
        G.add_edge('l'+lhs, 'r'+rhs, weight=score)

    print(f"{len(lhs_nodes)=} {len(rhs_nodes)=}")

    mapping = dict()

    to_remove = set()
    for u, v in G.edges:
        if G.degree(u) == 1 and G.degree(v) == 1:
            to_remove.add(u)
            to_remove.add(v)
            if u in lhs_nodes:
                mapping[u] = v
            else:
                mapping[v] = u

    G.remove_nodes_from(to_remove)
    lhs_nodes -= to_remove
    rhs_nodes -= to_remove
    
    print(f"trivial_mapping: {mapping}")
    
    m = bipartite.minimum_weight_full_matching(G, top_nodes=lhs_nodes)
    print(f"maximum_matching: {len(m)=} {m}")

    for u, v in m.items():
        if u in lhs_nodes:
            mapping[u] = v
        else:
            mapping[v] = u

    mapping = dict((k[1:], v[1:]) for k, v in mapping.items())

    print(f"final mapping: {mapping}")

    if dotfile:
        print("graph G {", file=dotfile)
        for n in G.nodes:
            shape = 'square' if n[0] == 'l' else 'circle'
            label = n[1:]
            print(f"  {n} [shape={shape} label=\"{label}\"]", file=dotfile)

        for u, v in G.edges:
            print(f"  {u} -- {v}", file=dotfile)

        print("}", file=dotfile)

    return mapping

def to_graph(scores, dotfile):

    G = nx.Graph()
    
    for lhs in scores:
        
        G.add_node('l'+lhs)
        for rhs in scores[lhs]:
            G.add_node('r'+rhs)
        
        for rhs, (rank, score) in scores[lhs].items():
            if rank == 1:
                G.add_edge('l'+lhs, 'r'+rhs)

    scores2 = collections.defaultdict(list)
    for lhs in scores:
        for rhs, (rank, score) in scores[lhs].items():
            scores2[rhs].append((score, lhs))

    for rhs in scores2:
        lhs = sorted(scores2[rhs])[0][1]
        G.add_edge('l'+lhs, 'r'+rhs)

    to_remove = set()
    for u, v in G.edges:
        if G.degree(u) == 1 and G.degree(v) == 1:
            to_remove.add(u)
            to_remove.add(v)

    G.remove_nodes_from(to_remove)

    shape_counts = collections.defaultdict(int)
    label_counts = collections.defaultdict(int)

    print("graph G {", file=dotfile)
    for n in G.nodes:
        assert n[0] in 'lr'
        shape = 'square' if n[0] == 'l' else 'circle'
        label = n[1:]
        print(f"  {n} [shape={shape} label=\"{label}\"]", file=dotfile)
        shape_counts[shape] += 1
        label_counts[label] += 1

    for u, v in G.edges:
        lhs, rhs = u[1:], v[1:]
        if u[0] == 'r':
            lhs, rhs = rhs, lhs
        print(f"  {u} -- {v}", file=dotfile)
    print("}", file=dotfile)

    print("shape_counts:", shape_counts)

    label_counts_not_2 = dict((k, v) for k, v in label_counts.items() if v != 2)
    print("label_counts:", label_counts_not_2)
        
def make_pdf(dotpath, pdfpath):
    # "C:\Program Files\Graphviz\bin\dot.exe" -Tpdf -Gsize=8,10.5 -Gpage=8.5,11 -o <pdf_path> <dot_path>
    exe = r'C:\Program Files\Graphviz\bin\dot.exe'
    args = [exe, '-Tpdf', '-Gsize=8,10.5', '-Gpage=8.5,11', '-o', pdfpath, dotpath]
    print(' '.join(args))
    subprocess.run(args)

def main():
    parser = argparse.ArgumentParser(prog='match_puzzles')
    parser.add_argument("-s", "--scores", help="scores", required=True)
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    scores = read_scores(args.scores)

    if False:
        with tempfile.NamedTemporaryFile(dir=r'C:\Temp', prefix='puzzler_', suffix='.dot',
                                         mode='w', newline='\n', delete=False) as dotfile:
            compute_renames(scores, dotfile)
            dotfile.close()
            make_pdf(dotfile.name, 'fnord.pdf')
            os.unlink(dotfile.name)
        return None
    
    renames = compute_renames(scores, None)

    puzzle = puzzler.file.load(args.input)
    
    for i in puzzle.pieces:
        i.label = renames[i.label]

    puzzler.file.save(args.output, puzzle)

if __name__ == '__main__':
    main()
