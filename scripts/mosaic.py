import os, sys

# blech, fix up the path to find the project-specific modules
lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")
sys.path.insert(0, lib)

import argparse
import collections
import csv
import itertools
import math
import networkx as nx
import numpy as np
import puzzler

from typing import NamedTuple, Tuple

#
#  +-- 1 --+
#  |       |
#  2       0
#  |       |
#  +-- 3 --+
#

class Cell(NamedTuple):
    piece: str
    sides: Tuple[int]

    def __eq__(self, other):
        return self.piece == other.piece and all(a is None and b is None or a == b for a, b in zip(self.sides, other.sides))

Feature = puzzler.raft.Feature

def cross2d(x, y):
    return x[...,0] * y[...,1] - x[...,1] * y[...,0]

class MosaicException(Exception):
    pass


def make_cell_for_piece(p):
        
    def get_tab_uv(i):
        t = p.tabs[i]
        v = p.points[np.array(t.tangent_indexes)] - t.ellipse.center
        v = v / np.linalg.norm(v, axis=1)
        v = np.sum(v, axis=0)
        v = v / np.linalg.norm(v)
        if not t.indent:
            v = -v
        return v

    def get_sides_for_piece():

        n = len(p.tabs)

        # shortcut for the common case, and it avoids tripping over
        # annoying pieces like P31 that might confuse us
        if n == 4:
            return (0, 3, 2, 1)

        uv = [get_tab_uv(i) for i in range(n)]

        sides = [None] * 4

        # by arbitrary choice
        sides[0] = 0

        for i in range(1,n):
            angle = math.degrees(np.atan2(cross2d(uv[i],uv[0]), np.sum(uv[i] * uv[0])))
            #print(f"{piece=} tab[{i}] at {angle=:.0f} degrees to tab[0]")
            if 45 <= angle < 135:
                j = 3
            elif -45 <= angle < 45:
                j = 0
            elif -135 <= angle < -45:
                j = 1
            else:
                j = 2
                if sides[j] is not None:
                    raise MosaicException(f"get_sides_for_piece: barfed on {p.label} with {n} sides")
            sides[j] = i

        return tuple(sides)

    return Cell(p.label, get_sides_for_piece())

def rotate_cell(cell, side_no, tab_no):
    
    i = cell.sides.index(tab_no)
    if i == side_no:
        return cell
    
    rot = i - side_no
    sides = cell.sides[rot:] + cell.sides[:rot]
    assert sides[side_no] == tab_no

    return Cell(cell.piece, sides)

def get_weakly_connected_nodes(G, source_nodes):
    visited = set()
    next_level = set(source_nodes)
    
    # Traverse graph structure ignoring edge directions
    while next_level:
        this_level = next_level
        next_level = set()
        for v in this_level:
            if v not in visited:
                visited.add(v)
                # Add successors (outgoing) and predecessors (incoming)
                next_level.update(G.succ[v])
                next_level.update(G.pred[v])
    return visited

def str_to_tab(s):
    p, i = s.split(':')
    return Feature(p, 'tab', int(i))

class MosaicGraph:

    def __init__(self, tab_pairs):
        self.graph = nx.DiGraph()
        self.tab_pairs = tab_pairs

        for dst in tab_pairs.id_to_tab:
            src = tab_pairs.get_ranked_fit(dst, 1)
            self.graph.add_edge(str(src), str(dst), rank=1)

    def resolve_match(self, tabA, tabB):

        nodeA = str(tabA)
        nodeB = str(tabB)
        g = self.graph

        remove_edges = []
        add_edges = []

        for pred, rank in g.pred[nodeB].data('rank'):
            if pred != nodeA:
                remove_edges.append((pred, nodeB, rank))

        if not g.has_edge(nodeA, nodeB):
            add_edges.append((nodeA, nodeB, None))

        for pred, rank in g.pred[nodeA].data('rank'):
            if pred != nodeB:
                remove_edges.append((pred, nodeA, rank))
                
        if not g.has_edge(nodeB, nodeA):
            add_edges.append((nodeB, nodeA, None))

        for prev, succ, rank in remove_edges:
            repl = self.tab_pairs.get_ranked_fit(str_to_tab(prev), rank+1)
            add_edges.append((prev, str(repl), rank+1))

        for u, v, _ in remove_edges:
            self.graph.remove_edge(u, v)
            
        for u, v, rank in add_edges:
            self.graph.add_edge(u, v, rank=rank)

        return remove_edges, add_edges

    def graph_it(dotty_path, removed_edges, added_edges):

        affected_nodes = set()
        for u, v, _ in removed_edges + added_edges:
            affected_nodes.add(u)
            affected_nodes.add(v)

        nodes_to_graph = get_weakly_connected_nodes(self.graph, affected_nodes)

        with open(path, 'w') as f:
            for u, v, rank in remove_edges:
                print(f"// remove: {(u,v)} {rank=}", file=f)
            for u, v, rank in add_edges:
                print(f"// add: {(u,v)} {rank=}", file=f)
            print("digraph G {", file=f)

            for pred in nodes_to_graph:
                for succ, rank in G.succ[pred]:
                    if rank is not None:
                        props = f"[label=\"{rank}\"]"
                    print(f"  \"{pred}\" --> \"{succ}\" {props}", file=f)

            print("}", file=f)

class MosaicBuilder:

    def __init__(self, pieces, graph):
        self.pieces = pieces
        self.graph = graph

    def get_best_match_for_tab(self, tab):
        
        G = self.graph.graph
        
        p = list(G.predecessors(str(tab)))
        if len(p) != 1:
            return None
        m = p[0]

        p = list(G.predecessors(m))
        if len(p) != 1 or p[0] != str(tab):
            return None

        return str_to_tab(m)

    def get_best_match_for_neighbors(self, neighbors):

        matches = []
        for i, n in enumerate(neighbors):
            if n:
                if m := self.get_best_match_for_tab(n):
                    matches.append((i, m))
                else:
                    return None

        if not matches:
            return None

        matches2 = []
        for side_no, m in matches:
            cell = make_cell_for_piece(self.pieces[m.piece])
            matches2.append(rotate_cell(cell, side_no, m.index))

        if len(matches2) == 1:
            return matches2[0]

        if all(matches2[0] == m for m in matches2[1:]):
            return matches2[0]

        print("*** an abundance of choices!")
        for m in matches2:
            print(f"--> {m}")
            
        return None

    def breadth_first_search(self):

        directions = ((1, 0), (0, -1), (-1, 0), (0, 1))

        def get_cell_feature(x, y, side_no):
            c = cells.get((x,y))
            if c is None:
                return None
            index = c.sides[side_no]
            if index is None:
                return None
            return Feature(c.piece, 'tab', index)

        def get_neighbors(x, y):
            return tuple(get_cell_feature(x+i, y+j, d^2)
                         for d, (i, j) in enumerate(directions))
                
        # cells[x,y]
        cells = dict()
        placed = dict()
        
        cells[0,0] = make_cell_for_piece(self.pieces['A1'])
        placed['A1'] = (0,0)
        q = collections.deque([(0,1), (1,0)])

        try:
            while q:
                x, y = q.popleft()
                if (x,y) in cells:
                    continue
                
                print(f"processing {(x,y)}:")
                
                neighbors = get_neighbors(x, y)
                print(f"  neighbors={tuple(str(i) if i else i for i in neighbors)}")

                match = self.get_best_match_for_neighbors(neighbors)
                if match is None:
                    continue
                
                if match.piece in placed:
                    raise MosaicException(f"cell conflict at {(x,y)}, expected{match} but {match.piece} has already been placed at {placed[match.piece]}")

                print(f"  placing ({x,y}): {match}")
                placed[match.piece] = (x,y)

                cells[x,y] = match
                for i, j in directions:
                    q.append((x+i, y+j))
                
        except MosaicException as x:
            print(x)

        return cells

def filter_best_fits(best_fits, expected=None):

    doubles = {'B18', 'B27', 'E26', 'G6', 'J17', 'R5', 'R14', 'X21'}

    retval = {}
    for dst, src in best_fits.items():

        # we don't support double pieces yet
        if dst.piece in doubles or src.piece in doubles:
            continue

        # if the expected matches are provided then *only* consider
        # the known good matches
        if expected and expected.get(dst,dst) != src:
            continue

        # require the matches to be symmetric (they might be a bad
        # match, but at least they love each other)
        if best_fits.get(src,src) != dst and False:
            continue
        
        retval[dst] = src

    return retval

def read_expected(path):

    expected = {}
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dst = Feature(row['dst_piece'], 'tab', int(row['dst_tab_no']))
            src = Feature(row['src_piece'], 'tab', int(row['src_tab_no']))
            expected[dst] = src
            
    return expected

def write_dotty(path, cells, expected = None):
    
    edges = set()
    for (x, y), cell in cells.items():
        for d, (i, j) in enumerate([(1, 0), (0, -1), (-1, 0), (0, 1)]):
            if neighbor := cells.get((x+i, y+j)):
                if cell.sides[d] is not None and neighbor.sides[d^2] is not None:
                    a = Feature(cell.piece, 'tab', cell.sides[d])
                    b = Feature(neighbor.piece, 'tab', neighbor.sides[d^2])
                    if a > b:
                        a, b = b, a
                    edges.add((a, b))
                    
    with open(path, 'w') as f:
        print("graph G {", file=f)
        print("  edge [fontname=\"Arial\", fontsize=8];", file=f)
        for (x, y), cell in cells.items():
            print(f"  {cell.piece} [pos=\"{x*1.5},{-y}!\"]", file=f)
        for a, b in sorted(edges):
            if expected:
                color = 'darkgreen' if expected.get(a) == b else 'red'
            else:
                color = 'black'
            print(f"  {a.piece} -- {b.piece} [color={color} taillabel=\"{str(a)}\" headlabel=\"{str(b)}\"]", file=f)
        print("}", file=f)

def graph_best_fits(path, tab_pairs, expected=None):

    best_fits = {}
    for dst in tab_pairs.id_to_tab:
        src = tab_pairs.get_ranked_fit(dst, 1)
        best_fits[dst] = src

    G = nx.DiGraph()

    # each node represents a single tab and has a *single* outbound
    # edge to the tab it would most like to be matched with. A node *may*
    # have multiple inbound edges.
    for a, b in best_fits.items():
        G.add_edge(str(a), str(b))

    max_nodes_per_graph = 500
    num_nodes_in_graph = 0
    graph_num = 0

    if expected:
        expected = {str(k):str(v) for k, v in expected.items()}

    def is_good_edge(a, b):
        return expected.get(a,'') == b

    def in_and_out_edges(n):
        return itertools.chain(G.in_edges(n), G.out_edges(n))
    
    def is_bad_boy(n):
        return not any(is_good_edge(a,b) for a, b in in_and_out_edges(n))
    
    for i, c in enumerate(sorted(nx.weakly_connected_components(G), key=len, reverse=True)):

        if not any((i.startswith('S6:') or i.startswith('G16:')) for i in c):
            continue

        if 0 < num_nodes_in_graph and num_nodes_in_graph+len(c) > max_nodes_per_graph:
            print("}", file=f)
            f.close()
            num_nodes_in_graph = 0
                
        if num_nodes_in_graph == 0:
            opath = f"{path}_{graph_num}.dot"
            print(opath)
            f = open(opath, 'w')
            graph_num += 1
            print("digraph G {", file=f)

        num_nodes_in_graph += len(c)

        print(f"  // component[{i}]: {c}", file=f)
        for j in c:
            if expected and is_bad_boy(j):
                print(f"  \"{j}\" [fillcolor=pink,style=filled]", file=f)
                
            for k in G.neighbors(j):
                props = ''
                if expected:
                    if is_good_edge(j,k):
                        props = '[color=darkgreen]'
                    else:
                        e = expected.get(j)
                        if e and e in c and not G.has_edge(e,j):
                            print(f"  \"{j}\" -> \"{e}\" [style=dashed color=darkgreen]", file=f)
                print(f"  \"{j}\" -> \"{k}\" {props}", file=f)

    print("}", file=f)
    f.close()
        
def main():

    parser = argparse.ArgumentParser("mosaic")
    parser.add_argument("-p", "--puzzle", required=True)
    parser.add_argument("-t", "--tab-pairs", required=True)
    parser.add_argument("-e", "--expected")
    parser.add_argument("-o", "--output")
    parser.add_argument("command", choices=['solve', 'graph-best-fits'], default='solve')

    args = parser.parse_args()

    puzzle = puzzler.file.load(args.puzzle)
    pieces = {i.label:i for i in puzzle.pieces}

    tab_pairs = puzzler.tabpairs.load_tab_pairs(args.tab_pairs)

    expected = read_expected(args.expected) if args.expected else None

    if args.command == 'graph-best-fits':
        graph_best_fits(args.output, tab_pairs, expected)
        return 0

    graph = MosaicGraph(tab_pairs)
    mosaic = MosaicBuilder(pieces, graph)
    cells = mosaic.breadth_first_search()

    print(f"placed {len(cells)} pieces!")

    if args.output:
        write_dotty(args.output, cells, expected)

if __name__ == '__main__':
    main()
