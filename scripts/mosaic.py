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

Feature = puzzler.raft.Feature

def cross2d(x, y):
    return x[...,0] * y[...,1] - x[...,1] * y[...,0]

class MosaicBuilder:

    def __init__(self, pieces, best_fits):
        self.pieces = pieces
        self.best_fits = best_fits

    def get_cell_sides_for_piece(self, piece):
        
        def get_tab_uv(p, i):
            t = p.tabs[i]
            v = p.points[np.array(t.tangent_indexes)] - t.ellipse.center
            v = v / np.linalg.norm(v, axis=1)
            v = np.sum(v, axis=0)
            v = v / np.linalg.norm(v)
            if not t.indent:
                v = -v
            return v
        
        p = self.pieces[piece]
        n = len(p.tabs)

        # shortcut for the common case, and it avoids tripping over
        # annoying pieces like P31 that might confuse us
        if n == 4:
            return (0, 3, 2, 1)
        
        uv = [get_tab_uv(p,i) for i in range(n)]

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
            assert sides[j] is None
            sides[j] = i

        return tuple(sides)

    def make_cell(self, piece, side_no, tab_no):

        sides = self.get_cell_sides_for_piece(piece)
        i = sides.index(tab_no)
        if i != side_no:
            rot = i - side_no
            sides = sides[rot:] + sides[:rot]
        assert sides[side_no] == tab_no
        return Cell(piece, sides)

    def get_matching_cell(self, cell, side_no):

        tab_no = cell.sides[side_no]
        if tab_no is None:
            return None

        m = self.best_fits.get(Feature(cell.piece, 'tab', tab_no))
        if m is None:
            return None

        return self.make_cell(m.piece, side_no^2, m.index)

    def breadth_first_search(self):

        # cells[x,y]
        cells = dict()
        cells[0,0] = Cell('A1', (0,None,None,1))

        q = collections.deque([(0,0)])
        while q:
            x, y = q.popleft()
            src = cells[x,y]
            print(f"processing {(x,y)}: {src}")
            for d, (i, j) in enumerate([(1, 0), (0, -1), (-1, 0), (0, 1)]):

                match = self.get_matching_cell(src, d)
                if match is None:
                    continue

                print(f"  side {d} {match=}")

                if dst := cells.get((x+i,y+j)):
                    if dst == match:
                        continue
                    
                    print(f"cell conflict at {(x+i,y+j)}, expected={match} actual={dst}")
                    break

                print(f"  placing ({x+i,y+j}): {match}")

                cells[x+i,y+j] = match
                q.append((x+i, y+j))

        return cells

def compute_best_fits(pieces, tab_pairs):

    best_fits = {}
    for p in pieces.values():
        for i in range(len(p.tabs)):
            dst = Feature(p.label, 'tab', i)
            src = tab_pairs.get_best_fit(dst)
            best_fits[dst] = src

    return best_fits

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
        if best_fits.get(src,src) != dst:
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

def read_best_fits(path):
    
    best_fits = {}
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dst = Feature(row['dst_piece'], 'tab', int(row['dst_tab_no']))
            src = Feature(row['src_piece'], 'tab', int(row['src_tab_no']))
            best_fits[dst] = src
            
    return best_fits

def write_best_fits(path, best_fits):

    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames='dst_piece dst_tab_no src_piece src_tab_no'.split())
        writer.writeheader()
        for dst, src in best_fits.items():
            writer.writerow({'dst_piece':dst.piece, 'dst_tab_no':dst.index,
                             'src_piece':src.piece, 'src_tab_no':src.index})

def write_dotty(path, cells):
    edges = set()
    for (x, y), cell in cells.items():
        for d, (i, j) in enumerate([(1, 0), (0, -1), (-1, 0), (0, 1)]):
            if neighbor := cells.get((x+i, y+j)):
                if cell.sides[d] is not None:
                    a, b = cell.piece, neighbor.piece
                    if a > b:
                        a, b = b, a
                    edges.add((a, b))
    with open(path, 'w') as f:
        print("graph G {", file=f)
        for (x, y), cell in cells.items():
            print(f"  {cell.piece} [pos=\"{x},{-y}!\"]", file=f)
        for a, b in sorted(edges):
            print(f"  {a} -- {b}", file=f)
        print("}", file=f)

def write_best_fits_dotty(path, best_fits, expected=None):

    G = nx.DiGraph()
    for k, v in best_fits.items():
        G.add_edge(str(k), str(v))

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

        print(f"// component[{i}]: {c}", file=f)
        for j in c:
            if expected and is_bad_boy(j):
                print(f"  \"{j}\" [fillcolor=pink,style=filled]", file=f)
                
            for k in G.neighbors(j):
                props = ''
                if expected:
                    if is_good_edge(j,k):
                        props = '[color=darkgreen]'
                    else:
                        props = '[style=dashed]'
                print(f"  \"{j}\" -> \"{k}\" {props}", file=f)

    print("}", file=f)
    f.close()
        
def main():

    parser = argparse.ArgumentParser("mosaic")
    parser.add_argument("-p", "--puzzle", required=True)
    parser.add_argument("-b", "--best", required=True)
    parser.add_argument("-e", "--expected")
    parser.add_argument("-o", "--output")

    args = parser.parse_args()

    puzzle = puzzler.file.load(args.puzzle)
    pieces = {i.label:i for i in puzzle.pieces}

    if False:
        tab_pairs = puzzler.tabpairs.load_tab_pairs(args.tabs)
        best_fits = compute_best_fits(pieces, tab_pairs)
        write_best_fits(args.output, best_fits)
        return 0
        
    best_fits = read_best_fits(args.best)

    expected = read_expected(args.expected) if args.expected else None

    best_fits = filter_best_fits(best_fits, expected)

    mosaic = MosaicBuilder(pieces, best_fits)
    cells = mosaic.breadth_first_search()

    print(f"placed {len(cells)} pieces!")

    if args.output:
        write_dotty(args.output, cells)

if __name__ == '__main__':
    main()
