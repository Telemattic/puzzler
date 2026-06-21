import os, sys

# blech, fix up the path to find the project-specific modules
lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")
sys.path.insert(0, lib)

import argparse
import collections
import csv
import functools
import itertools
import math
import networkx as nx
import numpy as np
import puzzler
import tqdm
from msat import *
from rocksdict import Rdict

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
    
    def get_tab_for_side(self, i):
        index = self.sides[i]
        if index is None:
            raise ValueError(f"no tab for piece {self.piece} on side {i} of cell")
        return Feature(self.piece, 'tab', index)

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

    if len(p.tabs) > 4:
        raise MosaicException(f"make_cell_for_piece: too many tabs on {p.label}!")

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

class RaftTester:

    def __init__(self, pieces, db_path=None):
        self.pieces = pieces
        self.raftinator = puzzler.raft.Raftinator(pieces)
        self.db = Rdict(db_path) if db_path else None

    def get_raft_error(self, feature_pairs):
        
        r = self.raftinator
        desc = r.format_feature_pairs(feature_pairs)
        
        mse = self.db.get(desc)
        if mse is not None:
            return mse

        raft = r.make_raft_from_feature_pairs(feature_pairs)

        # repeated refinement
        for _ in range(3):
            raft = r.refine_alignment_within_raft(raft)
            
        mse = r.get_total_error_for_raft_and_seams(raft)

        # print(f"{mse:8.1f} <-- {desc}")

        self.db[desc] = mse

        return mse

class MosaicBuilder:

    def __init__(self, pieces, graph, db_path=None):
        self.pieces = pieces
        self.graph = graph
        self.raft_tester = RaftTester(self.pieces, db_path)

    def get_matches_for_tab(self, tab):
        G = self.graph.graph
        s = set(G.predecessors(str(tab))) | set(G.successors(str(tab)))
        return list(str_to_tab(i) for i in s)

    @functools.cache
    def get_cell_for_tab(self, tab, side_no):
        cell = make_cell_for_piece(self.pieces[tab.piece])
        return rotate_cell(cell, side_no, tab.index)

    def is_good_raft(self, feature_pairs):
        return self.raft_tester.get_raft_error(feature_pairs) < 20.

    def test_tab_pair(self, tabA, tabB):

        def check_neighbors(side_no):

            # a non-empty list containing None is a sentinel for an edge match
            if cellA.sides[side_no] is None and cellB.sides[side_no] is None:
                return [None]

            # if one is an edge and the other isn't then something is
            # busted, the edge can't just disappear
            if cellA.sides[side_no] is None or cellB.sides[side_no] is None:
                return []

            tabAC = cellA.get_tab_for_side(side_no)
            tabBD = cellB.get_tab_for_side(side_no)

            good_matches = []
            for tabCA, tabDB in itertools.product(
                    self.get_matches_for_tab(tabAC),
                    self.get_matches_for_tab(tabBD)):
                fps = [(tabA, tabB), (tabAC, tabCA), (tabBD, tabDB)]
                if self.is_good_raft(fps):
                    good_matches.append(fps)
            return good_matches

        #    +---+
        #  D | B | D'
        # -- +---+---
        #  C | A | C'
        #    +---+

        print(f"test_tab_pair: {tabA=!s} {tabB=!s}")
        
        # arbitrarily make tabA face up (side 1)
        cellA = self.get_cell_for_tab(tabA, 1)

        # so tabB will face down to match it (side 3)
        cellB = self.get_cell_for_tab(tabB, 3)

        print(f"  {cellA=} {cellB=}")
        
        print("  left...")
        left = check_neighbors(2)
        
        print("  right...")
        right = check_neighbors(0)

        is_good = len(left) > 0 and len(right) > 0
        print("  is_good:", is_good)
        print()

        return is_good

    def test_single_tab(self, tabA):

        for tabB in self.get_matches_for_tab(tabA):
            self.test_tab_pair(tabA, tabB)

    def test_all_edges(self, G):

        for a, b in G.edges:
            self.test_tab_pair(str_to_tab(a), str_to_tab(b))

    def test_isolated_pairs(self):

        retval = {}
        
        for c in nx.weakly_connected_components(self.graph.graph):
            if len(c) != 2:
                continue
            tabA, tabB = (str_to_tab(i) for i in c)

            try:
                retval[tabA, tabB] = self.test_tab_pair(tabA, tabB)
            except (ValueError, MosaicException):
                retval[tabA, tabB] = None
                
        for (a, b), good in sorted(retval.items()):
            if good is None:
                g = 'Unknown'
            elif good:
                g = 'GOOD   '
            else:
                g = 'BAD    '
            print(f"{g} {a!s}={b!s}")

        return retval
                
    def check_tab_pair_for_rejects(self, tabA, tabB, expected=None):

        def check_neighbors(side_no):

            if cellA.sides[side_no] is None or cellB.sides[side_no] is None:
                return []

            tabAC = cellA.get_tab_for_side(side_no)
            tabBD = cellB.get_tab_for_side(side_no)

            rejects = []
            for tabCA, tabDB in itertools.product(
                    self.get_matches_for_tab(tabAC),
                    self.get_matches_for_tab(tabBD)):
                fps = [(tabA, tabB), (tabAC, tabCA), (tabBD, tabDB)]
                mse = self.raft_tester.get_raft_error(fps)
                if mse > 20.:
                    rejects.append(fps)

            return rejects

        #    +---+
        #  D | B | D'
        # -- +---+---
        #  C | A | C'
        #    +---+

        # arbitrarily make tabA face up (side 1)
        cellA = self.get_cell_for_tab(tabA, 1)

        # so tabB will face down to match it (side 3)
        cellB = self.get_cell_for_tab(tabB, 3)

        if any(cellA.sides[i] is None != cellB.sides[i] is None for i in (0, 2)):
            return [[(tabA, tabB)]]

        return check_neighbors(0) + check_neighbors(2)

    def check_all_tab_pairs_for_rejects(self, expected=None):

        G = self.graph.graph

        rejects = []
        for a, b in tqdm.tqdm(G.edges(), smoothing=0):
            try:
                rejects += self.check_tab_pair_for_rejects(
                    str_to_tab(a), str_to_tab(b), expected)
            except MosaicException:
                pass

        return rejects

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

def main():

    parser = argparse.ArgumentParser("mosaic")
    parser.add_argument("-p", "--puzzle", required=True)
    parser.add_argument("-t", "--tab-pairs", required=True)
    parser.add_argument("-e", "--expected")
    parser.add_argument("-o", "--output")
    parser.add_argument("-r", "--rocksdb")
    parser.add_argument("--fnord")

    args = parser.parse_args()

    puzzle = puzzler.file.load(args.puzzle)
    pieces = {i.label:i for i in puzzle.pieces}

    tab_pairs = puzzler.tabpairs.load_tab_pairs(args.tab_pairs)

    expected = read_expected(args.expected) if args.expected else None

    graph = MosaicGraph(tab_pairs)

    if args.fnord:
        G = graph.graph
        with open(args.fnord, 'r') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= 2000:
                    break
                dst, src = row['dst'], row['src']
                if not G.has_edge(dst, src):
                    G.add_edge(dst, src)
    
    mosaic = MosaicBuilder(pieces, graph, args.rocksdb)

    if False:
        mosaic.test_tab_pair(str_to_tab('Q14:2'), str_to_tab('N23:2'))
        return

    if True:
        G = graph.graph
        sat = MosaicSat(G.to_undirected())

        rejects = mosaic.check_all_tab_pairs_for_rejects()
        for i in rejects:
            s = ','.join(f"{a!s}={b!s}" for a, b in i)
            if expected:
                if all(expected.get(a,'') == b for a, b in i):
                    s += ' (WRONG, rejected a valid raft)'
                else:
                    s += ' (CORRECT, at least one bad match in raft)'
            print(f"reject {s}")
            sat.reject(i)
        
        edges = sat.check()
        if edges and args.output:
            with open(args.output, 'w') as f:
                for a, b in edges:
                    print(a, b, file=f)
        return

    if True:
        G = graph.graph
        c = get_weakly_connected_nodes(G, {'I18:0'})
        mosaic.test_all_edges(G.subgraph(c))
        return

    if True:
        for i in ('T7:3','Q19:1','R19:0','P31:3','Q17:3','C16:2','A5:1'):
            mosaic.test_single_tab(str_to_tab(i))
        return

    if True:
        mosaic.test_isolated_pairs()
        return
    
    cells = mosaic.breadth_first_search()

    print(f"placed {len(cells)} pieces!")

    if args.output:
        write_dotty(args.output, cells, expected)

if __name__ == '__main__':
    main()
