import os, sys

# blech, fix up the path to find the project-specific modules
lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")
sys.path.insert(0, lib)

import argparse
import collections
import csv
import functools
import numpy as np
import puzzler
import tqdm
from dataclasses import dataclass

Feature = puzzler.raft.Feature

def normalize(v):
    # the set insures we remove duplicates which may have been
    # introduced by normalizing feature pair order, etc.
    s = set((b, a) if a > b else (a, b) for a, b in v)
    return tuple(sorted(s))

@dataclass
class Quad:
    spec: puzzler.raft.FeaturePairs
    raft: puzzler.raft.Raft
    fslp: puzzler.raft.FeaturePairs

class FrontierException(Exception):
    pass

class QuadProp:

    def __init__(self, pieces, quads):
        self.pieces = pieces
        self.raftinator = puzzler.raft.Raftinator(self.pieces)
        self.quads = []
        for i in quads:
            try:
                q = self.make_quad(i)
                self.quads.append(q)
            except FrontierException:
                pass

    def make_quad(self, s):
        r = self.raftinator
        spec = r.parse_feature_pairs(s)
        raft = r.make_raft_from_feature_pairs(spec)
        raft = r.refine_alignment_within_raft(raft)
        fslp = self.get_frontier_straight_line_pairs(raft)
        return Quad(spec, raft, fslp)

    def doit(self):

        r = self.raftinator
        for q in self.quads:
            print(f"quad \"{r.format_feature_pairs(q.spec)}\"")
            for p in q.fslp:
                c = self.get_quad_internal_connector_for_fslp(q, p)
                print(f"   slfp={r.format_feature_pair(p)}, c={'None' if c is None else r.format_feature_pair(c)}:", self.match_count(p, c))

    def match_count(self, p, c):
        n = 0
        for q in self.quads:
            if any(p[0] in i for i in q.spec) and any(p[1] in i for i in q.spec) and (c is None or c in q.spec):
                n += 1
        return n

    def get_quad_internal_connector_for_fslp(self, q, p):
        # p is a straight-line feature pair, i.e. two tabs pointing in
        # the same direction. If p[0] and p[1] are from different
        # pieces then we'd like those pieces to be connected as well
        p0, p1 = p[0].piece, p[1].piece
        if p0 == p1:
            return None
        
        if p0 > p1:
            p0, p1 = p1, p0
            
        for c in q.spec:
            if c[0].piece == p0 and c[1].piece == p1:
                return c

        return None

    def compute_tab_matches(self, raft):
        fp = puzzler.solver.compute_tab_matches(self.pieces, raft.coords)
        return normalize(fp)

    def get_frontier_straight_line_pairs(self, raft, verbose=False):
        if verbose:
            print("get_straightline_pairs:")
        retval = []
        tabs = self.get_detailed_tabs(raft)
        for i, curr in enumerate(tabs):
            prev = tabs[i-1]
            dot_product = np.sum(prev.tab_uv * curr.tab_uv)
            if verbose:
                with np.printoptions(precision=3):
                    print(f"  {prev=}")
                    print(f"  {curr=}")
                    print(f"  {dot_product=}")
            if dot_product > .9:
                retval.append((Feature(prev.piece,'tab',prev.index), Feature(curr.piece, 'tab', curr.index)))
        if verbose:
            print(f"  {retval=}")
        return retval

    def get_detailed_tabs(self, raft):

        r = self.raftinator

        closest_pieces = puzzler.frontier.ClosestPieces(
            self.pieces, raft.coords, None, r.factory.distance_query_cache)
        adjacency = dict((i, closest_pieces(i)) for i in raft.coords)
    
        bc = puzzler.frontier.BoundaryComputer(self.pieces)
        frontiers = bc.find_boundaries_from_adjacency(adjacency)
        if len(frontiers) != 1:
            raise FrontierException(f"raft has {len(frontiers)} frontiers, should be 1")
        
        fe = puzzler.frontier.FrontierExplorer(self.pieces, raft.coords)
        return fe.get_detailed_tabs(frontiers[0])

def read_quads(path):
    s = set()
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if float(row['mse']) < 100:
                s.add(row['raft'])
    return sorted(s)

def main():
    parser = argparse.ArgumentParser("quadprop")
    parser.add_argument("-p", "--puzzle", required=True)
    parser.add_argument("-q", "--quads", required=True)

    args = parser.parse_args()

    puzzle = puzzler.file.load(args.puzzle)
    pieces = {i.label:i for i in puzzle.pieces}

    quads = read_quads(args.quads)

    qp = QuadProp(pieces, quads)
    print(f"started with {len(quads)} quads, ended up with {len(qp.quads)} after checking frontiers")
    qp.doit()
        
if __name__ == '__main__':
    main()
