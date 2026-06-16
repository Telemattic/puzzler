import os, sys

# blech, fix up the path to find the project-specific modules
lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")
sys.path.insert(0, lib)

import argparse
import collections
import csv
import functools
import numpy as np
import operator
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
    mse: float

class FrontierException(Exception):
    pass

class QuadMatch:

    def __init__(self, pieces, input_quads):
        self.pieces = pieces
        self.raftinator = puzzler.raft.Raftinator(self.pieces)
        self.quads = []

        unique_quads = collections.defaultdict(list)
        for mse, spec in input_quads:
            unique_quads[spec].append(mse)

        print(f"{len(unique_quads)} unique quads from {len(input_quads)} input quads")

        for spec, vec_mse in tqdm.tqdm(unique_quads.items(), smoothing=0):
            mse = min(vec_mse)
            try:
                if mse < 100.:
                    q = self.make_quad(spec, mse)
                    self.quads.append(q)
            except FrontierException:
                pass

        print(f"{len(self.quads)} usable quads from {len(unique_quads)} unique quads")

    def make_quad(self, spec, mse):
        r = self.raftinator
        spec = r.parse_feature_pairs(spec)
        raft = r.make_raft_from_feature_pairs(spec)
        raft = r.refine_alignment_within_raft(raft)
        fslp = self.get_frontier_straight_line_pairs(raft)
        return Quad(spec, raft, fslp, mse)

    def doit(self, dst: Quad):

        retval = []
        for i in dst.fslp:
            retval += self.doit2(dst, i)

        return retval

    def doit2(self, dst: Quad, i):

        retval = []
        for src in tqdm.tqdm(self.quads):
            for j in src.fslp:
                if m := self.try_slp_match(dst, i, src, j):
                    retval.append(m)

        retval.sort(key=operator.itemgetter('mse'))
        for rank, row in enumerate(retval, start=1):
            for key in ('dst_len', 'src_len', 'mse'):
                row[key] = f"{row[key]:.3f}"
            row['rank'] = rank

        return retval

    def doit3(self, expected):

        def is_good_quad(quad):
            for a, b in quad.spec:
                if expected.get(a,a) != b:
                    return False
            return True

        def rafts_overlap(a, b):
            return len(set(dst.raft.coords).intersection(src.raft.coords)) != 0

        def F(x):
            return Feature(x.piece,'tab',x.index)

        def FP(x):
            return (F(x[0]), F(x[1]))

        def is_good_match(i, j):
            a, b = F(i[0]), F(j[1])
            if expected.get(a,a) != b:
                return False
            a, b = F(i[1]), F(j[0])
            if expected.get(a,a) != b:
                return False
            return True

        def L(x):
            return np.linalg.norm(x[0].tab_xy - x[1].tab_xy)

        retval = []
        
        good_quads = [i for i in self.quads if is_good_quad(i)]
        r = self.raftinator

        print(f"{len(good_quads)} good quads (all tab matches are correct)")

        for dst in tqdm.tqdm(self.quads, smoothing=0):
            for src in self.quads:
                if rafts_overlap(dst.raft, src.raft):
                    continue
                for i in dst.fslp:
                    for j in src.fslp:
                        if i[0].indent == j[1].indent or i[1].indent == j[0].indent:
                            continue
                        delta = abs(L(i) - L(j))
                        good_match = is_good_match(i, j)
                        retval.append({'dst_quad':r.format_feature_pairs(dst.spec),
                                       'dst_good':is_good_quad(dst),
                                       'dst_mse':dst.mse,
                                       'dst_fslp':r.format_feature_pair(FP(i)),
                                       'dst_len':L(i),
                                       'src_quad':r.format_feature_pairs(src.spec),
                                       'src_good':is_good_quad(src),
                                       'src_mse':src.mse,
                                       'src_fslp':r.format_feature_pair(FP(j)),
                                       'src_len':L(j),
                                       'match_good':is_good_match(i,j)})

        return retval

    def try_slp_match(self, dst: Quad, i, src: Quad, j):

        if i[0].indent == j[1].indent or i[1].indent == j[0].indent:
            return None

        if set(dst.raft.coords).intersection(src.raft.coords):
            return None

        def F(x):
            return Feature(x.piece,'tab',x.index)

        match_fps = [(F(i[0]), F(j[1])), (F(i[1]), F(j[0]))]

        r = self.raftinator
        raft = r.align_and_merge_rafts_with_feature_pairs(dst.raft, src.raft, match_fps)
        raft = r.refine_alignment_within_raft(raft)

        mse = r.get_total_error_for_raft_and_seams(raft)

        def L(x):
            return np.linalg.norm(x[0].tab_xy - x[1].tab_xy)
        
        return {'dst_quad': r.format_feature_pairs(dst.spec),
                'src_quad': r.format_feature_pairs(src.spec),
                'match': r.format_feature_pairs(match_fps),
                'dst_len': L(i), 'src_len': L(j), 'mse': mse}

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
                retval.append((prev, curr))
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
    retval = []
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            retval.append((float(row['mse']), row['raft']))
    return retval

def read_expected(path):

    expected = {}
    with open(path, 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            dst = Feature(row['dst_piece'], 'tab', int(row['dst_tab_no']))
            src = Feature(row['src_piece'], 'tab', int(row['src_tab_no']))
            expected[dst] = src
            
    return expected

def output_quad_fslps(path, quads):
    
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames='quad_no fslp_no tab0 indent0 tab1 indent1 length'.split())
        writer.writeheader()
        
        for quad_no, quad in enumerate(quads):
            for fslp_no, fslp in enumerate(quad.fslp):
                tab0, tab1 = fslp
                writer.writerow({'quad_no': quad_no,
                                 'fslp_no': fslp_no,
                                 'tab0': tab0.piece + ':' + str(tab0.index),
                                 'tab1': tab1.piece + ':' + str(tab1.index),
                                 'indent0': tab0.indent,
                                 'indent1': tab1.indent,
                                 'length': np.linalg.norm(tab0.tab_xy - tab1.tab_xy)})
        
def main():
    parser = argparse.ArgumentParser("quadmatch")
    parser.add_argument("-p", "--puzzle", required=True)
    parser.add_argument("-q", "--quads", required=True)
    parser.add_argument("-e", "--expected", required=True)
    parser.add_argument("-o", "--output", required=True)

    args = parser.parse_args()

    puzzle = puzzler.file.load(args.puzzle)
    pieces = {i.label:i for i in puzzle.pieces}

    quads = read_quads(args.quads)
    expected = read_expected(args.expected)

    print("initializing...")

    qm = QuadMatch(pieces, quads)

    fieldnames = 'dst_quad dst_good dst_mse dst_fslp dst_len src_quad src_good src_mse src_fslp src_len match_good'.split()
    with open(args.output, 'w', newline='') as f:
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(qm.doit3(expected))
        
if __name__ == '__main__':
    main()
