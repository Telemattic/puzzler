import puzzler
import concurrent.futures
from datetime import datetime
import json
import numpy as np
import operator
import os
import scipy.spatial.distance
import time
from typing import Any, Iterable, Mapping, NamedTuple, Optional, Sequence, Tuple

class Worker:

    def __init__(self, puzzle_path):

        puzzle = puzzler.file.load(puzzle_path)
        
        self.puzzle_path = puzzle_path
        self.pieces = {i.label: i for i in puzzle.pieces}

        # HACK: 100.json
        if 'I1' in self.pieces:
            p = self.pieces['I1']
            if len(p.edges) == 2:
                p.edges = puzzler.commands.ellipse.clean_edges(p.label, p.edges)

        self.edge_scorer = puzzler.solver.EdgeScorer(self.pieces)
        
    def score_edge(self, dst, sources):

        return (dst[0], self.edge_scorer.score_edge_piece(dst, sources))

    def score_pocket(self, raft, pocket, pieces):

        fitter = puzzler.commands.quads.PocketFitter(
            self.pieces, raft, pocket, 1)

        fits = []
        
        for src_label in pieces:
            for mse, feature_pairs in fitter.measure_fit(src_label):
                fits.append((mse[-1], src_label, feature_pairs))

        if not fits:
            return None
        
        fits.sort(key=operator.itemgetter(0))
        return fits[0]

WORKER = None

def worker_initialize(puzzle_path):
    global WORKER
    WORKER = Worker(puzzle_path)
    return None

def worker_score_edge(dst, sources):
    return WORKER.score_edge(dst, sources)

def worker_score_pocket(raft, pocket, pieces):
    return WORKER.score_pocket(raft, pocket, pieces)

# parallel solver

Raft = puzzler.raft.Raft

class ParallelSolver:

    def __init__(self, puzzle_path, max_workers, expected=None):
        
        self.executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers, initializer=worker_initialize, initargs=(puzzle_path,))
        
        puzzle = puzzler.file.load(puzzle_path)
        
        self.puzzle_path = puzzle_path
        self.pieces = {i.label: i for i in puzzle.pieces}
        self.raftinator = puzzler.raft.Raftinator(self.pieces)
        self.expected = expected

    def solve(self):

        raft = self.solve_border()

        raft = self.refine_raft(raft)

        raft, finished = self.solve_field(raft)
        raft = self.refine_raft(raft)
        while not finished:
            raft, finished = self.solve_field(raft)
            raft = self.refine_raft(raft)

        return raft

    def solve_border(self):

        bs = puzzler.solver.BorderSolver(self.pieces)
        scores = self.score_edges(bs.pred, bs.succ)

        border = bs.link_pieces(scores)
        raft = bs.init_placement(border)
        
        width, height = raft.size
        print(f"puzzle_size: width={width:.1f} height={height:.1f}")

        path = self.next_path()
        self.save(path, raft)
        print(f"initial raft saved to {path}")

        return raft

    def score_edges(self, pred, succ):

        def submit_helper(dst):
            return self.executor.submit(worker_score_edge, dst, pred)            

        fs = set(submit_helper(dst) for dst in succ.items())

        scores = {}
        for f in concurrent.futures.wait(fs).done:
            dst, sources = f.result()
            scores[dst] = sources

        return scores

    def solve_field(self, raft):

        def submit_helper(pocket):
            # only the pieces that define the pocket
            score_raft = Raft({i: raft.coords[i] for i in pocket.pieces})
            # can only fit pieces that haven't already been placed
            score_pieces = self.pieces.keys() - raft.coords.keys()
            print(f"score_pocket: {pocket}")
            return self.executor.submit(worker_score_pocket, score_raft, pocket, score_pieces)

        n_init = len(raft.coords)

        pockets = self.find_pockets(raft)

        fs = set(submit_helper(p) for p in pockets)
        while fs:
            done, not_done = concurrent.futures.wait(fs, return_when=concurrent.futures.FIRST_COMPLETED)
            for f in done:
                raft = self.place_piece(raft, f.result())
            fs = not_done

        n_placed = len(raft.coords) - n_init

        print(f"{n_placed} pieces placed, {len(raft.coords)} so far")
        
        path = self.next_path()
        self.save(path, raft)
        print(f"raft saved to {path}")

        return (raft, n_placed == 0)

    def place_piece(self, dst_raft, placement):

        if placement is None:
            return dst_raft

        mse, src_label, feature_pairs = placement

        r = self.raftinator
        
        s = r.format_feature_pairs(feature_pairs)
        print(f"pocket: {src_label} {mse=:.3f} {s}")
        
        if self.expected and any(self.expected.get(dst, '') != src for dst, src in feature_pairs):
            print(f"WARNING: apparent bad fit for {src_label}: {s}")

        if src_label in dst_raft.coords:
            print(f"INFO: {src_label} has already been placed, skipping")
            return dst_raft

        if mse > 20:
            print(f"WARNING: {src_label} placed with high error, rejecting")
            return dst_raft

        src_raft = r.factory.make_raft_for_piece(src_label)
        
        if True:
            src_coord = r.aligner.rough_align(dst_raft, src_raft, feature_pairs)
            src_coord = r.aligner.refine_alignment_between_rafts(
                dst_raft, src_raft, src_coord)
            raft = r.factory.merge_rafts(dst_raft, src_raft, src_coord)
        else:
            raft = r.align_and_merge_rafts_with_feature_pairs(
                dst_raft, src_raft, feature_pairs)
    
        self.check_raft_well_formed(raft)

        path = self.next_path()
        self.save(path, raft)
        
        print(f"place_piece: {src_label} placed, saved to {os.path.basename(path)}")
        
        return raft

    def refine_raft(self, raft):

        t0 = time.monotonic()
        
        r = self.raftinator

        seams = r.get_seams_for_raft(raft)
        rfc = puzzler.raft.RaftFeaturesComputer(self.pieces)
        axis_features = rfc.get_axis_features(raft.coords)
        retval = r.aligner.refine_alignment_within_raft(raft, seams, axis_features)

        t_refine = time.monotonic() - t0
        print(f"raft_refined: {t_refine=:.1f}s")
        
        return retval

    def find_pockets(self, raft):

        pf = puzzler.commands.quads.PocketFinder(self.pieces, raft)
        return pf.find_pockets_on_frontiers()

    def check_raft_well_formed(self, raft):

        labels = []
        xy = []
        for l, c in raft.coords.items():
            labels.append(l)
            xy.append(c.xy)
        xy = np.array(xy)

        d = scipy.spatial.distance.cdist(xy, xy)
        for i in range(len(xy)):
            d[i,i] = 10000.
            
        # print(f"{xy.shape=} {d.shape=} {np.argmin(d)=}")
        i, j = np.unravel_index(np.argmin(d, axis=None), d.shape)

        dist = d[i,j]
        a, b = labels[i], labels[j]
        # print(f"check_raft_well_formed: {a} and {b} are {dist:.1f} units apart")

        if dist < min(self.pieces[a].radius, self.pieces[b].radius):
            print(f"WARNING: pieces {a} and {b} are {dist:.1f} units apart!!!")

    def save(self, path, raft):

        coords = {k: {'angle':v.angle, 'xy':v.xy.tolist()} for k, v in raft.coords.items()}
        pieces = sorted(self.pieces.keys())

        o = {'pieces': pieces, 'raft': {'size': raft.size, 'coords': coords}}

        with open(path, 'w') as f:
            f.write(json.JSONEncoder(indent=0).encode(o))

    @staticmethod
    def next_path():

        fname = datetime.now().strftime('psolve_%Y%m%d-%H%M%S')
        ext = 'json'

        dname = r'C:\temp\puzzler\align'
        i = 0
        while True:
            path = os.path.join(dname, f"{fname}_{i:02d}.{ext}")
            if not os.path.exists(path):
                return path
            i += 1
