import puzzler
import concurrent.futures
from datetime import datetime
import json
import numpy as np
import operator
import os
import scipy.spatial.distance
from typing import Any, Iterable, Mapping, NamedTuple, Optional, Sequence, Tuple

class Worker:

    def __init__(self, puzzle_path):

        puzzle = puzzler.file.load(puzzle_path)
        
        self.puzzle_path = puzzle_path
        self.pieces = {i.label: i for i in puzzle.pieces}
        self.edge_scorer = None # let it be created lazily
        self.raftinator = puzzler.raft.Raftinator(self.pieces)

        # HACK: 100.json
        if 'I1' in self.pieces:
            p = self.pieces['I1']
            if len(p.edges) == 2:
                p.edges = puzzler.commands.ellipse.clean_edges(p.label, p.edges)

    def score_edge(self, dst, sources):

        if self.edge_scorer is None:
            self.edge_scorer = puzzler.solver.EdgeScorer(self.pieces)

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

    def refine_raft(self, raft):

        r = self.raftinator

        seams = r.get_seams_for_raft(raft)
        rfc = puzzler.raft.RaftFeaturesComputer(self.pieces)
        axis_features = rfc.get_axis_features(raft.coords)
        return r.aligner.refine_alignment_within_raft(raft, seams, axis_features)

WORKER = None

def worker_initialize(puzzle_path):
    global WORKER
    WORKER = Worker(puzzle_path)
    return None

def worker_score_edge(dst, sources):
    t_start = datetime.now()
    result = WORKER.score_edge(dst, sources)
    t_finish = datetime.now()
    return {'result':result, 't_start':t_start, 't_finish':t_finish, 'pid':os.getpid()}

def worker_score_pocket(raft, pocket, pieces):
    t_start = datetime.now()
    result = WORKER.score_pocket(raft, pocket, pieces)
    t_finish = datetime.now()
    return {'result':result, 't_start':t_start, 't_finish':t_finish, 'pid':os.getpid()}

def worker_refine_raft(raft):
    t_start = datetime.now()
    result = WORKER.refine_raft(raft)
    t_finish = datetime.now()
    return {'result':result, 't_start':t_start, 't_finish':t_finish, 'pid':os.getpid()}

# parallel solver

Raft = puzzler.raft.Raft

class JobManager:

    def __init__(self, puzzle_path, max_workers):
        self.executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers, initializer=worker_initialize, initargs=(puzzle_path,))
        self.jobs = dict()
        self.job_no = 0

    def submit_job(self, args, func, callback):
        t_submit = datetime.now()
        f = self.executor.submit(func, *args)
        self.job_no += 1
        self.jobs[f] = {'job_no': self.job_no, 'args': args, 'callback': callback, 't_submit':t_submit}
        return self.job_no

    def wait_first_completed(self, timeout):
        done, not_done = concurrent.futures.wait(
            self.jobs, timeout=timeout, return_when=concurrent.futures.FIRST_COMPLETED)
        return [self.jobs.pop(f) | f.result() for f in done]

    def wait_for_specific_job(self, job_no):
        for k, v in self.jobs:
            if v['job_no'] == job_no:
                f = k
                break
        else:
            raise KeyError("no such job")

        return self.jobs.pop(f) | f.result()

    def num_jobs(self):
        return len(self.jobs)

class Solver:

    def solve_border(self):

        scores = self.score_edges()
        border = self.link_border(scores)
        raft = self.construct_border_raft(border)
        return self.refine_raft(raft)

    def score_edges(self):

        pred, succ = self.identify_edges()
        return {self.score_edge(dst, pred) for dst in succ.items()}

    def identify_edges(self):
        pass

    def score_edge(self, dst, sources):
        return (dst[0], self.edge_scorer.score_edge_piece(dst, sources))

    def link_edges(self, scores):
        return EdgeLinker(self.pieces).link_edges(scores)

    def construct_border_raft(self, border):
        return BorderConstructor(self.pieces).construct_border_raft(border)

    def iter_field(self, raft):

        for p in self.score_pockets(raft):
            raft = self.place_piece(raft, p)
        return self.refine_raft(raft)

    def score_pockets(self, raft):

        return [self.score_pocket(raft, p) for p in self.find_pockets(raft)]

    def find_pockets(self, raft):

        return PocketFinder(self.pieces, self.raft).find_pockets_on_frontiers()

    def score_pocket(self, raft, pocket):

        return self.pocket_scorer.score_pocket(raft, pocket)

    def place_piece(self, raft, placement):

        src_label, feature_pairs = placement
        return PiecePlacer(self.places).place_piece(raft, src_label, feature_pairs)

    def refine_raft(self, raft):
        r = self.raftinator

        seams = r.get_seams_for_raft(raft)
        axis_features = self.get_axis_features(raft)
        return r.aligner.refine_alignment_within_raft(raft, seams, axis_features)

class ParallelSolver:

    def __init__(self, puzzle_path, max_workers, expected=None):
        
        self.job_manager = JobManager(puzzle_path, max_workers)
        
        puzzle = puzzler.file.load(puzzle_path)
        
        self.puzzle_path = puzzle_path
        self.raft = None
        self.refine_job = 0
        self.pocket_jobs = dict()
        self.pieces = {i.label: i for i in puzzle.pieces}
        self.edge_scores = dict()
        self.raftinator = puzzler.raft.Raftinator(self.pieces)
        self.placement = []
        self.expected = expected
        self.generation = dict()
        self.start_solve()

    def solve(self, timeout=None):

        return self.solve_border(timeout) if self.raft is None else self.solve_field(timeout)

    def start_solve(self):

        bs = self.border_solver = puzzler.solver.BorderSolver(self.pieces)
        for dst in bs.succ.items():
            self.score_edge_async(dst, bs.pred)

    def solve_border(self, timeout):

        for job in self.job_manager.wait_first_completed(timeout):
            job['callback'](job)

        if 0 == self.job_manager.num_jobs():
            self.finish_border()

        return True

    def finish_border(self):

        bs = self.border_solver
        
        border = bs.link_pieces(self.edge_scores)
        geom = bs.init_placement(border)

        width, height = geom.size
        print(f"puzzle_size: width={width:.1f} height={height:.1f}")

        self.raft = puzzler.raft.Raft(geom.coords, geom.size)

        self.generation = {i: 0 for i in self.raft.coords}
        
        pf = puzzler.commands.quads.PocketFinder(self.pieces, self.raft)
        for pocket in pf.find_pockets_on_frontiers():
            self.score_pocket_async(pocket)
        
        path = self.next_path()
        self.save(path)
        print(f"finish_border[-]: initial raft saved to {path}")

    def solve_field(self, timeout):

        for job in self.job_manager.wait_first_completed(timeout):
            job['callback'](job)

        # start a refinement job if we've placed a bunch of pieces and
        # haven't refined their placement, *or* if we've run out of
        # work and have at least one piece without a refined placement
        thresh = 1 if self.job_manager.num_jobs() == 0 else 20
        if 0 == self.refine_job and len(self.placement) >= thresh:
            self.refine_raft_async()

        return 0 != self.job_manager.num_jobs()

    def score_edge_async(self, dst, sources) -> None:

        args = (dst, sources)
        self.job_manager.submit_job(args, worker_score_edge, self.edge_scored)

    def edge_scored(self, job) -> None:
        
        job_no = job['job_no']
        result = job['result']
        
        dst, scores = result
        self.edge_scores[dst] = scores

    def score_pocket_async(self, pocket) -> None:

        if pocket in self.pocket_jobs:
            job_no = self.pocket_jobs[pocket]
            print(f"score_pocket_async[-]: {pocket} already submitted as job {job_no}, skipping")
            return

        g = 1 + max(self.generation[i] for i in pocket.pieces)
        if g > 3:
            print(f"score_pocket_async[-]: {pocket} would be generation {g}, skipping")
            return

        score_raft = puzzler.raft.Raft({i: self.raft.coords[i] for i in pocket.pieces}, None)
        pieces = set(self.pieces) - set(self.raft.coords)
        args = (score_raft, pocket, pieces)
        
        job_no = self.job_manager.submit_job(args, worker_score_pocket, self.pocket_scored)
        print(f"score_pocket_async[{job_no}]: {pocket}")
        self.pocket_jobs[pocket] = job_no

    def pocket_scored(self, job) -> None:

        job_no = job['job_no']
        result = job['result']
        if result is None:
            print(f"pocket_scored[{job_no}]: None?")
            return

        pocket = job['args'][1]

        mse, label, feature_pairs = result
        s = self.raftinator.format_feature_pairs(feature_pairs)
        print(f"pocket_scored[{job_no}]: {label} {mse=:.3f} {s}")
        
        if self.expected and any(self.expected.get(dst, '') != src for dst, src in feature_pairs):
            print(f"pocket_scored[{job_no}]: WARNING: known bad fit for {label}: {s}")

        if label in self.raft.coords:
            print(f"pocket_scored[{job_no}]: {label} has already been placed, skipping")
            return

        if mse > 20:
            print(f"pocket_scored[{job_no}]: WARNING: {label} placed with high error, rejecting")
            return

        self.placement.append((label, feature_pairs))
        self.raft = self.place_piece(self.raft, (label, feature_pairs))

        self.check_raft_well_formed(self.raft)
        
        for pocket in self.find_pockets_for_piece(label):
            self.score_pocket_async(pocket)

        path = self.next_path()
        self.save(path)
        print(f"pocket_scored[{job_no}]: {label} is generation {self.generation[label]}")
        print(f"pocket_scored[{job_no}]: updated raft saved to {os.path.basename(path)}")

    def refine_raft_async(self) -> None:

        r = self.raft
        job_no = self.job_manager.submit_job((r,), worker_refine_raft, self.raft_refined)
        n_refine = len(r.coords)
        n_new = len(self.placement)
        
        self.refine_job = job_no
        self.placement = []
        
        print(f"refine_raft_async[{job_no}]: {n_refine} pieces, {n_new} of which are new")

    def raft_refined(self, job) -> None:

        job_no = job['job_no']
        result = job['result']

        r = result
        print(f"raft_refined[{job_no}]: {len(r.coords)} pieces, {len(self.placement)} new pieces to place")

        # only the pieces that had their coordinates refined should be
        # considered generation 0, everything else is at least
        # generation 1, and potentially higher
        self.generation = {i: 0 for i in r.coords}

        # add to this refined raft all the pieces that were placed
        # subsequently to the job being created
        for p in self.placement:
            r = self.place_piece(r, p)
            
        n_placed = len(self.placement)
        self.placement = []
        
        self.refine_job = 0
        self.raft = r

        new_pockets = 0
        pf = puzzler.commands.quads.PocketFinder(self.pieces, self.raft)
        for pocket in pf.find_pockets_on_frontiers():
            if pocket not in self.pocket_jobs:
                self.score_pocket_async(pocket)
                new_pockets += 1
        
        path = self.next_path()
        self.save(path)
        print(f"raft_refined[{job_no}]: submitted {new_pockets} new pockets")
        print(f"raft_refined[{job_no}]: updated raft saved to {os.path.basename(path)}")

        if n_placed >= 20:
            self.refine_raft_async()

        self.check_raft_well_formed(self.raft)

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

    def place_piece(self, dst_raft, placement):

        src_label, feature_pairs = placement
        
        r = self.raftinator

        if src_label in dst_raft.coords:
            raise ValueError("place_piece")

        self.generation[src_label] = 1 + max(self.generation[i.piece] for i, j in feature_pairs)

        src_raft = r.factory.make_raft_for_piece(src_label)
        return r.align_and_merge_rafts_with_feature_pairs(
            dst_raft, src_raft, feature_pairs)

    def find_pockets_for_piece(self, label):

        fc = puzzler.raft.RaftFeaturesComputer(self.pieces)
        tabs = fc.compute_frontier_tabs_for_piece(self.raft.coords, label)

        pockets = []
        if tabs:
            pf = puzzler.commands.quads.PocketFinder(self.pieces, self.raft)
            pockets = pf.find_pockets_on_frontiers([tabs])

        return pockets

    def save(self, path):

        r = self.raft
        coords = {k: {'angle':v.angle, 'xy':v.xy.tolist()} for k, v in r.coords.items()}
        pieces = sorted(self.pieces.keys())

        o = {'pieces': pieces, 'geometry': {'size': r.size, 'coords': coords}}

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
