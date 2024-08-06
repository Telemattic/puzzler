import puzzler
import concurrent.futures
import datetime
import numpy as np
import operator
import os
from typing import Any, Iterable, Mapping, NamedTuple, Optional, Sequence, Tuple

class Worker:

    def __init__(self, puzzle_path):

        puzzle = puzzler.file.load(puzzle_path)
        
        self.puzzle_path = puzzle_path
        self.pieces = {i.label: i for i in puzzle.pieces}
        self.edge_scorer = None # let it be created lazily
        self.raftinator = puzzler.raft.Raftinator(self.pieces)

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
        return fits[0][1:]

    def refine_raft(self, raft):

        r = self.raftinator

        seams = r.get_seams_for_raft(raft)
        axis_features = self.get_axis_features(raft)
        return r.aligner.refine_alignment_within_raft(raft, seams, axis_features)

    def get_axis_features(self, raft):

        rfc = puzzler.raft.RaftFeaturesComputer(self.pieces)
            
        border = None
        for frontier in rfc.compute_frontiers(raft.coords):
            if all(i.kind == 'edge' for i in frontier):
                border = frontier
                break

        if border is None:
            return None

        axes = rfc.split_frontier_into_axes(border, raft.coords)

        # convert from a dict to an array
        axes = [axes.get(i, []) for i in range(4)]

        fh = puzzler.raft.FeatureHelper(self.pieces, raft.coords)

        # rotate the array so that the "natural" axis 0 is first
        for i, axis in enumerate(axes):
            if len(axis) == 0:
                continue
            vec = fh.get_edge_unit_vector(axis[0])
            if np.dot(vec, np.array((-1, 0))) > .8:
                break

        if i < 4:
            axes = axes[i:] + axes[:i]

        return axes
    
WORKER = None

def worker_initialize(puzzle_path):
    global WORKER
    WORKER = Worker(puzzle_path)
    return None

def worker_score_edge(dst, sources):
    t_start = datetime.datetime.now()
    result = WORKER.score_edge(dst, sources)
    t_finish = datetime.datetime.now()
    return {'result':result, 't_start':t_start, 't_finish':t_finish, 'pid':os.getpid()}

def worker_score_pocket(raft, pocket, pieces):
    t_start = datetime.datetime.now()
    result = WORKER.score_pocket(raft, pocket, pieces)
    t_finish = datetime.datetime.now()
    return {'result':result, 't_start':t_start, 't_finish':t_finish, 'pid':os.getpid()}

def worker_refine_raft(raft):
    t_start = datetime.datetime.now()
    result = WORKER.refine_raft(raft)
    t_finish = datetime.datetime.now()
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
        t_submit = datetime.datetime.now()
        f = self.executor.submit(func, *args)
        self.job_no += 1
        self.jobs[f] = {'job_no': self.job_no, 'args': args, 'callback': callback, 't_submit':t_submit}
        return self.job_no

    def wait_first_completed(self, timeout):
        done, not_done = concurrent.futures.wait(
            self.jobs, timeout=timeout, return_when=concurrent.futures.FIRST_COMPLETED)
        return [self.jobs.pop(f) | f.result() for f in done]

    def num_jobs(self):
        return len(self.jobs)

class ParallelSolver:

    def __init__(self, puzzle_path, max_workers):
        
        self.job_manager = JobManager(puzzle_path, max_workers)
        
        puzzle = puzzler.file.load(puzzle_path)
        
        self.puzzle_path = puzzle_path
        self.raft = None
        self.refine_job = None
        self.pocket_jobs = dict()
        self.pieces = {i.label: i for i in puzzle.pieces}
        self.edge_scores = dict()
        self.raftinator = puzzler.raft.Raftinator(self.pieces)
        self.placement = []
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
        
        print(f"puzzle_size: width={geom.width:.1f} height={geom.height:.1f}")

        self.raft = puzzler.raft.Raft(geom.coords)

        pf = puzzler.commands.quads.PocketFinder(self.pieces, self.raft)
        for pocket in pf.find_pockets_on_frontiers():
            self.score_pocket_async(pocket)
        
    def solve_field(self, timeout):

        for job in self.job_manager.wait_first_completed(timeout):
            job['callback'](job)

        if not self.refine_job and len(self.placement) >= 20:
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
            print(f"score_pocket_async[----]: {pocket} already submitted as job {job_no}, skipping")
            return

        score_raft = puzzler.raft.Raft({i: self.raft.coords[i] for i in pocket.pieces})
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

        label, fp = result
        s = self.raftinator.format_feature_pairs(fp)
        print(f"pocket_scored[{job_no}]: {label:4s} {s}")

        self.placement.append(result)
        self.raft = self.place_piece(self.raft, result)

        for pocket in self.find_pockets_for_piece(label):
            self.score_pocket_async(pocket)

    def refine_raft_async(self) -> None:

        r = self.raft
        job_no = self.job_manager.submit_job((r,), worker_refine_raft, self.raft_refined)
        n_refine = len(r.coords)
        n_new = len(self.placement)
        
        self.refine_job = True
        self.placement = []
        
        print(f"refine_raft_async[{job_no}]: {n_refine} pieces, {n_new} of which are new")

    def raft_refined(self, job) -> None:

        job_no = job['job_no']
        result = job['result']

        r = result
        print(f"raft_refined[{job_no}]: {len(r.coords)} pieces, {len(self.placement)} new pieces to place")

        # add to this refined raft all the pieces that were placed
        # subsequently to the job being created
        for p in self.placement:
            r = self.place_piece(r, p)
        self.placement = []
        self.raft = r

    def place_piece(self, dst_raft, placement):

        src_label, feature_pairs = placement
        
        r = self.raftinator

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
