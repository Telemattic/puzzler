import puzzler
import concurrent.futures
import operator

class Worker:

    def __init__(self, puzzle_path):

        puzzle = puzzler.file.load(puzzle_path)
        
        self.puzzle_path = puzzle_path
        self.pieces = {i.label: i for i in puzzle.pieces}
        self.edge_scorer = None # let it be created lazily

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

WORKER = None

def worker_initialize(puzzle_path):

    global WORKER

    WORKER = Worker(puzzle_path)
    return None

def worker_score_edge(dst, sources):

    # print(f"worker_score_edge: {dst=} {sources=}")
    return WORKER.score_edge(dst, sources)

def worker_score_pocket(raft, pocket, pieces):

    # print(f"worker_score_pocket: {raft=} {pocket=} {pieces=}")
    ret = WORKER.score_pocket(raft, pocket, pieces)
    print(f"worker_score_pocket: {ret}")
    return ret

# parallel solver

Raft = puzzler.raft.Raft

SCORE_EDGE = 0
SCORE_POCKET = 1
REFINE_RAFT = 2

class ParallelSolver:

    def __init__(self, puzzle_path, max_workers):
        
        puzzle = puzzler.file.load(puzzle_path)
        
        self.puzzle_path = puzzle_path
        self.raft = None
        self.refine_job = None
        self.jobs = dict()
        self.pieces = {i.label: i for i in puzzle.pieces}
        self.executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers, initializer=worker_initialize, initargs=(puzzle_path,))
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

        for job, kind in self.wait_first_completed(timeout):

            if kind == SCORE_EDGE:
                self.edge_scored(job.result())

            else:
                raise ValueError("bad job")

        if not self.jobs:
            self.finish_border()

        return True

    def finish_border(self):

        bs = self.border_solver
        
        border = bs.link_pieces(self.edge_scores)
        geom = bs.init_placement(border)
        
        print(f"puzzle_size: width={geom.width:.1f} height={geom.height:.1f}")

        self.raft = puzzler.raft.Raft(geom.coords)

        pf = puzzler.commands.quads.PocketFinder(self.pieces)
        for pocket in pf.find_pockets(puzzler.raft.Raft(self.raft.coords)):
            self.score_pocket_async(pocket)
        
    def solve_field(self, timeout):

        for job, kind in self.wait_first_completed(timeout):

            if kind == SCORE_POCKET:
                self.pocket_scored(job.result())
                    
            elif kind == REFINE_RAFT:
                self.raft_refined(job.result())
                self.refine_job = None

            else:
                raise ValueError("bad job")

        if self.refine_job is None and len(self.placement) >= 20:
            self.refine_raft_async()

        return len(self.jobs) != 0

    def wait_first_completed(self, timeout):

        done, not_done = concurrent.futures.wait(
            self.jobs, timeout=timeout, return_when=concurrent.futures.FIRST_COMPLETED)
        return [(j, self.jobs.pop(j)) for j in done]

    def score_edge_async(self, dst, sources) -> None:
        f = self.executor.submit(worker_score_edge, dst, sources)
        self.jobs[f] = SCORE_EDGE

    def edge_scored(self, s) -> None:
        dst, scores = s
        self.edge_scores[dst] = scores

    def score_pocket_async(self, pocket) -> None:

        score_raft = puzzler.raft.Raft({i: self.raft.coords[i] for i in pocket.pieces})
        pieces = set(self.pieces) - set(self.raft.coords)

        f = self.executor.submit(worker_score_pocket, score_raft, pocket, pieces)
        self.jobs[f] = SCORE_POCKET

    def pocket_scored(self, p) -> None:

        self.placement.append(p)
        self.raft = self.place_piece(self.raft, p)

        for pocket in self.find_pockets_for_piece(p[0]):
            self.score_pocket_async(pocket)

    def refine_raft_async(self) -> None:

        f = self.executor.submit(worker_refine_raft, self.raft)
        self.futures[f] = REFINE_RAFT
        self.refine_job = f

    def raft_refined(self, r: Raft) -> None:

        # add to this refined raft all the pieces that were placed
        # subsequently to the job being created
        for p in self.placement:
            raft = self.place_piece(raft, p)
        self.placement = []
        self.raft = raft

    def place_piece(self, dst_raft, placement):

        src_label, feature_pairs = placement
        
        r = self.raftinator

        src_raft = r.factory.make_raft_for_piece(src_label)
        return r.align_and_merge_rafts_with_feature_pairs(
            dst_raft, src_raft, feature_pairs)

    def find_pockets_for_piece(self, label):
        raise NotImplementedError("find_pockets_for_piece")
