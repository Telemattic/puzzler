import puzzler

class Worker:

    def __init__(self, puzzle_path):

        puzzle = puzzler.file.load(puzzle_path)
        
        self.puzzle_path = puzzle_path
        self.pieces = {i.label: i for i in puzzle.pieces}
        self.border_solver = None # let it be created lazily

    def score_edge_piece(self, dst, sources):

        if self.border_solver is None:
            self.border_solver = puzzler.solver.BorderSolver(self.pieces)

        return self.border_solver.score_edge_piece(dst, sources)

    def score_pocket(self, raft, pocket, pieces):

        fitter = puzzler.commands.quads.PocketFitter(
            self.pieces, raft, pocket, 1)

        fits = []
        
        for src_label in pieces:
            for mse, feature_pairs in fitter.measure_fit(src_label):
                fits.append((mse[-1], src_label, feature_pairs))

        return sorted(fits, key=operator.itemgetter(0))
        

WORKER = None

def initializer(puzzle_path):

    global WORKER

    WORKER = Worker(puzzle_path)
    return None

def score_edge_piece(dst, sources):
    
    return WORKER.score_edge_piece(dst, sources)

def score_pocket(raft, pocket, pieces):

    return WORKER.score_pocket(raft, pocket, pieces)
