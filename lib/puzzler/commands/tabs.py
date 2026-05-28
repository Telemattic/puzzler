import csv
import decimal
import numpy as np
import puzzler
import multiprocessing as mp
import operator
import re
import itertools

from tqdm import tqdm

Feature = puzzler.raft.Feature

def format_feature(f):
    return f"{f.piece}:{f.index}" if f.kind == 'tab' else f"{f.piece}/{f.index}"

def format_feature_pair(p):
    return format_feature(p[0]) + '=' + format_feature(p[1])

def row_col_of_piece(p):
    m = re.match("^([A-Z]+)([0-9]+)", p.label)
    c = int(m[2])
    r = 0
    for i in m[1]:
        r = r*26 + (ord(i) - ord('A'))
    return (r, c)

def iterate_over_tabs(pieces):

    outdents = []
    indents = []

    for p in sorted(pieces, key=row_col_of_piece):

        for tab_no, tab in enumerate(p.tabs):
            f = Feature(p.label, 'tab', tab_no)
            if tab.indent:
                indents.append(f)
            else:
                outdents.append(f)

    #print(f"outdents={','.join(str(i) for i in outdents)}")
    #print(f"indents={','.join(str(i) for i in indents)}")

    # keep the batchsize larger to keep the grain from getting too
    # small, but don't let it get too large and blow out caching
    bs = 16
    n = len(outdents) + len(indents)
    while bs * bs < n and bs < 64:
        bs += 1

    #print(f"batchsize={bs}")

    return list(itertools.product(itertools.batched(outdents, bs), itertools.batched(indents, bs)))
    
class TabsComputer:

    def __init__(self, pieces, refine):
        self.pieces = dict((i.label, i) for i in pieces)
        self.refine = refine
        self.raftinator = puzzler.raft.Raftinator(self.pieces)

    def compute_rows(self, tabs_a, tabs_b):

        retval = []

        for i in tabs_a:
            for j in tabs_b:
                if i.piece == j.piece:
                    continue
                retval.append(self.compute_fit(i, j))
                retval.append(self.compute_fit(j, i))

        return retval

    def compute_fit(self, dst, src):

        desc = format_feature_pair((dst, src))

        retval = {'dst_label':dst.piece, 'dst_tab_no':dst.index,
                  'src_label':src.piece, 'src_tab_no':src.index,
                  'raft':desc, 'sse':None, 'n':None}
        try:
            r = self.raftinator

            raft = r.align_and_merge_rafts_with_feature_pairs(
                r.factory.make_raft_for_piece(dst.piece),
                r.factory.make_raft_for_piece(src.piece),
                [(dst, src)])

            for i in range(self.refine):
                seam = r.seamstress.seam_between_pieces(
                    dst.piece, raft.coords[dst.piece], src.piece, raft.coords[src.piece])
                raft = r.refine_alignment_within_raft(raft, seams=[seam], fixed=dst.piece)

            seam = r.seamstress.seam_between_pieces(
                dst.piece, raft.coords[dst.piece], src.piece, raft.coords[src.piece])

            fit_error = puzzler.raft.FitError(seam.error, len(seam.src.indices))
            retval['sse'] = fit_error.sse
            retval['n'] = fit_error.n
        except Exception as x:
            print(f"error processing {desc},", x)

        return retval

def worker(args, src_q, dst_q):

    puzzle = puzzler.file.load(args.puzzle)
    tabs_computer = TabsComputer(puzzle.pieces, args.refine)

    job = src_q.get()
    while job:

        rows = tabs_computer.compute_rows(*job)

        dst_q.put(rows)
        job = src_q.get()

    return
    
def tabs_main(args):

    puzzle = puzzler.file.load(args.puzzle)

    print("Tab alignment!")

    # HACK
    if False:
        puzzle.pieces = [i for i in puzzle.pieces if i.label in ('B4', 'A5', 'O20')]
                
    num_indents = 0
    num_outdents = 0
    for p in puzzle.pieces:
        for t in p.tabs:
            if t.indent:
                num_indents += 1
            else:
                num_outdents += 1
    
    print(f"{len(puzzle.pieces)} pieces: {num_indents} indents, {num_outdents} outdents")

    with open(args.output, 'w', newline='') as f:
        field_names = 'dst_label dst_tab_no src_label src_tab_no raft sse n mse rank'.split()
        writer = csv.DictWriter(f, field_names)
        writer.writeheader()

        if args.num_workers:
            src_q = mp.Queue()
            dst_q = mp.Queue()

            workers = [mp.Process(target=worker, args=(args, src_q, dst_q)) for _ in range(args.num_workers)]
            for p in workers:
                p.start()

            num_jobs = 0
            for p in iterate_over_tabs(puzzle.pieces):
                src_q.put(p)
                num_jobs += 1
                
            pbar = tqdm(total=num_jobs, smoothing=0)
            while num_jobs > 0:
                job = dst_q.get()
                num_jobs -= 1
                pbar.update()
                writer.writerows(job)

            for _ in workers:
                src_q.put(None)

            for p in workers:
                p.join()

        else:
            tabs_computer = TabsComputer(puzzle.pieces, args.refine)
            
            pieces = puzzle.pieces[:args.max_piece] if args.max_piece is not None else puzzle.pieces
            
            for tabs_a, tabs_b in tqdm(iterate_over_tabs(pieces), smoothing=0):
                writer.writerows(tabs_computer.compute_rows(tabs_a, tabs_b))

def add_parser(commands):

    parser_tabs = commands.add_parser("tabs", help="Output a CSV enumerating all possible tab matches")
    parser_tabs.add_argument("-o", "--output", help="output csv path", required=True)
    parser_tabs.add_argument("-r", "--refine", default=3, type=int,
                             help="number of refinement passes (default %(default)i)")
    parser_tabs.add_argument("-n", "--num-workers", default=0, type=int,
                             help="number of workers (default %(default)i)")
    parser_tabs.add_argument("--max-piece", default=None, type=int,
                             help="number of pieces to process, useful for limited profiling runs")
    parser_tabs.set_defaults(func=tabs_main)
