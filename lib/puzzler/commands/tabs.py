import csv
import decimal
import numpy as np
import puzzler
import multiprocessing as mp
import operator

from tqdm import tqdm

Feature = puzzler.raft.Feature

def format_feature(f):
    return f"{f.piece}:{f.index}" if f.kind == 'tab' else f"{f.piece}/{f.index}"

def format_feature_pair(p):
    return format_feature(p[0]) + '=' + format_feature(p[1])
    
class TabsComputer:

    def __init__(self, puzzle_path, refine):
        self.puzzle = puzzler.file.load(puzzle_path)
        self.pieces = dict((i.label, i) for i in self.puzzle.pieces)
        self.refine = refine

    def compute_rows_for_dst(self, dst_label):

        raftinator = puzzler.raft.Raftinator(self.pieces)

        retval = []

        dst = self.pieces[dst_label]
        
        for dst_tab_no, dst_tab in enumerate(dst.tabs):
            rows = []
            for src in self.pieces.values():
                if src is dst:
                    continue
                for src_tab_no, src_tab in enumerate(src.tabs):
                    if dst_tab.indent == src_tab.indent:
                        continue

                    feature_pair = (Feature(dst_label, 'tab', dst_tab_no), Feature(src.label, 'tab', src_tab_no))
                    
                    raft = raftinator.make_raft_from_feature_pairs([feature_pair])

                    for _ in range(3):
                        raft = raftinator.refine_alignment_within_raft(raft)

                    mse = raftinator.get_total_error_for_raft_and_seams(raft)
                    desc = format_feature_pair(feature_pair)

                    rows.append({'dst_label':dst_label, 'dst_tab_no':dst_tab_no, 'src_label':src.label, 'src_tab_no':src_tab_no, 'raft':desc, 'mse':mse, 'rank':None})

            rows.sort(key=operator.itemgetter('mse'))
            for i, row in enumerate(rows, start=1):
                row['rank'] = i
                row['mse'] = decimal.Decimal(f"{row['mse']:.3f}")

            retval += rows

        return retval

def worker(args, src_q, dst_q):

    tabs_computer = TabsComputer(args.puzzle, args.refine)

    job = src_q.get()
    while job:

        rows = tabs_computer.compute_rows_for_dst(job)

        dst_q.put(rows)
        job = src_q.get()

    return
    
def tabs_main(args):

    puzzle = puzzler.file.load(args.puzzle)

    print("Tab alignment!")

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
        field_names = 'dst_label dst_tab_no src_label src_tab_no raft mse rank'.split()
        writer = csv.DictWriter(f, field_names)
        writer.writeheader()

        if args.num_workers:
            src_q = mp.Queue()
            dst_q = mp.Queue()

            workers = [mp.Process(target=worker, args=(args, src_q, dst_q)) for _ in range(args.num_workers)]
            for p in workers:
                p.start()

            for p in puzzle.pieces:
                src_q.put(p.label)
                
            num_jobs = len(puzzle.pieces)
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
            tabs_computer = TabsComputer(args.puzzle, args.refine)
            for dst in tqdm(puzzle.pieces, smoothing=0):
                writer.writerows(tabs_computer.compute_rows_for_dst(dst.label))

def add_parser(commands):

    parser_tabs = commands.add_parser("tabs", help="Output a CSV enumerating all possible tab matches")
    parser_tabs.add_argument("-o", "--output", help="output csv path", required=True)
    parser_tabs.add_argument("-r", "--refine", default=2, type=int,
                             help="number of refinement passes (default %(default)i)")
    parser_tabs.add_argument("-n", "--num-workers", default=0, type=int,
                             help="number of workers (default %(default)i)")
    parser_tabs.set_defaults(func=tabs_main)
