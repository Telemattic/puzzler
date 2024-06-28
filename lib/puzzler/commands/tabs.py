import csv
import numpy as np
import re
import puzzler
import multiprocessing as mp

from tqdm import tqdm

def to_row(s):
    row = 0
    for i in s.upper():
        row *= 26
        row += ord(i) + 1 - ord('A')
    return row

def to_col(s):
    return int(s)

def to_row_col(label):
    m = re.fullmatch("([a-zA-Z]+)(\d+)", label)
    return (to_row(m[1]), to_col(m[2])) if m else (None, None)

class TabsComputer:

    def __init__(self, puzzle_path, refine):
        self.puzzle = puzzler.file.load(puzzle_path)
        self.pieces = dict((i.label, i) for i in self.puzzle.pieces)
        self.refine = refine
        self.tab_aligner = None

    def compute_rows_for_dst(self, dst_label):

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

                    rows.append(self.compute_alignment_for_dst_src(dst_label, dst_tab_no, src.label, src_tab_no))
            for i, row in enumerate(sorted(rows, key=lambda row: 10000 if row['mse'] is None else row['mse']), start=1):
                row['rank'] = i

            retval += rows

        return retval

    def compute_alignment_for_dst_src(self, dst_label, dst_tab_no, src_label, src_tab_no):
        
        dst_row_no, dst_col_no = to_row_col(dst_label)
        src_row_no, src_col_no = to_row_col(src_label)
        neighbor = None
        if dst_row_no == src_row_no:
            if dst_col_no == src_col_no - 1:
                neighbor = 'E'
            if dst_col_no == src_col_no + 1:
                neighbor = 'W'
        elif dst_col_no == src_col_no:
            if dst_row_no == src_row_no - 1:
                neighbor = 'S'
            if dst_row_no == src_row_no + 1:
                neighbor = 'N'

        dst = self.pieces[dst_label]
        src = self.pieces[src_label]
        
        if not self.tab_aligner or self.tab_aligner.dst != dst:
            self.tab_aligner = puzzler.align.TabAligner(dst)

        mse, src_coords, sfp, dfp = self.tab_aligner.compute_alignment(
            dst_tab_no, src, src_tab_no, refine=self.refine)
        
        return {'dst_label': dst_label,
                'dst_tab_no': dst_tab_no,
                'dst_col_no': dst_col_no,
                'dst_row_no': dst_row_no,
                'src_label': src_label,
                'src_tab_no': src_tab_no,
                'src_col_no': src_col_no,
                'src_row_no': src_row_no,
                'src_coord_x': src_coords.dxdy[0],
                'src_coord_y': src_coords.dxdy[1],
                'src_coord_angle': src_coords.angle,
                'src_index_0': sfp[0],
                'src_index_1': sfp[1],
                'mse': mse,
                'neighbor': neighbor}

def worker(args, src_q, dst_q):

    tabs_computer = TabsComputer(args.puzzle, args.refine)

    job = src_q.get()
    while job:

        rows = tabs_computer.compute_rows_for_dst(job)

        dst_q.put(rows)
        job = src_q.get()

    return
    
def output_tabs(args):

    puzzle = puzzler.file.load(args.puzzle)

    print("Tab alignment!")

    def sort_key(piece):
        row, col = to_row_col(piece.label)
        return (row, col, piece.label)

    pieces = sorted([i for i in puzzle.pieces], key=sort_key)
    
    num_indents = 0
    num_outdents = 0
    for p in pieces:
        for t in p.tabs:
            if t.indent:
                num_indents += 1
            else:
                num_outdents += 1
    
    print(f"{len(pieces)} pieces: {num_indents} indents, {num_outdents} outdents")

    with open(args.output, 'w', newline='') as f:
        field_names = 'dst_label dst_tab_no dst_col_no dst_row_no src_label src_tab_no src_col_no src_row_no src_coord_x src_coord_y src_coord_angle src_index_0 src_index_1 mse neighbor rank'.split()
        writer = csv.DictWriter(f, field_names)
        writer.writeheader()

        if args.num_workers:
            src_q = mp.Queue()
            dst_q = mp.Queue()

            workers = [mp.Process(target=worker, args=(args, src_q, dst_q)) for _ in range(args.num_workers)]
            for p in workers:
                p.start()

            for p in pieces:
                src_q.put(p.label)
                
            num_jobs = len(pieces)
            pbar = tqdm(total=num_jobs, ascii=True)
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
            for dst in tqdm(pieces, ascii=True):
                writer.writerows(tabs_computer.compute_rows_for_dst(dst.label))

def add_parser(commands):

    parser_tabs = commands.add_parser("tabs", help="Output a CSV enumerating all possible tab matches")
    parser_tabs.add_argument("-o", "--output", help="output csv path")
    parser_tabs.add_argument("-r", "--refine",
                             help="number of refinement passes (default %(default)s)", default=0, type=int)
    parser_tabs.add_argument("-n", "--num-workers",  help="number of workers (default %(default)s)", default=0, type=int)
    parser_tabs.set_defaults(func=output_tabs)
