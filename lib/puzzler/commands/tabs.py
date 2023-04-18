import csv
import numpy as np
import re
import puzzler

from tqdm import tqdm

def output_tabs(args):

    puzzle = puzzler.file.load(args.puzzle)

    print("Tab alignment!")

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

        for dst in tqdm(pieces, ascii=True):
            tab_aligner = puzzler.align.TabAligner(dst)
            for dst_tab_no, dst_tab in enumerate(dst.tabs):
                rows = []
                for src in pieces:
                    if src is dst:
                        continue
                    for src_tab_no, src_tab in enumerate(src.tabs):
                        if dst_tab.indent == src_tab.indent:
                            continue
                        dst_row_no, dst_col_no = to_row_col(dst.label)
                        src_row_no, src_col_no = to_row_col(src.label)
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
                            
                        mse, src_coords, sfp, dfp = tab_aligner.compute_alignment(dst_tab_no, src, src_tab_no, refine=args.refine)
                        rows.append({'dst_label': dst.label,
                                     'dst_tab_no': dst_tab_no,
                                     'dst_col_no': dst_col_no,
                                     'dst_row_no': dst_row_no,
                                     'src_label': src.label,
                                     'src_tab_no': src_tab_no,
                                     'src_col_no': src_col_no,
                                     'src_row_no': src_row_no,
                                     'src_coord_x': src_coords.dxdy[0],
                                     'src_coord_y': src_coords.dxdy[1],
                                     'src_coord_angle': src_coords.angle,
                                     'src_index_0': sfp[0],
                                     'src_index_1': sfp[1],
                                     'mse': mse,
                                     'neighbor': neighbor})

                for i, row in enumerate(sorted(rows, key=lambda row: 10000 if row['mse'] is None else row['mse']), start=1):
                    row['rank'] = i
                writer.writerows(rows)

def add_parser(commands):

    parser_tabs = commands.add_parser("tabs", help="Output a CSV enumerating all possible tab matches")
    parser_tabs.add_argument("-o", "--output", help="output csv path")
    parser_tabs.add_argument("-r", "--refine", help="number of refinement passes", default=0, type=int)
    parser_tabs.set_defaults(func=output_tabs)
