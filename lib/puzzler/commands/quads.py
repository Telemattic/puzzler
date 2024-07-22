import csv
import decimal
import itertools
import operator
import math
import numpy as np
import puzzler
from tqdm import tqdm

def axis_for_angle(phi):

    # [0, 2*pi) -> [0,4)
    phi = (phi % (2. * math.pi)) * 2. / math.pi
    if phi > 3.5:
        d = 0
    elif phi > 2.5:
        d = 3
    elif phi > 1.5:
        d = 2
    elif phi > 0.5:
        d = 1
    else:
        d = 0

    return d

def tab_pairs_for_piece(piece, verbose=False):
    
    ref_angle = None
    dirs = [None, None, None, None]

    if verbose:
        for tab_no, tab in enumerate(piece.tabs):

            x, y = tab.ellipse.center
            angle = math.atan2(y, x)

            print(f"tab[{tab_no}]: x,y={x:.1f},{y:.1f} {angle=:.3f} semi_major={tab.ellipse.semi_major:.3f} (degrees={math.degrees(angle):.0f})")
        
    for tab_no, tab in enumerate(piece.tabs):

        # HACK: skip known bad runt tab
        if piece.label == 'K33' and len(piece.tabs) == 5 and tab_no == 1:
            continue

        x, y = tab.ellipse.center
        angle = math.atan2(y, x)
            
        if ref_angle is None:
            ref_angle = angle
            d = 0
        else:
            d = axis_for_angle(angle - ref_angle)

        assert dirs[d] is None
        dirs[d] = (piece.label, tab_no, tab.indent)

    retval = []
    for i in range(4):
        a = dirs[i-1]
        b = dirs[i]
        if a is not None or b is not None:
            retval.append((a, b))

    return retval

def quadmaster(pieces, quad):

    Feature = puzzler.raft.Feature

    def make_feature_pair(a, b):
        if a is None or b is None:
            return None
        if a[2] == b[2]:
            raise ValueError("tab conflict")
        return (Feature(a[0],'tab',a[1]), Feature(b[0],'tab',b[1]))

    def make_feature_pairs(tab_pairs):
        
        retval = []

        try:
            for i in range(4):
                fp = make_feature_pair(tab_pairs[i-1][1], tab_pairs[i][0])
                if fp is not None:
                    retval.append(fp)
        except ValueError:
            retval = []

        return retval

    def format_feature(f):
        return f"{f.piece}:{f.index}" if f.kind == 'tab' else f"{f.piece}/{f.index}"

    def format_feature_pair(p):
        return format_feature(p[0]) + '=' + format_feature(p[1])

    def format_feature_pairs(pairs):
        return ','.join(format_feature_pair(i) for i in pairs)

    raftinator = puzzler.raft.Raftinator(pieces)

    ul, ur, ll, lr = quad

    ul_tabs = tab_pairs_for_piece(pieces[ul])
    ur_tabs = tab_pairs_for_piece(pieces[ur])
    ll_tabs = tab_pairs_for_piece(pieces[ll])
    lr_tabs = tab_pairs_for_piece(pieces[lr])

    retval = []
    # note tab_pairs are in CW order here
    for i in itertools.product(ul_tabs, ur_tabs, lr_tabs, ll_tabs):

        feature_pairs = make_feature_pairs(i)
        if len(feature_pairs) < 3:
            continue

        raft = raftinator.make_raft_from_feature_pairs(feature_pairs)

        # repeated refinement
        for _ in range(3):
            raft = raftinator.refine_alignment_within_raft(raft)
            
        mse = raftinator.get_total_error_for_raft_and_seams(raft)
        desc = format_feature_pairs(feature_pairs)
        
        retval.append({'raft':desc, 'mse':mse, 'rank':None})

    retval.sort(key=operator.itemgetter('mse'))
    for i, row in enumerate(retval, start=1):
        row['rank'] = i
        row['mse'] = decimal.Decimal(f"{row['mse']:.3f}")

    return retval
        
def quadrophenia(pieces, output_csv_path):

    assert len(pieces) == 1026

    rows = [chr(ord('A')+i) for i in range(26)] + ['AA']
    cols = [str(i) for i in range(1,39)]
    # print(f"{rows=} {cols=} n_pieces={len(rows)*len(cols)}")

    quads = []
    for row_no in range(len(rows)-1):
        for col_no in range(len(cols)-1):
            quads.append((row_no, col_no))

    for label, piece in pieces.items():
        try:
            tab_pairs_for_piece(piece)
        except AssertionError:
            print(f"{label} has problem with tab_pair_for_piece!")

    f = open(output_csv_path, 'w', newline='')
    writer = csv.DictWriter(f, fieldnames='col_no row_no ul_piece ur_piece ll_piece lr_piece raft mse rank'.split())
    writer.writeheader()

    for row_no, col_no in tqdm(quads, smoothing=0):
        
        row0, row1 = rows[row_no], rows[row_no+1]
        col0, col1 = cols[col_no], cols[col_no+1]
        quad = (row0+col0, row0+col1, row1+col0, row1+col1)

        props = {'row_no':row_no, 'col_no':col_no, 'ul_piece':quad[0], 'ur_piece':quad[1], 'll_piece':quad[2], 'lr_piece':quad[3]}
        
        for i in quadmaster(pieces, quad):
            writer.writerow(props | i)

def quads_main(args):

    puzzle = puzzler.file.load(args.puzzle)
    
    pieces = {p.label: p for p in puzzle.pieces}
    
    quadrophenia(pieces, args.output)

def add_parser(commands):

    parser_quads = commands.add_parser("quads", help="process quads to compute expected matches")
    parser_quads.add_argument("-o", "--output", help="output CSV path for computed quads", required=True)
    parser_quads.set_defaults(func=quads_main)
