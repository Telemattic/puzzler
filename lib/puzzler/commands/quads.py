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

def tab_pairs_for_piece(piece):
    
    ref_angle = None
    dirs = [None, None, None, None]

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
            print(f"{label} has problem with tab_pairs_for_piece!")

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

def try_triples(pieces, quad):

    Feature = puzzler.raft.Feature
    
    raftinator = puzzler.raft.Raftinator(pieces)
    
    def remove_piece_from_raft(raft, label):
        coords = raft.coords.copy()
        coords.pop(label)
        frontiers = raftinator.factory.compute_frontiers(coords)
        new_raft = puzzler.raft.Raft(coords, frontiers)
        return raftinator.refine_alignment_within_raft(new_raft)

    def get_fit_features(feature_string, drop_label):

        fit = []
        drop = []
        for a, b in raftinator.parse_feature_pairs(feature_string):
            if a.piece == drop_label:
                drop.append(a)
                fit.append(b)
            if b.piece == drop_label:
                drop.append(b)
                fit.append(a)

        assert len(drop) in (1,2)
        
        if len(drop) == 2:
            drop_piece = pieces[drop_label]
            n = len(drop_piece.tabs)
            a, b = drop[0].index, drop[1].index
            if a + 1 == b or a + 1 == n and b == 0 and n > 2:
                do_reverse = False
            elif b + 1 == a or b + 1 == n and a == 0 and n > 2:
                do_reverse = True
            else:
                assert False, "tabs not adjacent????"
            if do_reverse:
                fit = fit[::-1]
                drop = drop[::-1]

        return fit

    def make_feature_pairs(try_piece, try_tab_pair, fit_features):

        tab_a, tab_b = try_tab_pair

        if tab_a is None:
            return None

        if len(fit_features) == 2 and tab_b is None:
            return None

        if len(fit_features) == 1 and tab_b is not None:
            return None

        for i, f in enumerate(fit_features):
            if try_tab_pair[i][2] == pieces[f.piece].tabs[f.index].indent:
                return None

        try_features = [Feature(try_piece.label, 'tab', tab_a[1])]
        if tab_b is not None:
            try_features.append(Feature(try_piece.label, 'tab', tab_b[1]))

        return list(zip(fit_features, try_features))

    debug = False

    # HACK
    if debug:
        print(f"raft_desc={quad['raft']}")
            
    good_raft = raftinator.make_raft_from_string(quad['raft'])

    retval2 = []

    for drop_quadrant in 'ul_piece', 'ur_piece', 'll_piece', 'lr_piece':

        drop_label = quad[drop_quadrant]

        # HACK
        if debug and drop_label != 'B2':
            continue

        triple_raft = remove_piece_from_raft(good_raft, drop_label)

        fit_features = get_fit_features(quad['raft'], drop_label)

        # HACK
        if debug:
            print(f"{drop_label=} {fit_features=}")

        retval = []

        for try_label, try_piece in pieces.items():
            
            if try_label in triple_raft.coords:
                continue

            # HACK
            if debug and try_label not in ('B2', 'G17'):
                continue

            try_raft = raftinator.factory.make_raft_for_piece(try_label)

            for try_tab_pair in tab_pairs_for_piece(try_piece):

                # tab_pairs_for_piece() returns tab pairs in CCW order
                # (nice job Matt) while tabs are number in CW order,
                # and we've chosen to order the feature pairs in CW
                # order for consistency.  In other words, reverse the
                # tab pairs now...
                try_tab_pair = try_tab_pair[1], try_tab_pair[0]
                
                if debug:
                    print(f"  {try_tab_pair=}")

                feature_pairs = make_feature_pairs(try_piece, try_tab_pair, fit_features)
                if feature_pairs is None:
                    if debug:
                        print("  -- not valid")
                    continue

                if debug:
                    print(f"  {feature_pairs=}")

                new_raft = raftinator.align_and_merge_rafts_with_feature_pairs(
                    triple_raft, try_raft, feature_pairs)

                for _ in range(3):
                    new_raft = raftinator.refine_alignment_within_raft(new_raft)

                mse = raftinator.get_total_error_for_raft_and_seams(new_raft)
                desc = quad['raft'].replace(drop_label, try_label)

                if debug:
                    print(f"  {mse=:.3f} {desc=}")

                row = {'ul_piece': quad['ul_piece'],
                       'ur_piece': quad['ur_piece'],
                       'll_piece': quad['ll_piece'],
                       'lr_piece': quad['lr_piece'],
                       'fit_piece': try_label,
                       'raft': desc,
                       'mse': mse,
                       'rank': None}
                row[drop_quadrant] = None
                
                retval.append(row)

        retval.sort(key=operator.itemgetter('mse'))
        for i, row in enumerate(retval, start=1):
            row['rank'] = i
            row['mse'] = decimal.Decimal(f"{row['mse']:.3f}")

        retval2 += retval

    return retval2

def triplets(pieces, input_csv_path, output_csv_path):

    assert len(pieces) == 1026

    quads = []
    with open(input_csv_path, 'r', newline='') as ifile:
        reader = csv.DictReader(ifile)
        for row in reader:
            if int(row['rank']) == 1:
                quads.append(row)

    with open(output_csv_path, 'w', newline='') as ofile:
        fieldnames = 'ul_piece ur_piece ll_piece lr_piece fit_piece raft mse rank'
        writer = csv.DictWriter(ofile, fieldnames=fieldnames.split())
        writer.writeheader()

        # HACK
        if False:
            writer.writerows(try_triples(pieces, quads[0]))
            return

        for quad in tqdm(quads, smoothing=0):
            writer.writerows(try_triples(pieces, quad))

def quads_main(args):

    puzzle = puzzler.file.load(args.puzzle)
    
    pieces = {p.label: p for p in puzzle.pieces}

    if args.quads is not None:
        triplets(pieces, args.quads, args.output)
        return
    
    quadrophenia(pieces, args.output)

def add_parser(commands):

    parser_quads = commands.add_parser("quads", help="process quads to compute expected matches")
    parser_quads.add_argument("-q", "--quads", help="inputs quads path, runs triplets algorithm")
    parser_quads.add_argument("-o", "--output", help="output CSV path for computed quads", required=True)
    parser_quads.set_defaults(func=quads_main)
