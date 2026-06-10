import collections
import csv
import decimal
import itertools
import operator
import math
import multiprocessing as mp
import numpy as np
import puzzler
import re
import scipy
from typing import Any, Iterable, Mapping, Optional, Sequence, Set, Tuple

import logging

logger = logging.getLogger('puzzler')

from tqdm import tqdm

HACK = False

def label_to_row_col(s):
    m = re.fullmatch(r"([A-Z]+)(\d+)", s)
    c = int(m[2])
    r = 0
    for i in m[1]:
        r = r*26 + ord(i) - ord('A') + 1
    return (r-1, c-1)

def row_col_to_label(r, c):
    s = ''
    if r >= 26:
        s = 'A'
        r -= 26

    assert 0 <= r < 26
    s = s + chr(r + ord('A'))
    return s + str(c+1)

def griddy(pieces):

    grid = {label_to_row_col(i): i for i in pieces.keys()}

    num_row = 1+max(r for r, c in grid.keys())
    num_col = 1+max(c for r, c in grid.keys())
    print(f"{num_row=} {num_col=}")

    missing = []
    for r in range(num_row):
        for c in range(num_col):
            if (r, c) not in grid:
                missing.append((r,c))

    fixed = {}
    for r, c in missing:
        s = set()
        if r > 0 and (r-1,c) in grid:
            s.add(row_col_to_label(r-1,c))
        if c > 0 and (r,c-1) in grid:
            s.add(row_col_to_label(r,c-1))
        print(f"missing piece {r},{c} ({row_col_to_label(r, c)}), covered by one of {', '.join(s)}")
        v = sorted((len(pieces[i].tabs), i) for i in s)
        print(f"  choosing {v[-1][1]}")
        fixed[r,c] = v[-1][1]

    if fixed:
        print(f"{fixed=}")

    grid |= fixed

    return (num_row, num_col, grid)

# see also puzzler.solver.compute_tab_matches
def compute_tab_matches(pieces, raft):
    tab_xy = []
    radii = []
    features = []
    for k, v in raft.coords.items():
        p = pieces[k]
        centers = np.array([t.ellipse.center for t in p.tabs])
        radii  += [t.ellipse.semi_major for t in p.tabs]
        features += [puzzler.raft.Feature(p.label, 'tab', i) for i in range(len(p.tabs))]
        tab_xy += [xy for xy in v.xform.apply_v2(centers)]

    retval = set()

    kdtree = scipy.spatial.KDTree(tab_xy)
    neighbor_dist, neighbor_index = kdtree.query(tab_xy, 2)
    for i, neighbors in enumerate(neighbor_index):
        for j, k in enumerate(neighbors):
            if k == i:
                continue
            distance = neighbor_dist[i][j]
            if distance > radii[i]:
                continue
            # put the feature in a deterministic order
            a = features[i]
            b = features[k]
            if b < a:
                a, b = b, a
            retval.add((a,b))

    return retval

def quadmaster(pieces, quad):

    Feature = puzzler.raft.Feature

    def make_feature_pair(a, b):
        if a is None or b is None:
            return None
        if a is None or b is None:
            raise ValueError("tab can't match missing tab")
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

    raftinator = puzzler.raft.Raftinator(pieces)
    
    def FFP(x):
        return raftinator.format_feature_pairs(sorted(x))

    ul = quad['ul_piece']
    ur = quad['ur_piece']
    ll = quad['ll_piece']
    lr = quad['lr_piece']

    tab_features_for_piece = puzzler.pocket.tab_features_for_piece

    try:
        ul_tabs = tab_features_for_piece(pieces[ul])
        ur_tabs = tab_features_for_piece(pieces[ur])
        ll_tabs = tab_features_for_piece(pieces[ll])
        lr_tabs = tab_features_for_piece(pieces[lr])
    except puzzler.pocket.PocketFitter.FitException as x:
        print(f"skipping quad {ul},{ur},{ll},{lr} due to error, ", x)
        return []
    
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
        desc = raftinator.format_feature_pairs(feature_pairs)

        actual_matches = compute_tab_matches(pieces, raft)
        expected_matches = {(a,b) if a<b else (b,a) for a,b in feature_pairs}
        # we expect the actual matches to be a subset of the expected
        # matches, if there are any "extra" matches in the actuals
        # that means something went wrong
        if False and not (actual_matches <= expected_matches):
            e1 = FFP(expected_matches)
            a1 = FFP(actual_matches)
            logger.warn(f"actual matches not proper subset of expected matches\n       {quad=}\n      desc={desc} {mse=:.3f}\n  expected={e1}\n    actual={a1}")
        
        retval.append(quad | {'raft':desc, 'mse':mse, 'rank':None,
                              'expected_matches':FFP(expected_matches),
                              'actual_matches':FFP(actual_matches),
                              'expected_only':FFP(expected_matches - actual_matches),
                              'actual_only':FFP(actual_matches - expected_matches)})

    retval.sort(key=operator.itemgetter('mse'))
    for i, row in enumerate(retval, start=1):
        row['rank'] = i
        row['mse'] = decimal.Decimal(f"{row['mse']:.3f}")

    return retval

def quadrophenia_worker(puzzle_path, src_q, dst_q):
    
    puzzle = puzzler.file.load(puzzle_path)
    
    pieces = {p.label: p for p in puzzle.pieces}

    job = src_q.get()
    while job:

        rows = quadmaster(pieces, job)

        dst_q.put(rows)
        job = src_q.get()

    return
        
def quads_main(args):

    puzzle_path = args.puzzle
    output_csv_path = args.output
    
    puzzle = puzzler.file.load(puzzle_path)

    pieces = {p.label: p for p in puzzle.pieces}

    assert len(pieces) in (1000, 1026)

    num_rows, num_cols, grid = griddy(pieces)

    quads = []
    for r in range(num_rows-1):
        for c in range(num_cols-1):
            quad = {'row_no': r,
                    'col_no': c,
                    'ul_piece': grid[r+0,c+0],
                    'ur_piece': grid[r+0,c+1],
                    'll_piece': grid[r+1,c+0],
                    'lr_piece': grid[r+1,c+1]}
            quads.append(quad)
    print(len(quads), "quads")

    for label, piece in pieces.items():
        try:
            puzzler.pocket.tab_features_for_piece(piece, be_forgiving=False)
        except puzzler.pocket.PocketFitter.FitException as x:
            print(f"{label} has problem with tab_features_for_piece!", x)

    f = open(output_csv_path, 'w', newline='')
    writer = csv.DictWriter(f, fieldnames='col_no row_no ul_piece ur_piece ll_piece lr_piece raft mse rank expected_matches actual_matches expected_only actual_only'.split())
    writer.writeheader()

    if args.num_workers:
        src_q = mp.Queue()
        dst_q = mp.Queue()

        workers = [mp.Process(target=quadrophenia_worker, args=(puzzle_path, src_q, dst_q)) for _ in range(args.num_workers)]
        for p in workers:
            p.start()

        for q in quads:
            src_q.put(q)
                
        num_jobs = len(quads)
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

        for quad in tqdm(quads, smoothing=0):
            writer.writerows(quadmaster(pieces, quad))

def add_parser(commands):

    parser_quads = commands.add_parser("quads", help="process quads to compute expected tab matches")
    parser_quads.add_argument("-o", "--output", help="output CSV path for computed quads", required=True)
    parser_quads.add_argument("-n", "--num-workers", default=0, type=int,
                              help="number of workers (default %(default)i)")
    parser_quads.set_defaults(func=quads_main)
