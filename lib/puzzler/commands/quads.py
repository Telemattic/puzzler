import collections
import csv
import decimal
import itertools
import operator
import math
import multiprocessing as mp
import numpy as np
import puzzler
from typing import Any, Iterable, Mapping, Optional, Sequence, Set, Tuple

from tqdm import tqdm

HACK = False

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

    ul = quad['ul_piece']
    ur = quad['ur_piece']
    ll = quad['ll_piece']
    lr = quad['lr_piece']

    tab_features_for_piece = puzzler.pocket.tab_features_for_piece

    ul_tabs = tab_features_for_piece(pieces[ul])
    ur_tabs = tab_features_for_piece(pieces[ur])
    ll_tabs = tab_features_for_piece(pieces[ll])
    lr_tabs = tab_features_for_piece(pieces[lr])

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
        
        retval.append(quad | {'raft':desc, 'mse':mse, 'rank':None})

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

    assert len(pieces) == 1026

    rows = [chr(ord('A')+i) for i in range(26)] + ['AA']
    cols = [str(i) for i in range(1,39)]
    # print(f"{rows=} {cols=} n_pieces={len(rows)*len(cols)}")

    quads = []
    for row_no in range(len(rows)-1):
        for col_no in range(len(cols)-1):
            row0, row1 = rows[row_no], rows[row_no+1]
            col0, col1 = cols[col_no], cols[col_no+1]
            quad = {'row_no': row_no,
                    'col_no': col_no,
                    'ul_piece': row0+col0,
                    'ur_piece': row0+col1,
                    'll_piece': row1+col0,
                    'lr_piece': row1+col1}
            quads.append(quad)

    for label, piece in pieces.items():
        try:
            puzzler.pocket.tab_features_for_piece(piece, be_forgiving=False)
        except puzzler.pocket.FitException:
            print(f"{label} has problem with tab_features_for_piece!")

    f = open(output_csv_path, 'w', newline='')
    writer = csv.DictWriter(f, fieldnames='col_no row_no ul_piece ur_piece ll_piece lr_piece raft mse rank'.split())
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

Feature = puzzler.raft.Feature

def try_triples(pieces, quad, num_refine, fit_error_for_tab_pairs = None):

    raftinator = puzzler.raft.Raftinator(pieces)
    
    def remove_piece_from_raft(raft, label):
        coords = raft.coords.copy()
        coords.pop(label)
        new_raft = puzzler.raft.Raft(coords, None)
        return raftinator.refine_alignment_within_raft(new_raft)

    good_raft = raftinator.make_raft_from_string(quad['raft'])

    retval2 = []

    quad_no = quad['quad_no']

    if HACK and quad_no != 8:
        return []

    for drop_quadrant in 'ul_piece', 'ur_piece', 'll_piece', 'lr_piece':

        drop_label = quad[drop_quadrant]

        triple_raft = remove_piece_from_raft(good_raft, drop_label)

        pockets = puzzler.pocket.PocketFinder(pieces, triple_raft).find_pockets_on_frontiers()
        assert len(pockets) == 1

        pocket = pockets.pop()

        assert pocket.pieces == frozenset(triple_raft.coords.keys())

        triple_feature_pairs = []
        for a, b in raftinator.parse_feature_pairs(quad['raft']):
            if a.piece != drop_label and b.piece != drop_label:
                triple_feature_pairs.append((a,b))

        retval = []

        pocket_fitter = puzzler.pocket.PocketFitter(pieces, triple_raft, pocket, num_refine)

        for fit_label in pieces:

            for mse, try_feature_pairs, seam_fit_error in pocket_fitter.measure_fit(fit_label, compute_seam_fit_error=True):

                desc = raftinator.format_feature_pairs(
                    triple_feature_pairs + try_feature_pairs)

                if fit_error_for_tab_pairs:
                    fit_error = puzzler.raft.FitError(0.,0)
                    for i in try_feature_pairs:
                        fit_error = fit_error + fit_error_for_tab_pairs[i]
                    lower_bound_mse = fit_error.mse
                else:
                    lower_bound_mse = None

                seam_mse = seam_fit_error.mse if seam_fit_error else None
                
                row = {'quad_no': quad_no,
                       'ul_piece': quad['ul_piece'],
                       'ur_piece': quad['ur_piece'],
                       'll_piece': quad['ll_piece'],
                       'lr_piece': quad['lr_piece'],
                       'drop_piece': drop_label,
                       'fit_piece': fit_label,
                       'raft': desc,
                       'mse': tuple(mse),
                       'seam_mse': seam_mse,
                       'lower_bound_mse': lower_bound_mse,
                       'rank': None}

                retval.append(row)

        retval.sort(key=lambda x: x['mse'][-1])
        for i, row in enumerate(retval, start=1):
            row['rank'] = i
            row['mse'] = ','.join(f"{i:.3f}" for i in row['mse'])
            for k in 'seam_mse', 'lower_bound_mse':
                if row[k] is not None:
                    row[k] = decimal.Decimal(f"{row[k]:.3f}")
                   

        retval2 += retval
        quad_no += 1

        if HACK:
            break

    return retval2

TRIPLES_PUZZLE = None
TRIPLES_REFINE = None
TRIPLES_FIT_ERROR = None

def triples_init(puzzle_path, num_refine, tabs_path):

    global TRIPLES_PUZZLE, TRIPLES_REFINE, TRIPLES_FIT_ERROR

    TRIPLES_PUZZLE = puzzler.file.load(puzzle_path)
    TRIPLES_REFINE = num_refine
    TRIPLES_FIT_ERROR = load_fit_error_for_tab_pairs(tabs_path)

def triples_work(quad):

    pieces = {p.label: p for p in TRIPLES_PUZZLE.pieces}
    return try_triples(pieces, quad, TRIPLES_REFINE, TRIPLES_FIT_ERROR)

def load_fit_error_for_tab_pairs(path):

    retval = dict()
    with open(path, 'r', newline='') as ifile:
        reader = csv.DictReader(ifile)

        for row in reader:
            dst = Feature(row['dst_label'], 'tab', int(row['dst_tab_no']))
            src = Feature(row['src_label'], 'tab', int(row['src_tab_no']))
            fit_error = puzzler.raft.FitError(float(row['fwd_sse']), int(row['fwd_n']))
            retval[dst,src] = fit_error

    return retval

def triples_main(args):

    puzzle_path = args.puzzle
    input_csv_path = args.quads
    output_csv_path = args.output

    puzzle = puzzler.file.load(puzzle_path)
    
    pieces = {p.label: p for p in puzzle.pieces}
    
    assert len(pieces) == 1026

    quads = []
    with open(input_csv_path, 'r', newline='') as ifile:
        reader = csv.DictReader(ifile)
        for row in reader:
            if int(row['rank']) == 1:
                # 4x because we'll really do 4 separately ranked experiments per quad.  Gross.
                row['quad_no'] = 4 * len(quads)
                quads.append(row)

    with open(output_csv_path, 'w', newline='') as ofile:
        fieldnames = 'quad_no ul_piece ur_piece ll_piece lr_piece drop_piece fit_piece raft mse seam_mse lower_bound_mse rank'
        writer = csv.DictWriter(ofile, fieldnames=fieldnames.split())
        writer.writeheader()

        if args.num_workers:
            
            with mp.Pool(args.num_workers,
                         triples_init,
                         [puzzle_path, args.refine, args.tabs],
                         maxtasksperchild=None) as pool:
                
                pbar = tqdm(total=len(quads), smoothing=0)
                for result in pool.imap_unordered(triples_work, quads):
                    writer.writerows(result)
                    pbar.update()
        else:

            fit_error_for_tab_pairs = load_fit_error_for_tab_pairs(args.tabs) if args.tabs else None

            for quad in tqdm(quads, smoothing=0):
                writer.writerows(try_triples(pieces, quad, args.refine, fit_error_for_tab_pairs))

def add_parser(commands):

    parser_quads = commands.add_parser("quads", help="process quads to compute expected tab matches")
    parser_quads.add_argument("-o", "--output", help="output CSV path for computed quads", required=True)
    parser_quads.add_argument("-n", "--num-workers", default=0, type=int,
                              help="number of workers (default %(default)i)")
    parser_quads.set_defaults(func=quads_main)

    parser_triples = commands.add_parser("triples", help="process triples to validate pocket (interior corner) matches")
    parser_triples.add_argument("-q", "--quads", required=True,
                                help="inputs quads path")
    parser_triples.add_argument("-o", "--output", required=True,
                                help="output CSV path for triples")
    parser_triples.add_argument("-n", "--num-workers", default=0, type=int,
                                help="number of workers (default %(default)i)")
    parser_triples.add_argument("-r", "--refine", default=1, type=int,
                                help="number of refinement passes for fit (default %(default)i)")
    parser_triples.add_argument("--tabs", default=None, help="CSV of fit error for all possible tab matches")
    parser_triples.set_defaults(func=triples_main)
    
