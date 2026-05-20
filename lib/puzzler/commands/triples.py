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

        pf = puzzler.pocket.PocketFitter(pieces, triple_raft, pocket, num_refine)

        candidates = set(pieces.keys()) - set(triple_raft.coords.keys())

        min_seam_error = 10000.

        for match in pf.candidate_matches(candidates, fit_error_for_tab_pairs):

            if match.min_seam_error > min_seam_error:
                break

            mse, seam_fit_error = pf.measure_fit(match.src_label, match.feature_pairs, compute_seam_fit_error=True)
            if min_seam_error > seam_fit_error.mse:
                min_seam_error = seam_fit_error.mse

            desc = raftinator.format_feature_pairs(
                triple_feature_pairs + match.feature_pairs)

            row = {'quad_no': quad_no,
                   'ul_piece': quad['ul_piece'],
                   'ur_piece': quad['ur_piece'],
                   'll_piece': quad['ll_piece'],
                   'lr_piece': quad['lr_piece'],
                   'drop_piece': drop_label,
                   'fit_piece': match.src_label,
                   'raft': desc,
                   'mse': tuple(mse),
                   'seam_mse': seam_fit_error.mse,
                   'lower_bound_mse': match.min_seam_error,
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
    
