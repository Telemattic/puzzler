import collections
import copy
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

Feature = puzzler.raft.Feature

def try_triples(quad, *, pieces, num_refine=1, tab_pairs=None, early_exit=False):

    raftinator = puzzler.raft.Raftinator(pieces)
    
    def remove_piece_from_raft(raft, label):
        # make a deep copy so that coords in the cloned raft are
        # distinct objects from the original raft coordinates
        coords = copy.deepcopy(raft.coords)
        coords.pop(label)
        new_raft = puzzler.raft.Raft(coords, None)
        return raftinator.refine_alignment_within_raft(new_raft)

    good_raft = raftinator.make_raft_from_string(quad['raft'])

    retval2 = []

    quad_no = quad['quad_no']

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

        pf = puzzler.pocket.PocketFitter(raftinator, triple_raft, pocket, num_refine)

        candidates = set(pieces.keys()) - set(triple_raft.coords.keys())

        min_seam_error = 10000.

        for match in pf.candidate_matches(candidates, tab_pairs):

            if early_exit and match.min_seam_error is not None and match.min_seam_error > min_seam_error:
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
                   'mse': mse,
                   'seam_mse': seam_fit_error.mse,
                   'lower_bound_mse': match.min_seam_error,
                   'rank': None}

            retval.append(row)

        retval.sort(key=operator.itemgetter('mse'))
        for i, row in enumerate(retval, start=1):
            row['rank'] = i
            for k in 'mse', 'seam_mse', 'lower_bound_mse':
                if row[k] is not None:
                    row[k] = decimal.Decimal(f"{row[k]:.3f}")
                   
        retval2 += retval
        quad_no += 1

    return retval2

TRIPLES_KWARGS = None

def triples_init(puzzle_path, num_refine, tabs_path, early_exit):

    global TRIPLES_KWARGS

    puzzle = puzzler.file.load(puzzle_path)
    pieces = {p.label: p for p in puzzle.pieces}
    
    kwargs = {'pieces':pieces, 'num_refine':num_refine, 'tab_pairs':None, 'early_exit':early_exit}
    if tabs_path:
        kwargs['tab_pairs'] = puzzler.tabpairs.load_tab_pairs(tabs_path)

    TRIPLES_KWARGS = kwargs

def triples_work(quad):

    return try_triples(quad, **TRIPLES_KWARGS)

def triples_main(args):

    puzzle = puzzler.file.load(args.puzzle)
    
    pieces = {p.label: p for p in puzzle.pieces}
    
    assert len(pieces) == 1026

    quads = []
    with open(args.quads, 'r', newline='') as ifile:
        reader = csv.DictReader(ifile)
        for row in reader:
            if int(row['rank']) == 1:
                # 4x because we'll really do 4 separately ranked experiments per quad.  Gross.
                row['quad_no'] = 4 * len(quads)
                quads.append(row)

    with open(args.output, 'w', newline='') as ofile:
        fieldnames = 'quad_no ul_piece ur_piece ll_piece lr_piece drop_piece fit_piece raft mse seam_mse lower_bound_mse rank'
        writer = csv.DictWriter(ofile, fieldnames=fieldnames.split())
        writer.writeheader()

        if args.num_workers:
            
            with mp.Pool(args.num_workers,
                         triples_init,
                         [args.puzzle, args.refine, args.tabs, args.early_exit],
                         maxtasksperchild=None) as pool:
                
                pbar = tqdm(total=len(quads), smoothing=0)
                for result in pool.imap_unordered(triples_work, quads):
                    writer.writerows(result)
                    pbar.update()
        else:

            kwargs = {'pieces':pieces, 'num_refine':args.refine, 'tab_pairs':None, 'early_exit':args.early_exit}
            if args.tabs:
                kwargs['tab_pairs'] = puzzler.tabpairs.load_tab_pairs(args.tabs)
            
            for quad in tqdm(quads, smoothing=0):
                writer.writerows(try_triples(quad, **kwargs))

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
    parser_triples.add_argument("--early-exit", default=False, action='store_true',
                                help="stop the search if the tab-pairs says no better match is possible")
    parser_triples.set_defaults(func=triples_main)
    
