import collections
import csv
import decimal
import itertools
import operator
import math
import multiprocessing as mp
import numpy as np
import puzzler
from typing import Any, Iterable, Mapping, NamedTuple, Optional, Sequence, Set, Tuple

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

    ul = quad['ul_piece']
    ur = quad['ur_piece']
    ll = quad['ll_piece']
    lr = quad['lr_piece']

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
            tab_pairs_for_piece(piece)
        except AssertionError:
            print(f"{label} has problem with tab_pairs_for_piece!")

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

class Pocket(NamedTuple):
    tab_a: Feature
    tab_b: Feature
    pieces: Set[str]

    def __str__(self):
        s = '{' + ', '.join(self.pieces) + '}'
        return f"Pocket({self.tab_a!s}, {self.tab_b!s}, {s})"

class PocketFinder:

    class Helper:
        def __init__(self, pieces, coords):
            self.pieces = pieces
            self.coords = coords
            self.helper = puzzler.raft.FeatureHelper(pieces, coords)
            self.overlaps = puzzler.solver.OverlappingPieces(pieces, coords)

        def get_tab_unit_vector(self, f):
            p0 = self.coords[f.piece].xy
            p1 = self.helper.get_tab_center(f)
            return puzzler.math.unit_vector(p1 - p0)

        def find_tab_in_direction(self, p, v0):
            tabs = self.pieces[p].tabs
            if len(tabs) == 0:
                return None
            
            tab_vecs = np.array([tab.ellipse.center for tab in tabs])
            tab_vecs = tab_vecs / np.linalg.norm(tab_vecs, axis=1, keepdims=True)
            tab_vecs = self.coords[p].xform.apply_n2(tab_vecs)
            
            dp = np.sum(tab_vecs * v0, axis=1)

            i = np.argmax(dp)
            return int(i) if dp[i] > .7 else None

        def get_neighbor(self, label, vec):

            coord = self.coords[label]
            piece = self.pieces[label]
            for neighbor in self.overlaps(coord.xy, piece.radius):
                if neighbor == label:
                    continue
                neighbor_coord = self.coords[neighbor]
                nvec = puzzler.math.unit_vector(neighbor_coord.xy - coord.xy)
                if np.sum(nvec * vec) > .95:
                    return str(neighbor)

            return None

    def __init__(self, pieces, raft):
        self.pieces = pieces
        self.raft = raft
        self.helper = PocketFinder.Helper(pieces, raft.coords)

    def find_pockets_on_frontiers(self, frontiers = None):

        if frontiers is None:
            fc = puzzler.raft.RaftFeaturesComputer(self.pieces)
            frontiers = fc.compute_frontiers(self.raft.coords)

        pockets = []
        for tab in self.get_tabs_on_frontiers(frontiers):
            pockets += self.get_pockets_for_tab(tab)

        return set(pockets)

    def get_pockets_for_tab(self, tab):

        helper = self.helper

        vt = helper.get_tab_unit_vector(tab)
        vl = np.array((vt[1], -vt[0]))
        vr = -vl

        pockets = []

        if nl := helper.get_neighbor(tab.piece, vl):
            if nll := helper.get_neighbor(nl, vt):
                tabl = helper.find_tab_in_direction(nll, -vl)
                if tabl is not None:
                    tabl = Feature(nll, 'tab', tabl)
                pockets.append(Pocket(tabl, tab, frozenset([tab.piece, nl, nll])))
            
        if nr := helper.get_neighbor(tab.piece, vr):
            if nrr := helper.get_neighbor(nr, vt):
                tabr = helper.find_tab_in_direction(nrr, -vr)
                if tabr is not None:
                    tabr = Feature(nrr, 'tab', tabr)
                pockets.append(Pocket(tab, tabr, frozenset([tab.piece, nr, nrr])))

        return pockets

    @staticmethod
    def get_tabs_on_frontiers(frontiers):

        return set([i for i in itertools.chain(*frontiers) if i.kind == 'tab'])

class PocketFitter:

    def __init__(self, pieces, dst_raft, pocket, num_refine):
        self.pieces = pieces
        self.raftinator = puzzler.raft.Raftinator(pieces)
        self.num_refine = num_refine

        pocket_coords = {i: dst_raft.coords[i].copy() for i in pocket.pieces}
        pocket_raft = puzzler.raft.Raft(pocket_coords, None)
        self.dst_raft = self.raftinator.refine_alignment_within_raft(pocket_raft)

        dst_tab_a = None
        if pocket.tab_a is not None:
            t = pocket.tab_a
            dst_tab_a = (t.piece, t.index, self.pieces[t.piece].tabs[t.index].indent)

        dst_tab_b = None
        if pocket.tab_b is not None:
            t = pocket.tab_b
            dst_tab_b = (t.piece, t.index, self.pieces[t.piece].tabs[t.index].indent)

        self.dst_tab_pair = (dst_tab_a, dst_tab_b)

    def measure_fit(self, src_label):

        if src_label in self.dst_raft.coords:
            return []

        src_raft = self.raftinator.factory.make_raft_for_piece(src_label)

        retval = []
        for src_tab_pair in self.tab_pairs_for_piece(src_label):

            feature_pairs = self.make_feature_pairs(src_tab_pair)

            if len(feature_pairs) < 1:
                continue

            new_raft = self.raftinator.align_and_merge_rafts_with_feature_pairs(
                self.dst_raft, src_raft, feature_pairs)

            mse = self.compute_mse_with_refinement(new_raft)
            
            retval.append((mse, feature_pairs))

        return retval

    def compute_mse_with_refinement(self, raft):

        mse = []

        r = self.raftinator

        seams = r.get_seams_for_raft(raft)
        mse.append(r.get_total_error_for_raft_and_seams(raft, seams))
                
        for _ in range(self.num_refine):
            raft = r.refine_alignment_within_raft(raft, seams)
            
            seams = r.get_seams_for_raft(raft)
            mse.append(r.get_total_error_for_raft_and_seams(raft, seams))

        return mse

    def tab_pairs_for_piece(self, label):

        piece = self.pieces[label]

        ref_angle = None
        dirs = [None, None, None, None]

        for tab_no, tab in enumerate(piece.tabs):

            x, y = tab.ellipse.center
            angle = math.atan2(y, x)
            
            if ref_angle is None:
                ref_angle = angle
                d = 0
            else:
                d = axis_for_angle(angle - ref_angle)

            # HACK, see piece L16 from 300.json
            if dirs[d] is not None:
                assert label in {'L16'}
                continue
             
            assert dirs[d] is None
            dirs[d] = (piece.label, tab_no, tab.indent)

        retval = []
        for i in range(4):
            a = dirs[i-1]
            b = dirs[i]
            if a is not None or b is not None:
                retval.append((a, b))

        return retval

    def make_feature_pairs(self, src_tab_pair):

        retval = []
        try:
            fp = self.make_feature_pair(self.dst_tab_pair[0], src_tab_pair[1])
            if fp is not None:
                retval.append(fp)
            fp = self.make_feature_pair(self.dst_tab_pair[1], src_tab_pair[0])
            if fp is not None:
                retval.append(fp)
            
        except ValueError:
            retval = []

        return retval

    @staticmethod
    def make_feature_pair(a, b):
        if a is None or b is None:
            return None
        if a[2] == b[2]:
            raise ValueError("tab conflict")
        return (Feature(a[0],'tab',a[1]), Feature(b[0],'tab',b[1]))

def try_triples(pieces, quad, num_refine):

    raftinator = puzzler.raft.Raftinator(pieces)
    
    def remove_piece_from_raft(raft, label):
        coords = raft.coords.copy()
        coords.pop(label)
        new_raft = puzzler.raft.Raft(coords, None)
        return raftinator.refine_alignment_within_raft(new_raft)

    def make_feature_pair(a, b):
        if a is None or b is None:
            return None
        if a[2] == b[2]:
            raise ValueError("tab conflict")
        return (Feature(a[0],'tab',a[1]), Feature(b[0],'tab',b[1]))

    def make_feature_pairs(dst_tab_pair, src_tab_pair):
        
        retval = []
        try:
            fp = make_feature_pair(dst_tab_pair[0], src_tab_pair[1])
            if fp is not None:
                retval.append(fp)
            fp = make_feature_pair(dst_tab_pair[1], src_tab_pair[0])
            if fp is not None:
                retval.append(fp)
            
        except ValueError:
            retval = []

        return retval

    def make_tab_pair(a, b):

        tab_a = (a.piece, a.index, pieces[a.piece].tabs[a.index].indent) if a is not None else None
        tab_b = (b.piece, b.index, pieces[b.piece].tabs[b.index].indent) if b is not None else None
        return (tab_a, tab_b)

    good_raft = raftinator.make_raft_from_string(quad['raft'])

    retval2 = []

    quad_no = quad['quad_no']

    for drop_quadrant in 'ul_piece', 'ur_piece', 'll_piece', 'lr_piece':

        drop_label = quad[drop_quadrant]

        triple_raft = remove_piece_from_raft(good_raft, drop_label)

        pockets = PocketFinder(pieces, triple_raft).find_pockets_on_frontiers()
        assert len(pockets) == 1

        pocket = pockets.pop()

        assert pocket.pieces == frozenset(triple_raft.coords.keys())

        fit_tab_pair = make_tab_pair(pocket.tab_a, pocket.tab_b)

        triple_feature_pairs = []
        for a, b in raftinator.parse_feature_pairs(quad['raft']):
            if a.piece != drop_label and b.piece != drop_label:
                triple_feature_pairs.append((a,b))

        retval = []

        use_legacy_algorithm = False

        if use_legacy_algorithm:
            
            for try_label, try_piece in pieces.items():

                if try_label in triple_raft.coords:
                    continue
    
                try_raft = raftinator.factory.make_raft_for_piece(try_label)
    
                for try_tab_pair in tab_pairs_for_piece(try_piece):

                    try_feature_pairs = make_feature_pairs(fit_tab_pair, try_tab_pair)

                    if len(try_feature_pairs) < 1:
                        continue
    
                    new_raft = raftinator.align_and_merge_rafts_with_feature_pairs(
                        triple_raft, try_raft, try_feature_pairs)
    
                    mse = []
    
                    new_seams = raftinator.get_seams_for_raft(new_raft)
                    mse.append(raftinator.get_total_error_for_raft_and_seams(new_raft, new_seams))
                    
                    for _ in range(num_refine):
                        new_raft = raftinator.refine_alignment_within_raft(new_raft, new_seams)
                               
                        new_seams = raftinator.get_seams_for_raft(new_raft)
                        mse.append(raftinator.get_total_error_for_raft_and_seams(new_raft, new_seams))
    
                    desc = raftinator.format_feature_pairs(triple_feature_pairs + try_feature_pairs)

                    row = {'quad_no': quad_no,
                           'ul_piece': quad['ul_piece'],
                           'ur_piece': quad['ur_piece'],
                           'll_piece': quad['ll_piece'],
                           'lr_piece': quad['lr_piece'],
                           'drop_piece': drop_label,
                           'fit_piece': try_label,
                           'raft': desc,
                           'mse': tuple(mse),
                           'rank': None}
                    
                    retval.append(row)

        else:
                    
            pocket_fitter = PocketFitter(pieces, good_raft, pocket, num_refine)

            for fit_label in pieces:
    
                for mse, try_feature_pairs in pocket_fitter.measure_fit(fit_label):
    
                    desc = raftinator.format_feature_pairs(
                        triple_feature_pairs + try_feature_pairs)
                    
                    row = {'quad_no': quad_no,
                           'ul_piece': quad['ul_piece'],
                           'ur_piece': quad['ur_piece'],
                           'll_piece': quad['ll_piece'],
                           'lr_piece': quad['lr_piece'],
                           'drop_piece': drop_label,
                           'fit_piece': fit_label,
                           'raft': desc,
                           'mse': tuple(mse),
                           'rank': None}
                    
                    retval.append(row)

        retval.sort(key=lambda x: x['mse'][-1])
        for i, row in enumerate(retval, start=1):
            row['rank'] = i
            row['mse'] = ','.join(f"{i:.3f}" for i in row['mse'])

        retval2 += retval
        quad_no += 1

    return retval2

TRIPLES_PUZZLE = None
TRIPLES_REFINE = None

def triples_init(puzzle_path, num_refine):

    global TRIPLES_PUZZLE, TRIPLES_REFINE

    TRIPLES_PUZZLE = puzzler.file.load(puzzle_path)
    TRIPLES_REFINE = num_refine

def triples_work(quad):

    pieces = {p.label: p for p in TRIPLES_PUZZLE.pieces}
    return try_triples(pieces, quad, TRIPLES_REFINE)

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
        fieldnames = 'quad_no ul_piece ur_piece ll_piece lr_piece drop_piece fit_piece raft mse rank'
        writer = csv.DictWriter(ofile, fieldnames=fieldnames.split())
        writer.writeheader()

        if args.num_workers:
            
            with mp.Pool(args.num_workers,
                         triples_init,
                         [puzzle_path, args.refine],
                         maxtasksperchild=None) as pool:
                
                pbar = tqdm(total=len(quads), smoothing=0)
                for result in pool.imap_unordered(triples_work, quads):
                    writer.writerows(result)
                    pbar.update()
        else:

            for quad in tqdm(quads, smoothing=0):
                writer.writerows(try_triples(pieces, quad, args.refine))

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
    parser_triples.set_defaults(func=triples_main)
    
