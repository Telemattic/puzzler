import puzzler
import itertools
import math
import numpy as np
import operator
import scipy.spatial
from typing import NamedTuple, Optional, Set

Feature = puzzler.raft.Feature

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

def tab_features_for_piece(piece, be_forgiving=True):

    ref_angle = None
    dirs = [None, None, None, None]
    n_collisions = 4

    for tab_no, tab in enumerate(piece.tabs):

        x, y = tab.ellipse.center
        angle = math.atan2(y, x)

        if ref_angle is None:
            ref_angle = angle
            d = 0
        else:
            d = axis_for_angle(angle - ref_angle)

        if dirs[d] is not None:

            # HACK, see piece L16 from 300.json
            if be_forgiving and len(piece.tabs) == 4:
                n_collisions += 1
                continue

            raise PocketFitter.FitException(f"{piece.label} has multiple tabs at {d=} and {len(piece.tabs)} total tabs")

        dirs[d] = (piece.label, tab_no, tab.indent)

    # HACK: if we couldn't figure out cardinal directions for a
    # piece with 4 tabs just assign them sequentially *but* note
    # that we actually walk the tabs in reverse order ordinarily
    # because of the angle computation.  Blech.
    if be_forgiving and n_collisions and len(piece.tabs) == 4:
        for tab_no, tab in enumerate(piece.tabs):
            i = (4 - tab_no) % 4
            dirs[i] = (piece.label, tab_no, tab.indent)

    retval = []
    for i in range(4):
        a = dirs[i-1]
        b = dirs[i]
        if a is not None or b is not None:
            retval.append((a, b))

    return retval

def find_unmatched_tabs(pieces, coords):

    tab_features = []
    tab_coords = []
    for l, c in coords.items():
        p = pieces[l]
        for tab_no, tab in enumerate(p.tabs):
            tab_features.append(Feature(l,'tab',tab_no))
            tab_coords.append(c.xform.apply_v2(tab.ellipse.center))

    covered = set()
    
    kdtree = scipy.spatial.KDTree(np.vstack(tab_coords))
    for a, b in kdtree.query_pairs(100.):
        covered.add(a)
        covered.add(b)

    unmatched = set()
    overlaps = puzzler.solver.OverlappingPieces(pieces, coords)

    for i, f in enumerate(tab_features):
        
        if i in covered:
            continue
    
        is_unmatched = True
        
        p = pieces[f.piece]
        t = p.tabs[f.index]
        for n in overlaps(tab_coords[i], 0): #t.ellipse.semi_major):
            if n == f.piece:
                continue
            print(f"Ignoring unmatched tab {f!s} because it overlaps {n}")
            is_unmatched = False

        if is_unmatched:
            unmatched.add(f)

    return unmatched

class PocketTabMatcher:

    class Match(NamedTuple):
        src_label: str
        feature_pairs: puzzler.raft.FeaturePairs
        min_seam_error: Optional[float]

    def __init__(self, pieces, pocket):
        self.pieces = pieces
        self.pocket = pocket
        
        dst_tab_a = None
        if pocket.tab_a is not None:
            t = pocket.tab_a
            dst_tab_a = (t.piece, t.index, self.pieces[t.piece].tabs[t.index].indent)

        dst_tab_b = None
        if pocket.tab_b is not None:
            t = pocket.tab_b
            dst_tab_b = (t.piece, t.index, self.pieces[t.piece].tabs[t.index].indent)

        self.dst_tab_pair = (dst_tab_a, dst_tab_b)

    def candidate_matches_for_piece(self, src_label, tab_pairs=None):

        def lower_bound_error(feature_pairs):

            err = puzzler.raft.FitError(0.,0)
            for fp in feature_pairs:
                err += tab_pairs.get_fit_error(*fp)
            return err.mse

        retval = []
        for src_tab_pair in tab_features_for_piece(self.pieces[src_label]):

            feature_pairs = PocketTabMatcher.make_feature_pairs(self.dst_tab_pair, src_tab_pair)
            if len(feature_pairs):
                mse = lower_bound_error(feature_pairs) if tab_pairs else None
                retval.append(PocketTabMatcher.Match(src_label, feature_pairs, mse))
                
        return retval
        
    def candidate_matches(self, candidates, tab_pairs=None):

        retval = []
        for src_label in candidates:
            try:
                retval += self.candidate_matches_for_piece(src_label, tab_pairs)
            except PocketFitter.FitException as x:
                print(x)

        if tab_pairs:
            retval.sort(key=operator.attrgetter('min_seam_error'))
        
        return retval

    @staticmethod
    def make_feature_pairs(dst_tab_pair, src_tab_pair):

        retval = []
        try:
            fp = PocketTabMatcher.make_feature_pair(dst_tab_pair[0], src_tab_pair[1])
            if fp is not None:
                retval.append(fp)
            fp = PocketTabMatcher.make_feature_pair(dst_tab_pair[1], src_tab_pair[0])
            if fp is not None:
                retval.append(fp)
            
        except ValueError:
            retval = []

        return retval

    @staticmethod
    def make_feature_pair(a, b):
        if a is None and b is None:
            return None
        if a is None or b is None:
            raise ValueError("tab can't match missing tab")
        if a[2] == b[2]:
            raise ValueError("tab conflict")
        return (Feature(a[0],'tab',a[1]), Feature(b[0],'tab',b[1]))

class Pocket(NamedTuple):
    tab_a: Feature
    tab_b: Feature
    pieces: Set[str]

    def tab_dsts(self):
        s = set()
        if self.tab_a is not None:
            s.add(self.tab_a.piece)
        if self.tab_b is not None:
            s.add(self.tab_b.piece)
        return tuple(s)

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

        pockets = []
        for tab in find_unmatched_tabs(self.pieces, self.raft.coords):
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

    class FitException(Exception):
        pass

    def __init__(self, raftinator, dst_raft, pocket, num_refine):
        self.raftinator = raftinator
        self.pieces = raftinator.pieces
        self.num_refine = num_refine
        self.pocket = pocket

        pocket_coords = {i: dst_raft.coords[i].copy() for i in pocket.pieces}
        pocket_raft = puzzler.raft.Raft(pocket_coords, None)
        self.dst_raft = self.raftinator.refine_alignment_within_raft(pocket_raft)

        self.tab_matcher = PocketTabMatcher(self.pieces, pocket)

    def candidate_matches(self, candidates, tab_pairs=None):
        return self.tab_matcher.candidate_matches(candidates, tab_pairs)

    def measure_fit(self, src_label, feature_pairs, compute_seam_fit_error=False):

        r = self.raftinator

        src_raft = r.factory.make_raft_for_piece(src_label)

        src_coord = r.aligner.rough_align(self.dst_raft, src_raft, feature_pairs)

        try:
            src_coord = r.aligner.refine_alignment_between_rafts(
                self.dst_raft, src_raft, src_coord)
        except puzzler.raft.RaftAligner.AlignException as x:
            print(f"PocketFitter.measure_fit: {r.format_feature_pairs(feature_pairs)} has no seams for alignment!", x)
            return (float("+inf"), puzzler.raft.FitError(0.,0.))

        if compute_seam_fit_error:
            seams = r.seamstress.seams_between_rafts(self.dst_raft, src_raft, src_coord)

            good_dsts = {i[0] for i in self.tab_matcher.dst_tab_pair if i is not None}
            seam_fe = r.seamstress.fit_error_for_seams([s for s in seams if s.dst.piece in good_dsts])
        else:
            seam_fe = None

        raft = r.factory.merge_rafts(self.dst_raft, src_raft, src_coord)

        seams = r.get_seams_for_raft(raft)
        for _ in range(self.num_refine):
            raft = r.refine_alignment_within_raft(raft, seams)
            seams = r.get_seams_for_raft(raft)
            
        mse = r.get_total_error_for_raft_and_seams(raft, seams)

        if mse is None:
            mse = float("+inf")
        
        return (mse, seam_fe)
