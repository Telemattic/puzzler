import puzzler
import itertools
import math
import numpy as np
from typing import NamedTuple, Set

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

            raise PocketFitter.FitException(f"{label} has multiple tabs at {d=}")

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

class PocketTabMatcher:

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

    def possible_matches(self, src_label):

        retval = []
        for src_tab_pair in tab_features_for_piece(self.pieces[src_label]):

            feature_pairs = PocketTabMatcher.make_feature_pairs(self.dst_tab_pair, src_tab_pair)
            if len(feature_pairs):
                retval.append(feature_pairs)
        return retval

    def possible_matches_ordered_by_lower_bound_error(self, candidates, fit_error_for_tabs):

        def lower_bound_error(registration):

            err = puzzler.raft.FitError(0.,0)
            for fp in registration:
                err += fit_error_for_tabs[fp]
            return err.mse

        retval = []
        for src_label in candidates:
            registrations = self.possible_matches(src_label)
            for reg in registrations:
                mse = lower_bound_error(reg)
                retval.append((mse, src_label, reg))
        retval.sort()
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

    class FitException(Exception):
        pass

    def __init__(self, pieces, dst_raft, pocket, num_refine):
        self.pieces = pieces
        self.raftinator = puzzler.raft.Raftinator(pieces)
        self.num_refine = num_refine
        self.pocket = pocket

        pocket_coords = {i: dst_raft.coords[i].copy() for i in pocket.pieces}
        pocket_raft = puzzler.raft.Raft(pocket_coords, None)
        self.dst_raft = self.raftinator.refine_alignment_within_raft(pocket_raft)

        self.tab_matcher = PocketTabMatcher(pieces, pocket)

    def measure_fit(self, src_label, compute_seam_fit_error=False):

        if src_label in self.dst_raft.coords:
            return []

        possible_matches = self.tab_matcher.possible_matches(src_label)
        if len(possible_matches) == 0:
            return []

        retval = []
        for feature_pairs in possible_matches:

            try:
                mse, seam_fe = self.measure_fit2(src_label, feature_pairs, compute_seam_fit_error)
                retval.append((mse, feature_pairs, seam_fe))
            except PocketFitter.FitException as x:
                s = self.raftinator.format_feature_pairs(feature_pairs)
                print(f"measure_fit: {self.pocket!s} feature_pairs={s} failed explosively!")

        return retval

    def measure_fit2(self, src_label, feature_pairs, compute_seam_fit_error=False):

        r = self.raftinator

        src_raft = r.factory.make_raft_for_piece(src_label)

        src_coord = r.aligner.rough_align(self.dst_raft, src_raft, feature_pairs)
        src_coord = r.aligner.refine_alignment_between_rafts(
            self.dst_raft, src_raft, src_coord)

        if compute_seam_fit_error:
            seams = r.seamstress.seams_between_rafts(self.dst_raft, src_raft, src_coord)

            good_dsts = {i[0] for i in self.tab_matcher.dst_tab_pair if i is not None}
            seam_fe = r.seamstress.fit_error_for_seams([s for s in seams if s.dst.piece in good_dsts])
        else:
            seam_fe = None

        new_raft = r.factory.merge_rafts(self.dst_raft, src_raft, src_coord)

        mse = self.compute_mse_with_refinement(new_raft)

        return (mse, seam_fe)

    def compute_mse_with_refinement(self, raft):
        return [i.mse for i in self.compute_fit_error_with_refinement(raft)]

    def compute_fit_error_with_refinement(self, raft):

        fit_error = []

        r = self.raftinator

        seams = r.get_seams_for_raft(raft)
        fit_error.append(r.get_total_fit_error_for_raft_and_seams(raft, seams))
                
        for _ in range(self.num_refine):
            raft = r.refine_alignment_within_raft(raft, seams)
            
            seams = r.get_seams_for_raft(raft)
            fit_error.append(r.get_total_fit_error_for_raft_and_seams(raft, seams))

        return fit_error

