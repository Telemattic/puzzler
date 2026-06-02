import puzzler
import collections
import copy
import csv
from datetime import datetime
import functools
import heapq
import itertools
import json
import math
import networkx as nx
import numpy as np
import operator
import os
import re
import scipy
import time
from dataclasses import dataclass
from typing import NamedTuple

def pairwise_circular(iterable):
    # https://stackoverflow.com/questions/36917042/pairwise-circular-python-for-loop
    a, b = itertools.tee(iterable)
    first = next(b, None)
    return zip(a, itertools.chain(b, (first,)))

Coord = puzzler.align.Coord
Size = puzzler.raft.Size

class BorderSolver:

    def __init__(self, raftinator, expected=None):
        self.pieces = raftinator.pieces
        self.pred = dict()
        self.succ = dict()
        self.corners = []
        self.edges = []
        self.raftinator = raftinator
        self.expected = expected

        for p in self.pieces.values():
            n = len(p.edges)
            if n == 0:
                continue

            # HACK: reject pieces with too many edges that aren't clearly corner pieces
            if True and n > 1:
                assert n == 2
                l0, l1 = p.edges[0].line, p.edges[1].line
                v0 = puzzler.math.unit_vector(l0.pts[1] - l0.pts[0])
                v1 = puzzler.math.unit_vector(l1.pts[1] - l1.pts[0])
                cross = np.cross(v0, v1)
                with np.printoptions(precision=3):
                    print(f"CORNER: {p.label} {v0=} {v1=} {cross=}")
                if np.abs(cross) < 0.9:
                    print(f"HACK: piece {p.label} has {n} edges, but doesn't look like a corner, only keeping longest edge")
                    len0 = np.linalg.norm(l0.pts[1] - l0.pts[0])
                    len1 = np.linalg.norm(l1.pts[1] - l1.pts[0])
                    p.edges.pop(1 if len0 > len1 else 0)
                    n = 1
                elif cross > 0:
                    # are the edges in the right order?
                    print(f"CORNER: edges of corner {p.label} in wrong order, reversing them")
                    p.edges = [p.edges[1], p.edges[0]]

            (pred, succ) = self.compute_edge_info(p)
            self.pred[p.label] = pred
            self.succ[p.label] = succ

            if n == 2:
                self.corners.append(p.label)
            elif n == 1:
                self.edges.append(p.label)

    def init_placement(self, border):

        icp = puzzler.icp.IteratedClosestPoint()
        
        axes = [
            icp.make_axis(np.array((0, -1), dtype=float), 0., True),
            icp.make_axis(np.array((1, 0), dtype=float)),
            icp.make_axis(np.array((0, 1), dtype=float)),
            icp.make_axis(np.array((-1, 0), dtype=float), 0., True)
        ]

        bodies = dict()

        axis_no = 3

        for i in border:
            
            p = self.pieces[i]

            e = p.edges[self.succ[i][0]]
            v = e.line.pts[0] - e.line.pts[1]
            angle = axis_no * math.pi / 2 - np.arctan2(v[1], v[0])
            
            bodies[i] = icp.make_rigid_body(angle)
            icp.add_axis_correspondence(bodies[i], e.line.pts, axes[axis_no])

            if self.pred[i][0] != self.succ[i][0]:
                axis_no = (axis_no + 1) % 4
                e = p.edges[self.pred[i][0]]
                icp.add_axis_correspondence(bodies[i], e.line.pts, axes[axis_no])

        for prev, curr in pairwise_circular(border):

            p0 = self.pieces[prev]
            p1 = self.pieces[curr]

            v0 = p0.tabs[self.pred[prev][1]].ellipse.center
            v1 = p1.tabs[self.succ[curr][1]].ellipse.center

            dst_normal = puzzler.math.unit_vector(v1)

            icp.add_body_correspondence(
                bodies[prev], np.atleast_2d(v0),
                bodies[curr], np.atleast_2d(v1), np.atleast_2d(dst_normal))

        icp.solve()

        coords = {k: Coord(v.angle, v.center) for k, v in bodies.items()}
        size = (axes[1].value, axes[2].value)

        return puzzler.raft.Raft(coords, size)
    
    def link_pieces(self, scores):

        max_mse = 100090. if len(self.corners) + len(self.edges) < 50 else 20.
        num_extra_edges = 10

        print(f"link_pieces: corners={self.corners}, no. edges={len(self.edges)}")
        
        rescore = dict()
        for dst, sources in scores.items():
            # source is a dict, flatten it into a list so
            # we can sort it by MSE
            sources = [(*score, src) for src, score in sources.items()]
            sources.sort(key=operator.itemgetter(0))
            rescore[dst] = sources

        with open('temp/rescores.json', 'w') as f:
            o = dict()
            o['scores'] = rescore
            f.write(json.dumps(o, indent=4))
                
        G = nx.DiGraph()
        extra_edges = []
        
        for dst in self.corners + self.edges:
            for rank, score in enumerate(rescore[dst], start=1):
                mse, _, _, src = score
                if mse > max_mse:
                    # really high error is assumed to be a mislabeled edge
                    pass
                elif rank == 1:
                    # every node gets its favorite predecessor
                    G.add_edge(src, dst, mse=mse)
                else:
                    # the remaining predecessors (with tolerable
                    # error) are kept as backups
                    extra_edges.append((mse, src, dst))

        # find the longest cycle(s), given just the initial rank 1 edges
        l_max = 0
        longest_cycles = []
        for cycle in nx.simple_cycles(G):
            l = len(cycle)
            if l < l_max:
                continue

            cost = nx.path_weight(G, cycle + [cycle[0]], 'mse')
            s = ' '.join(cycle)
            print(f"Found cycle len={l}, {cost=:.3f}, path={s}")

            if l > l_max:
                l_max = l
                longest_cycles = [(cost, cycle)]
            else:
                longest_cycles.append((cost, cycle))

        # add extra edges in order by lowest error, it's not
        # necessarily the right order (the best order is of course
        # adding just the necessary extra edges to get the optimal
        # solution) but what else can we do?
        extra_edges.sort()

        # num_extra_edges = len(extra_edges)
        print(f"HACK: setting {num_extra_edges=}")

        for i in range(num_extra_edges):

            if l_max == G.number_of_nodes():
                # longest_cycles contains maximal length cycles, although
                # not necessarily the single maximal length cycle with the
                # lowest cost, N.B. that's an instance of the TSP so we
                # shouldn't get too fixated on it
                break

            mse, src, dst = extra_edges[i]
            G.add_edge(src, dst, mse=mse)

            # we added an edge from src -> dst, so now we're looking
            # for the possibility of a cycle that starts at dst and
            # returns to src (which may not exist of course):
            for cycle in nx.all_simple_paths(G, dst, src):
                l = len(cycle)
                if l < l_max:
                    continue

                cost = nx.path_weight(G, cycle + [cycle[0]], 'mse')
                s = ' '.join(cycle)
                print(f"Found cycle len={l}, {cost=:.3f}, path={s}")

                if l > l_max:
                    l_max = l
                    longest_cycles = [(cost, cycle)]
                else:
                    longest_cycles.append((cost, cycle))

        # put them in order of ascending cost
        longest_cycles.sort()

        retval = longest_cycles[0][1]

        # HACK: reverse the cycle because the below assumes it's
        # backward, and we'll reverse it again before returning it
        retval = retval[::-1]

        # rotate the cycle to start with a deterministic corner
        if i := retval.index(min(self.corners)):
            retval = retval[i:] + retval[:i]

        n = len(retval)
        k = len(self.corners) + len(self.edges) - len(retval)
        print(f"Found edge solution of length {n}, which omits {k} edge pieces")

        axis_no = 3
        axes = [0] * 4
        for i in retval:
            if len(self.pieces[i].edges) == 2:
                axis_no = (axis_no + 1) % 4
            axes[axis_no] += 1

        # rotate to an assumed landscape orientation
        print(f"pieces on each axis: {axes}")
        
        w, h = axes[:2]
        if w * 1.1 < h:
            print("rotating to landscape orientation")
            retval = retval[w:] + retval[:w]

        return retval[::-1]
    
    def output_border_match_data(self, match_data, expected_pairs, actual_pairs):
        
        ts = datetime.now().strftime('%Y%m%d-%H%M%S')
        path = PuzzleSolver.next_path('border_match_data_' + ts, 'json')
        
        o = {'match_data': match_data,
             'expected_pairs': expected_pairs,
             'actual_pairs': actual_pairs}
        
        with open(path, 'w', newline='') as f:
            f.write(json.dumps(o, indent=4))
                
    def score_matches(self):

        s = EdgeScorer(self.raftinator)
        return {dst[0]: s.score_edge_piece(dst, self.pred) for dst in self.succ.items()}

    def compute_edge_info(self, piece):

        edges = piece.edges
        tabs = piece.tabs

        edge_succ = len(edges) - 1
        tab_succ = 0
        for i, tab in enumerate(tabs):
            if edges[edge_succ].fit_indexes < tab.fit_indexes:
                tab_succ = i
                break

        edge_pred = 0
        tab_pred = len(tabs) - 1
        for i, tab in enumerate(tabs):
            if edges[edge_pred].fit_indexes < tab.fit_indexes:
                break
            tab_pred = i

        return (edge_pred, tab_pred), (edge_succ, tab_succ)

class EdgeScorer:

    def __init__(self, raftinator):
        self.pieces = raftinator.pieces
        self.raftinator = raftinator

    def score_edge_piece(self, dst, sources):

        dst_label, dst_desc = dst

        dst_piece = self.pieces[dst_label]

        scores = dict()
        for src_label, src_desc in sources.items():

            # while the fit might be excellent, this would prove
            # topologically difficult
            if src_label == dst_label:
                continue

            src_piece = self.pieces[src_label]

            # tabs have to be complementary (one indent and one
            # outdent)
            if dst_piece.tabs[dst_desc[1]].indent == src_piece.tabs[src_desc[1]].indent:
                continue
            
            mse = self.score_edge_pair(dst_label, dst_desc, src_label, src_desc)
            scores[src_label] = (mse, dst_desc, src_desc)

        return scores
                
    def score_edge_pair(self, dst_label, dst_desc, src_label, src_desc):

        r = self.raftinator
            
        Feature = puzzler.raft.Feature
        edge_pair = (Feature(dst_label, 'edge', dst_desc[0]), Feature(src_label, 'edge', src_desc[0]))
        tab_pair = (Feature(dst_label, 'tab', dst_desc[1]), Feature(src_label, 'tab', src_desc[1]))

        dst_raft = r.factory.make_raft_for_piece(dst_label)
        src_raft = r.factory.make_raft_for_piece(src_label)

        src_coord = r.aligner.rough_align_edge_and_tab(dst_raft, src_raft, edge_pair, tab_pair)
        raft = r.factory.merge_rafts(dst_raft, src_raft, src_coord)

        seam = r.seamstress.seam_between_pieces(
            dst_label, raft.coords[dst_label], src_label, raft.coords[src_label])
        raft = r.aligner.refine_edge_alignment_within_raft(raft, [seam], edge_pair)

        if True:
            seam = r.seamstress.seam_between_pieces(
                dst_label, raft.coords[dst_label], src_label, raft.coords[src_label])
            return puzzler.raft.FitError(seam.error, len(seam.src.indices)).mse
        
        return r.get_total_error_for_raft_and_seams(raft)

class OverlappingPiecesLinear:

    def __init__(self, pieces, coords):
        labels = []
        centers = []
        radii = []
        for k, v in pieces.items():
            if k not in coords:
                continue
            labels.append(k)
            centers.append(coords[k].xy)
            radii.append(v.radius)

        self.labels = labels
        self.centers = np.array(centers)
        self.radii = np.array(radii)

    def __call__(self, center, radius):
        dist = np.linalg.norm(center - self.centers, axis=1)
        ii = np.nonzero(self.radii + radius > dist)[0]
        return np.take(self.labels, ii)
                
class OverlappingPiecesKDTree:

    def __init__(self, pieces, coords):
        labels = []
        centers = []
        radii = []
        for k, v in pieces.items():
            if k not in coords:
                continue
            labels.append(k)
            centers.append(coords[k].xy)
            radii.append(v.radius)

        self.labels = np.array(labels)
        self.centers = np.array(centers)
        self.radii = np.array(radii)
        self.max_radius = np.max(radii)
        self.kdtree = scipy.spatial.KDTree(self.centers)

    def __call__(self, center, radius):
        # get all the points possibly close enough (bounded by the
        # largest radius associated with any candidate point)
        retval = []
        for i in self.kdtree.query_ball_point(center, radius+self.max_radius):
            r = radius + self.radii[i]
            d = center - self.centers[i]
            if r*r >= np.sum(d*d):
                retval.append(i)
        return np.take(self.labels, retval)

def OverlappingPieces(pieces, coords):
    return OverlappingPiecesKDTree(pieces, coords)
                
class PuzzleSolver:

    def __init__(self, pieces, *, dirname = None, expected = None, tab_pairs = None):
        self.pieces = pieces
        self.raft = None
        self.distance_query_cache = puzzler.align.DistanceQueryCache()
        self.use_raftinator = True
        self.raftinator = puzzler.raft.Raftinator(pieces)
        self.seams = []
        self.start_time = time.monotonic()
        self.dirname = dirname
        self.expected = expected
        self.tab_pairs = tab_pairs
        self.last_refine = None
        self.pocket_cache = collections.OrderedDict()
        self.pocket_nscore = 2

    def solve(self):
        if self.raft:
            self.solve_field()
        else:
            self.solve_border()

    @staticmethod
    def save_border_scores(path, scores):

        def to_rows(scores):

            retval = []
            for dst, sources in scores.items():
                rows = []
                for src, score in sources.items():
                    mse, dst_desc, src_desc = score
                    raft = f"{dst}/{dst_desc[0]}={src}/{src_desc[0]},{dst}:{dst_desc[1]}={src}:{src_desc[1]}"
                    rows.append({'dst':dst, 'src':src, 'raft':raft, 'mse':mse, 'rank':None})
                rows.sort(key=operator.itemgetter('mse'))
                for i, row in enumerate(rows, start=1):
                    row['rank'] = i
                    row['mse'] = f"{row['mse']:.3f}"
                retval += rows

            return retval

        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, dialect='excel-tab', fieldnames='dst src rank mse raft'.split())
            writer.writeheader()

            writer.writerows(to_rows(scores))

    def solve_border(self):

        bs = BorderSolver(self.raftinator, expected=self.expected)

        scores = bs.score_matches()

        ts = datetime.now().strftime('%Y%m%d-%H%M%S')
        
        if self.dirname:
            path = self.next_path('border_match_scores_' + ts, 'tab')
            print(f"Saving border score data to {path}")
            self.save_border_scores(path, scores)

        border = bs.link_pieces(scores)

        self.raft = bs.init_placement(border)

        width, height = self.raft.size
        print(f"puzzle_size: {width=:.1f} {height=:.1f}")

        if self.dirname:
            self.save_tab_matches(self.next_path('matches_' + ts, 'csv'))
            save_json(self.next_path('solver_' + ts, 'json'), self)

    def next_path(self, fname, ext):

        i = 0
        while True:
            path = os.path.join(self.dirname, f"{fname}_{i}.{ext}")
            if not os.path.exists(path):
                return path
            i += 1

    def solve_field(self):

        if self.raft is None:
            return False

        if len(self.pieces) <= len(self.raft.coords):
            return False

        self.distance_query_cache.purge()

        fits = self.score_pockets()

        if not fits:
            return False

        # rank the fits by MSE, previously they were ranked by how
        # good they were compared to the second best fit, but that
        # only works if you're doing the work to find the second-best
        # fit, and instead we're trying to score as few pieces as
        # possible
        if False and self.tab_pairs:
            fits.sort(key=operator.itemgetter(1))
        else:
            fits.sort(key=operator.itemgetter(0))

        for i, f in enumerate(fits[:20]):
            r, (mse, src_label, feature_pairs, _) = f
            dst = ','.join(str(i[0]) for i in feature_pairs)
            src = ','.join(str(i[1]) for i in feature_pairs)
            print(f"{i:2d}: {src_label:4s} {mse=:5.1f} {src=} {dst=} {r=:.3f}")

        fit = fits[0][1]

        src_label = fit.src_label
        feature_pairs = fit.feature_pairs

        status = self.is_good_match(feature_pairs)
        if status is None or status:
            status = ''
        else:
            status = ' <BAD MATCH>'
        print(f"Placing {src_label}: {self.raftinator.format_feature_pairs(feature_pairs)}{status}")

        dst_raft = self.raft
        src_raft = self.raftinator.factory.make_raft_for_piece(src_label)
        
        assert src_label not in dst_raft.coords

        new_raft = self.raftinator.align_and_merge_rafts_with_feature_pairs(
            dst_raft, src_raft, feature_pairs)

        assert src_label in new_raft.coords

        self.update_raft(new_raft)
        
        if len(new_raft.coords) % 20 == 0:
            self.refine()

        return status == ''

    def is_good_match(self, feature_pairs):
        if self.expected is None:
            return None
        return all(self.expected.get(a) == b for a, b in feature_pairs)

    def refine(self):

        if self.raft is None:
            return

        delta = None
        not_a_superset = False
        if self.last_refine:
            delta = set(self.raft.coords.keys()) - set(self.last_refine.coords.keys())
            for p, c1 in self.last_refine.coords.items():
                c2 = self.raft.coords.get(p)
                if c2 is None:
                    # print(f"refine: not a superset, {p} from last refine is missing")
                    not_a_superset = True
                elif c2.angle != c1.angle or np.any(c2.xy != c1.xy):
                    # print(f"refine: coordinate of {p} has changed, adding to delta")
                    delta.add(p)
        else:
            not_a_superset = True

        if not_a_superset:
            delta = set(self.raft.coords.keys())

        # print(f"refine: delta=({','.join(delta)})")

        new_raft = self.raft

        seams = self.raftinator.get_seams_for_raft(new_raft)

        rafc = puzzler.raft.RaftAxisFeaturesComputer(self.pieces)
        axis_features = rafc.compute_axis_features(new_raft.coords)

        if len(delta):
            refined_raft = self.raftinator.aligner.delta_refine_alignment_within_raft(
                new_raft, delta, seams, axis_features)
        else:
            refined_raft = self.raftinator.aligner.refine_alignment_within_raft(
                new_raft, seams, axis_features)

        self.last_refine = copy.deepcopy(refined_raft)

        self.update_raft(refined_raft)

    def update_raft(self, raft):

        self.raft = raft

        if self.dirname:
            ts = datetime.now().strftime('%Y%m%d-%H%M%S')
            self.save_tab_matches(self.next_path('matches_' + ts, 'csv'))
            save_json(self.next_path('solver_' + ts, 'json'), self)

    def score_pockets(self):

        raft = self.raft
        pocket_finder = puzzler.pocket.PocketFinder(self.pieces, raft)
        pockets = pocket_finder.find_pockets_on_frontiers()

        fits = []
        for pocket in pockets:
            
            v = self.score_pocket(pocket)
            # caching means we have to filter out any fits that are
            # invalid
            v = [i for i in v if i.src_label not in raft.coords]
            if not v:
                continue

            r = float('+inf')
            if len(v) > 1 and v[1].mse != 0.:
                r = v[0].mse / v[1].mse
            fits.append((r, v[0]))

            s = [f"{i.src_label}:{i.mse:.1f}" for i in v[:3]]

            print(f"{pocket!s}: " + ', '.join(s))

        return fits

    def make_key_for_pocket(self, pocket):
        keys = []
        for p in pocket.pieces:
            c = self.raft.coords[p]
            a = c.angle
            x, y = tuple(c.xy)
            keys.append((p, float(a), float(x), float(y)))
        return (pocket.tab_a, pocket.tab_b, tuple(sorted(keys)))

    def score_pocket(self, pocket):

        cache = self.pocket_cache

        key = self.make_key_for_pocket(pocket)
        if key in cache:
            # cache hit: move the entry to the end to mark it as
            # most-recently-used
            cache.move_to_end(key)
            result = cache[key]
            # only return this cached result if all the returned fits
            # are still available to be fit
            if all(i.src_label not in self.raft.coords for i in result):
                return result

        result = self.score_pocket_impl(pocket)

        cache[key] = result
        if len(cache) > 128:
            # pop the first item, aka the LRU
            cache.popitem(last=False)

        return result

    class PocketScore(NamedTuple):
        mse: float
        src_label: str
        feature_pairs: puzzler.raft.FeaturePairs
        tab_error: float

    def score_pocket_impl(self, pocket):

        pf = puzzler.pocket.PocketFitter(
            self.raftinator, self.raft, pocket, num_refine=1, tab_pairs=self.tab_pairs)

        def measure_fit(i):
            mse, tab_error = pf.measure_fit(i.src_label, i.feature_pairs)
            return PuzzleSolver.PocketScore(mse, i.src_label, i.feature_pairs, tab_error)

        def get_max_tab_error(scores):
            return max(i.tab_error.mse for i in scores)

        def heapify_max(scores):
            # heapq.heapify_max requires 3.14
            scores.sort(reverse=True)
            return get_max_tab_error(scores)

        def heapreplace_max(scores, m):
            # heapq.heapreplace_max requires 3.14
            scores[0] = m
            scores.sort(reverse=True)
            return get_max_tab_error(scores)

        free_pieces = set(self.pieces.keys()) - set(self.raft.coords.keys())
        candidates = pf.candidate_matches(free_pieces)

        scores = [measure_fit(i) for i in candidates[:self.pocket_nscore]]
        
        if len(candidates) > self.pocket_nscore:

            # we want a max heap so the top of the heap will be the
            # element with the worst score so far (highest MSE) and
            # thus the one that we'd like to replace should we find
            # anything better
            max_tab_error = heapify_max(scores)
        
            for i in candidates[self.pocket_nscore:]:

                # if the never-to-be-achieved min_tab_error for this
                # candidate is worse than the actual tab_error for the
                # worst candidate found so far then we're done
                # searching, as we "know" we'll be worse than it
                if i.min_tab_error > max_tab_error:
                    break

                m = measure_fit(i)
                if m.mse < scores[0].mse:
                    max_tab_error = heapreplace_max(scores, m)

        return sorted(scores)

    def save_tab_matches(self, path):

        tab_xy = []
        radii = []
        labels = []
        for k, v in self.raft.coords.items():
            p = self.pieces[k]
            centers = np.array([t.ellipse.center for t in p.tabs])
            radii  += [t.ellipse.semi_major for t in p.tabs]
            labels += [(p.label, i) for i in range(len(p.tabs))]
            tab_xy += [xy for xy in v.xform.apply_v2(centers)]

        rows = []

        kdtree = scipy.spatial.KDTree(tab_xy)
        neighbor_dist, neighbor_index = kdtree.query(tab_xy, 2)
        for i, neighbors in enumerate(neighbor_index):
            for j, k in enumerate(neighbors):
                if k == i:
                    continue
                dst = labels[i]
                src = labels[k]
                distance = neighbor_dist[i][j]
                if distance > radii[i]:
                    continue
                # print(f"{labels[i]} - {labels[k]} # {distance=:.1f}")
                rows.append({'dst_piece':dst[0], 'dst_tab_no':dst[1], 'src_piece':src[0], 'src_tab_no':src[1], 'distance':distance})

        with open(path, 'w', newline='') as f:
            field_names = 'dst_piece dst_tab_no src_piece src_tab_no'.split()
            # ignore: don't complain that 'distance' isn't being output
            writer = csv.DictWriter(f, field_names, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
                
def load_json(path, pieces):

    with open(path, 'r') as f:
        s = f.read()

    return from_json(pieces, s)

def from_json(pieces, s):

    def parse_raft(o):
        if o is None:
            return None
        
        size = o.get('size')
        if size is not None:
            size = tuple(size)
            
        coords = dict()
        for k, v in o['coords'].items():
            coords[k] = parse_coord(v)

        return puzzler.raft.Raft(coords, size)

    def parse_coord(o):
        angle = o['angle']
        xy = tuple(o['xy'])
        return Coord(angle, xy)

    o = json.loads(s)

    assert set(pieces.keys()) == set(o['pieces'])

    raft = parse_raft(o['raft'])

    solver = PuzzleSolver(pieces)
    solver.raft = raft

    return solver

def save_json(path, solver):

    with open(path, 'w') as f:
        f.write(to_json(solver))

def to_json(solver):

    def format_pieces(pieces):
        return sorted(pieces.keys())

    def format_raft(raft):
        if raft is None:
            return None

        o = dict()
        if raft.size is not None:
            o['size'] = raft.size

        o['coords'] = dict((k, format_coord(v)) for k, v in raft.coords.items())

        return o

    def format_coord(t):
        return {'angle':t.angle, 'xy':t.xy.tolist()}

    o = dict()
    o['pieces'] = format_pieces(solver.pieces)
    o['raft'] = format_raft(solver.raft)

    return json.JSONEncoder(indent=0).encode(o)
