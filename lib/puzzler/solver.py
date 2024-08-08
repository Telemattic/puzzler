import puzzler
import collections
import csv
from datetime import datetime
import functools
import itertools
import json
import math
import numpy as np
import operator
import os
import re
import scipy
import time
from dataclasses import dataclass

def pairwise_circular(iterable):
    # https://stackoverflow.com/questions/36917042/pairwise-circular-python-for-loop
    a, b = itertools.tee(iterable)
    first = next(b, None)
    return zip(a, itertools.chain(b, (first,)))

Coord = puzzler.align.Coord
Size = puzzler.raft.Size

@dataclass
class Geometry:
    coords: "dict[str,Coord]"
    size: Size

class BorderSolver:

    def __init__(self, pieces, puzzle_path = None):
        self.pieces = pieces
        self.pred = dict()
        self.succ = dict()
        self.corners = []
        self.edges = []
        self.raftinator = puzzler.raft.Raftinator(self.pieces)
        self.puzzle_path = puzzle_path

        for p in self.pieces.values():
            n = len(p.edges)
            if n == 0:
                continue

            # HACK: drop incorrectly labeled border pieces
            if False and len(self.pieces) == 1026 and not re.fullmatch(r"([A-Z]+(1|38))|((A|AA)\d+)", p.label):
                print(f"HACK: Skipping {p.label}, not actually a border piece!")
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

        return Geometry(coords, size)
    
    def estimate_puzzle_size(self, border):

        axis = 3
        size = np.zeros(4, dtype=float)
        for label in border:
            p = self.pieces[label]

            pred_tab = p.tabs[self.pred[label][1]]
            succ_tab = p.tabs[self.succ[label][1]]

            pred_edge = p.edges[self.pred[label][0]]
            succ_edge = p.edges[self.succ[label][0]]
            
            if len(p.edges) == 2:
                size[axis] += puzzler.math.distance_to_line(
                    pred_tab.ellipse.center, succ_edge.line.pts)
                axis = (axis + 1) % 4
                size[axis] += puzzler.math.distance_to_line(
                    succ_tab.ellipse.center, pred_edge.line.pts)
            else:
                vec = puzzler.math.unit_vector(pred_edge.line.pts[1] - pred_edge.line.pts[0])
                size[axis] += np.dot(vec, succ_tab.ellipse.center - pred_tab.ellipse.center)

        with np.printoptions(precision=1):
            print(f"estimate_puzzle_size: {size=}")

        return size

    def link_pieces_revised(self, scores):

        rescore = dict()
        for dst, sources in scores.items():
            # source is a dict, flatten it into a list so
            # we can sort it by MSE
            sources = [(*score, src) for src, score in sources.items()]
            sources.sort(key=operator.itemgetter(0))
            rescore[dst] = sources

        # put border candidate pieces in ascending order of MSE for
        # best fit, hopefully insuring that the false edge pieces get
        # processed after the true edge pieces
        candidates = sorted(set(self.corners + self.edges), key=lambda x: rescore[x][0])

        all_pairs = dict()
        pairs = dict()
        used = set()

        for dst in candidates:

            best_src = rescore[dst][0]
            best = best_src[-1]

            all_pairs[dst] = best
            
            print(f"{dst:4s} <- {best:4s} (mse={best_src[0]:.3f})")

            if best in used:
                print(f"best match for {dst} is {best} and it has already been used :O")
                continue

            used.add(best)
            pairs[dst] = best

        if False:
            with open('temp/rescores.json', 'w') as f:
                o = dict()
                o['scores'] = rescore
                o['pairs'] = pairs
                o['all_pairs'] = all_pairs
                o['corners'] = self.corners
                o['edges'] = self.edges
                o['candidates'] = candidates
                o['pred'] = self.pred
                o['succ'] = self.succ
                f.write(json.dumps(o, indent=4))
    
        print(f"{pairs=}")

        retval = []
        visited = set()
        curr = min(self.corners)  # arbitrary, just make it deterministic
        while curr not in visited:
            retval.append(curr)
            visited.add(curr)
            curr = pairs[curr]

        if curr != retval[0]:
            raise ValueError(f"When following edge loop didn't circle back to start!")

        n = len(retval)
        k = len(pairs) - n
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
    
    def link_pieces(self, scores):

        print(f"link_pieces: corners={self.corners}, no. edges={len(self.edges)}")
        # print(f"{scores=}")

        return self.link_pieces_revised(scores)

        expected_pairs = dict()

        # define expected pairs for the 1026 piece puzzle
        # automagically so we can identify problems as they occur when
        # attemping to solve it programmatically
        if False and len(self.pieces) == 1026:
            for i in range(1,38):
                expected_pairs[f"A{i}"] = f"A{i+1}"
                expected_pairs[f"AA{i+1}"] = f"AA{i}"
            for i in range(0,26):
                j = chr(i+ord('A'))
                if i == 25:
                    k = 'AA'
                else:
                    k = chr(i+1+ord('A'))
                expected_pairs[f"{j}38"] = f"{k}38"
                expected_pairs[f"{k}1"] = f"{j}1"
    
            assert set(expected_pairs.keys()) == set(expected_pairs.values())

        # HACK: define pairs for the 1026 piece puzzle automagically
        if False and len(self.pieces) == 1026:
            print(f"HACK: forcing known border solution for puzzle")
            retval = [min(self.corners)]
            curr = expected_pairs[retval[0]]
            while curr != retval[0]:
                retval.append(curr)
                curr = expected_pairs[curr]

            return retval[::-1]

        border = set(self.corners + self.edges)

        pairs = dict()
        all_pairs = dict()
        used = set()
        match_data = list()
        for dst in self.corners + self.edges:

            ss = scores[dst]
            best = min(ss.keys(), key=lambda x: ss[x][0])

            all_pairs[dst] = best

            # print(f"{dst} <- {best} (mse={ss[best][0]})")

            if expected_pairs:
                
                kind = None
                details = ''
                
                sss = sorted([(v[0], k) for k, v in ss.items()])
                expected = expected_pairs.get(dst)
                if expected is None:
                    kind = 'bad'
                    details = "dst not a border piece!"
                elif expected != best:
                    kind = 'bad'
                    details = "expected src not scored!"
                    for i, (v, k) in enumerate(sss):
                        if k == expected:
                            details = f"found at position {i}, mse={v:.1f}"
                else:
                    kind = 'good'
                
                match_data.append({'dst': dst,
                                   'src': best,
                                   'mse': sss[0][0],
                                   'kind': kind,
                                   'expected': expected,
                                   'details': details})

            expected = expected_pairs.get(dst)
            sss = sorted([(v[0], k) for k, v in ss.items()])
            if expected_pairs and expected != best:
                if expected in border:
                    details = "not scored!"
                    for i, (v, k) in enumerate(sss):
                        if k == expected:
                            details = f"found at position {i} mse={v:.1f}"
                else:
                    details = "not a border piece!"
                print(f"{dst:4s} <- {best:4s} (mse={sss[0][0]:.1f}, expected {expected} {details})")
                continue
            
            print(f"{dst:4s} <- {best:4s} (mse={sss[0][0]:.1f})")

            if best in used:
                print(f"best match match for {dst} is {best} and it has already been used!")
                sss = sorted([(f"{v[0]:.1f}", k) for k, v in ss.items()])
                # print(sss)
                continue

            # greedily assume the best fit will be available, if it
            # isn't then we'll have to try harder (possibly *much*
            # harder)
            assert best not in used
            used.add(best)
            pairs[dst] = best

        print(f"{pairs=}")

        if match_data:
            self.output_border_match_data(match_data, expected_pairs, all_pairs)

        # make sure the border pieces form a single ring
        visited = set()
        curr = next(iter(pairs.keys()))
        while curr not in visited:
            visited.add(curr)
            curr = pairs[curr]

        assert visited == used == set(pairs.keys())

        retval = [min(self.corners)] # ["H1"] if "H1" in self.corners else [self.corners[0]]
        curr = pairs[retval[0]]
        while curr != retval[0]:
            retval.append(curr)
            curr = pairs[curr]

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

        borders = self.corners + self.edges
        return {dst: self.score_edge_piece(dst, borders) for dst in borders}

    def score_edge_piece(self, dst, sources):

        dst_piece = self.pieces[dst]
        dst_desc = self.succ[dst]

        scores = dict()
        for src in sources:

            # while the fit might be excellent, this would prove
            # topologically difficult
            if src == dst:
                continue

            src_piece = self.pieces[src]
            src_desc = self.pred[src]

            # tabs have to be complementary (one indent and one
            # outdent)
            if dst_piece.tabs[dst_desc[1]].indent == src_piece.tabs[src_desc[1]].indent:
                continue

            scores[src] = (self.score_edge_pair(dst, dst_desc, src, src_desc), dst_desc, src_desc)

        return scores
                
    def score_edge_pair(self, dst_label, dst_desc, src_label, src_desc):

        r = self.raftinator
            
        Feature = puzzler.raft.Feature
        edge_pair = (Feature(dst_label, 'edge', dst_desc[0]), Feature(src_label, 'edge', src_desc[0]))
        tab_pair = (Feature(dst_label, 'tab', dst_desc[1]), Feature(src_label, 'tab', src_desc[1]))

        dst_raft = r.factory.make_raft_for_piece(dst_label)
        src_raft = r.factory.make_raft_for_piece(src_label)

        src_coord = r.aligner.rough_align_edge_and_tab(dst_raft, src_raft, edge_pair, tab_pair)
        raft = r.factory.merge_rafts(puzzler.raft.RaftAlignment(dst_raft, src_raft, src_coord))

        seams = r.get_seams_for_raft(raft)
        raft = r.aligner.refine_edge_alignment_within_raft(raft, seams, edge_pair)
                    
        return r.get_total_error_for_raft_and_seams(raft)

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

    def __init__(self, pieces):
        self.pieces = pieces
        self.raftinator = puzzler.raft.Raftinator(pieces)

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
        raft = r.factory.merge_rafts(puzzler.raft.RaftAlignment(dst_raft, src_raft, src_coord))

        seams = r.get_seams_for_raft(raft)
        raft = r.aligner.refine_edge_alignment_within_raft(raft, seams, edge_pair)
                    
        return r.get_total_error_for_raft_and_seams(raft)

class OverlappingPieces:

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
                
class ClosestPieces:

    def __init__(self, pieces, coords, axes_opt, distance_query_cache):
        self.pieces = pieces
        self.coords = coords
        self.axes_opt = axes_opt
        self.overlaps = OverlappingPieces(pieces, coords)
        self.max_dist = 50
        self.distance_query_cache = distance_query_cache

        # axes_opt: optional coordinates of the 4 axes
        #
        #    +<--- 2 ---+
        #    |          ^
        #    3          |
        #    |          1
        #    v          |
        #    +--- 0 --->+
        #
        # e.g. (0, puzzle_width, puzzle_height, 0)
        assert axes_opt is None or len(axes_opt) == 4

    def __call__(self, src_label):

        src_piece = self.pieces[src_label]
        src_coords = self.coords[src_label]
        num_points = len(src_piece.points)
        
        ret_dist = np.full(num_points, self.max_dist)
        ret_no = np.zeros(num_points, dtype=np.int32)
        ret_labels = ['none']

        def piece_overlap(dst_label):
            dst_piece = self.pieces[dst_label]
            dst_coords = self.coords[dst_label]

            dst_dist = np.abs(
                self.distance_query_cache.query(dst_piece, dst_coords, src_piece, src_coords))

            ii = np.nonzero(dst_dist < ret_dist)[0]
            if 0 == len(ii):
                return
            
            ret_dist[ii] = dst_dist[ii]
            ret_no[ii] = len(ret_labels)
            ret_labels.append(dst_label)

        candidates = self.overlaps(src_coords.xy, src_piece.radius + self.max_dist).tolist()

        for dst_label in candidates:

            if dst_label != src_label:
                piece_overlap(dst_label)

        src_points = None

        def axis_overlap(xy, loc, label):

            center = src_coords.xy[xy]
            radius = src_piece.radius + self.max_dist
            overlaps = center - radius <= loc <= center + radius
            if not overlaps:
                return

            nonlocal src_points
            if src_points is None:
                transform = puzzler.render.Transform()
                transform.translate(src_coords.xy).rotate(src_coords.angle)
                src_points = transform.apply_v2(src_piece.points)
            
            dst_dist = np.abs(src_points[:,xy] - loc)
            ii = np.nonzero(dst_dist < ret_dist)[0]
            if 0 == len(ii):
                return
            
            ret_dist[ii] = dst_dist[ii]
            ret_no[ii] = len(ret_labels)
            ret_labels.append(label)

        if self.axes_opt:
            axis_overlap(1, self.axes_opt[0], 'axis0')
            axis_overlap(0, self.axes_opt[1], 'axis1')
            axis_overlap(1, self.axes_opt[2], 'axis2')
            axis_overlap(0, self.axes_opt[3], 'axis3')

        retval = collections.defaultdict(list)

        i = 0
        for key, group in itertools.groupby(ret_no):
            j = len(list(group))
            key_s = ret_labels[key]
            retval[key_s].append((i, i+j-1))
            i += j
            
        for v in retval.values():
            if len(v) >= 2:
                head = v[0]
                tail = v[-1]
                if 0 == head[0] and i-1 == tail[1]:
                    v[0] = (tail[0], head[1])
                    v.pop()

        return retval

class BoundaryComputer:

    def __init__(self, pieces):
        self.pieces = pieces

    def find_boundaries_from_adjacency(self, adjacency):

        successors, neighbors, nodes_on_frontier = self.compute_successors_and_neighbors(adjacency)
        boundaries = self.find_boundaries(successors, neighbors, nodes_on_frontier)[0]
        return [self.simplify_boundary(i) for i in boundaries]

    @staticmethod
    def to_dotty(f, successors, neighbors, nodes_on_frontier):

        assert all(neighbors[v] == k for k, v in neighbors.items())

        node_set = successors.keys() | set(successors.values()) | neighbors.keys() | set(neighbors.values())
        nodes = dict()
        for i, n in enumerate(node_set):
            nodes[n] = f"node_{i}"

        print('digraph G {', file=f)

        for k, v in nodes.items():
            attr = f'label="{k}"'
            if k in nodes_on_frontier:
                attr += ' style=bold'
            print(f'  {v} [{attr}]', file=f)

        for k, v in successors.items():
            print(f'  {nodes[k]} -> {nodes[v]} [style=dashed]', file=f)

        for k, v in neighbors.items():
            if nodes[k] <= nodes[v]:
                print(f'  {nodes[k]} -> {nodes[v]} [dir=both]', file=f)

        print('}', file=f)

    @staticmethod
    def simplify_boundary(boundary):

        # Gross: we can end up with the same piece appearing
        # consecutively on the boundary and get confused, just
        # smoosh them all together and pray

        if len(boundary) > 1 and boundary[0][0] == boundary[-1][0]:
            for i, b in enumerate(boundary):
                if boundary[0][0] != b[0]:
                    break

            boundary = boundary[i:] + boundary[:i]

        retval = []
        for k, g in itertools.groupby(boundary, key=operator.itemgetter(0)):
            g = list(j for _, j in g)
            a, b = g[0][0], g[-1][1]
            retval.append((k, (a,b)))
                    
        return retval
            
    def compute_successors_and_neighbors(self, adjacency):

        src_and_range_to_dst = dict()
        src_and_dst_to_range = dict()

        successors = dict()
    
        for src_label, src_adjacency_list in adjacency.items():
        
            src = self.pieces[src_label]
            n = len(src.points)
        
            def range_length(r):
                a, b = r
                return len(puzzler.commands.align.RingRange(a, b+1, n))

            ranges = []
            for dst_label, src_ranges in src_adjacency_list.items():
                # print(f"{src_label=} {dst_label=} {src_ranges=}")
                if dst_label == 'none':
                    ranges += src_ranges
                else:
                    r = max(src_ranges, key=range_length)
                    ranges.append(r)
                    src_and_range_to_dst[(src_label, r)] = dst_label
                    src_and_dst_to_range[(src_label, dst_label)] = r

            ranges.sort()
            for prev, curr in pairwise_circular(ranges):
                successors[(src_label, prev)] = (src_label, curr)

        neighbors = dict()

        nodes_on_frontier = set()

        for (src_label, src_range) in successors:

            if dst_label := src_and_range_to_dst.get((src_label, src_range)):
                if dst_range := src_and_dst_to_range.get((dst_label, src_label)):
                    neighbors[(src_label, src_range)] = (dst_label, dst_range)
            else:
                nodes_on_frontier.add((src_label, src_range))

        return (successors, neighbors, nodes_on_frontier)

    def find_boundaries(self, successors, neighbors, nodes_on_frontier):

        covered = set()
        retval = []
        fullpaths = []

        for head in nodes_on_frontier:

            if head in covered:
                continue

            frontier = [head]
            visited = set(frontier)

            fullpath = [head]

            # on the frontier, therefore has no neighbor, must take
            # successor
            curr = successors[head]

            while curr != head:

                if neighbor := neighbors.get(curr):
                    # if the current node has a neighbor (is not on the
                    # frontier) then jump to the neighbor and then to its
                    # successor (by definition the neighbor's neighbor would
                    # take us back to curr)
                    curr = neighbor
                else:
                    # the current node has no neighbor, and is
                    # therefore on the frontier
                    visited.add(curr)
                    frontier.append(curr)

                fullpath.append(curr)
                curr = successors[curr]

            covered |= visited

            retval.append(frontier)
            fullpaths.append(fullpath)

        return retval, fullpaths

class FrontierExplorer:

    def __init__(self, pieces):
        self.pieces = pieces

    def find_tabs(self, frontier):

        retval = []
        
        for l, (a, b) in frontier:
            p = self.pieces[l]
            rr = puzzler.commands.align.RingRange(a, b, len(p.points))
            
            included_tabs = [i for i, tab in enumerate(p.tabs) if all(j in rr for j in tab.tangent_indexes)]

            def position_in_ring(i):
                tab = p.tabs[i]
                begin = tab.tangent_indexes[0]
                if begin < rr.a:
                    begin += rr.n
                return begin

            included_tabs.sort(key=position_in_ring)

            retval += [(l, i) for i in included_tabs]

        return retval

    def find_interesting_corners(self, frontier, coords):

        tabs = self.find_tabs(frontier)
        dirs = []
        for tab in tabs:
            p, v = self.get_tab_center_and_direction(tab)
            t = coords[tab[0]].xform
            dirs.append((t.apply_v2(p), t.apply_n2(v)))
            
        scores = []
        for i, curr_dir in enumerate(dirs):
            p1, v2 = dirs[i-1]
            p3, v4 = curr_dir
            t = np.cross(p3 - p1, v4)
            u = np.cross(p3 - p1, v2)
            d = np.cross(v2, v4)
            if d != 0.:
                t /= d
                u /= d
            else:
                t = 0.
                u = 0.
            scores.append((np.dot(v2, v4), t, u))
            
        return [(scores[i], tabs[i-1], tabs[i]) for i in range(len(tabs))]

    def get_tab_center_and_direction(self, tab):

        p = self.pieces[tab[0]]
        t = p.tabs[tab[1]]
        v = p.points[np.array(t.tangent_indexes)] - t.ellipse.center
        v = v / np.linalg.norm(v, axis=1)
        v = np.sum(v, axis=0)
        v = v / np.linalg.norm(v)
        if not t.indent:
            v = -v
        return (t.ellipse.center, v)

class PuzzleSolver:

    def __init__(self, pieces, *, expected=None, puzzle_path=None):
        self.pieces = pieces
        self.geometry = None
        self.frontiers = None
        self.corners = []
        self.distance_query_cache = puzzler.align.DistanceQueryCache()
        self.use_raftinator = True
        self.raftinator = puzzler.raft.Raftinator(pieces)
        self.seams = []
        self.expected_tab_matches = expected
        self.start_time = time.monotonic()
        self.puzzle_path = puzzle_path

    def solve(self):
        if self.geometry:
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
        
        bs = BorderSolver(self.pieces, self.puzzle_path)

        t0 = time.monotonic()

        scores = bs.score_matches()

        t_score = time.monotonic() - t0

        print(f"solve_border: {t_score=:.1f}")

        if True:
            ts = datetime.now().strftime('%Y%m%d-%H%M%S')
            path = self.next_path('border_match_scores_' + ts, 'tab')
            print(f"Saving border score data to {path}")
            self.save_border_scores(path, scores)

        if False:
            coords = dict()
            coords['A22'] = Coord()
            coords['A23'] = scores['A22']['A23'][1]
            print(f"{coords=} {scores['A22']['A23']}")
            self.geometry = Geometry(coords, (0., 0.))
            return
        
        border = bs.link_pieces(scores)
        # print(f"{border=}")
        bs.estimate_puzzle_size(border)

        self.geometry = bs.init_placement(border)

        width, height = self.geometry.size
        print(f"puzzle_size: {width=:.1f} {height=:.1f}")

        self.update_adjacency()

        ts = datetime.now().strftime('%Y%m%d-%H%M%S')

        if False:
            path = self.next_path('matches_' + ts, 'csv')
            self.save_tab_matches(path)
        
        path = self.next_path('solver_' + ts, 'json')
        save_json(path, self)

    @staticmethod
    def next_path(fname, ext):

        dname = r'C:\temp\puzzler\align'
        i = 0
        while True:
            path = os.path.join(dname, f"{fname}_{i}.{ext}")
            if not os.path.exists(path):
                return path
            i += 1

    def solve_field(self):

        t_start = time.monotonic() - self.start_time

        Feature = puzzler.raft.Feature

        def get_axis_features(raft):

            rfc = puzzler.raft.RaftFeaturesComputer(self.pieces)
            
            border = None
            for frontier in rfc.compute_frontiers(raft.coords):
                if all(i.kind == 'edge' for i in frontier):
                    border = frontier
                    break

            if border is None:
                return None

            axes = rfc.split_frontier_into_axes(border, raft.coords)

            # convert from a dict to an array
            axes = [axes.get(i, []) for i in range(4)]

            fh = puzzler.raft.FeatureHelper(self.pieces, raft.coords)

            # rotate the array so that the "natural" axis 0 is first
            for i, axis in enumerate(axes):
                if len(axis) == 0:
                    continue
                vec = fh.get_edge_unit_vector(axis[0])
                if np.dot(vec, np.array((-1, 0))) > .8:
                    break

            if i < 4:
                axes = axes[i:] + axes[:i]

            return axes

        if self.geometry is None:
            return

        if not self.corners:
            return

        self.distance_query_cache.purge()

        t0 = time.monotonic()

        fits = self.score_pockets()

        t_score = time.monotonic() - t0

        if not fits:
            # UI sentinel -- there is no more progress to be made
            self.corners = []
            return
        
        fits.sort(key=operator.itemgetter(0))

        for i, f in enumerate(fits[:20]):
            r, mse, src_label, feature_pairs = f
            dst = ','.join(str(i[0]) for i in feature_pairs)
            src = ','.join(str(i[1]) for i in feature_pairs)
            print(f"{i:2d}: {src_label:4s} {mse=:5.1f} {src=} {dst=}")

        _, _, src_label, feature_pairs = fits[0]

        dst_raft = puzzler.raft.Raft(self.geometry.coords, self.geometry.size)
        src_raft = self.raftinator.factory.make_raft_for_piece(src_label)
        new_raft = self.raftinator.align_and_merge_rafts_with_feature_pairs(
            dst_raft, src_raft, feature_pairs)

        do_refine = len(new_raft.coords) % 20 == 0

        if do_refine:

            t0 = time.monotonic()

            seams = self.raftinator.get_seams_for_raft(new_raft)

            t_seams = time.monotonic() - t0
        
            t0 = time.monotonic()

            axis_features = get_axis_features(new_raft)

            t_axis = time.monotonic() - t0

            t0 = time.monotonic()

            refined_raft = self.raftinator.aligner.refine_alignment_within_raft(
                new_raft, seams, axis_features)

            t_refine = time.monotonic() - t0

        else:

            t_seams = t_axis = t_refine = 0.
            refined_raft = new_raft

        self.geometry.coords = refined_raft.coords
        self.geometry.size = refined_raft.size

        self.update_adjacency()

        ts = datetime.now().strftime('%Y%m%d-%H%M%S')
        
        path = self.next_path('matches_' + ts, 'csv')
        self.save_tab_matches(path)
        
        path = self.next_path('solver_' + ts, 'json')
        save_json(path, self)

        t_finish = time.monotonic() - self.start_time
        
        t_process = time.monotonic() - self.start_time
        print(f"solve_field: wall: {t_start=:.1f} {t_finish=:.1f} elapsed: {t_process=:.1f} {t_score=:.1f} {t_seams=:.1f} {t_axis=:.1f} {t_refine=:.1f}\n")

    def refine(self):

        def get_axis_features(raft):

            rfc = puzzler.raft.RaftFeaturesComputer(self.pieces)
            
            border = None
            for frontier in rfc.compute_frontiers(raft.coords):
                if all(i.kind == 'edge' for i in frontier):
                    border = frontier
                    break

            if border is None:
                return None

            axes = rfc.split_frontier_into_axes(border, raft.coords)

            # convert from a dict to an array
            axes = [axes.get(i, []) for i in range(4)]

            fh = puzzler.raft.FeatureHelper(self.pieces, raft.coords)

            # rotate the array so that the "natural" axis 0 is first
            for i, axis in enumerate(axes):
                if len(axis) == 0:
                    continue
                vec = fh.get_edge_unit_vector(axis[0])
                if np.dot(vec, np.array((-1, 0))) > .8:
                    break

            if i < 4:
                axes = axes[i:] + axes[:i]

            return axes

        if self.geometry is None:
            return

        new_raft = puzzler.raft.Raft(self.geometry.coords, self.geometry.size)

        do_refine = True

        if do_refine:

            seams = self.raftinator.get_seams_for_raft(new_raft)

            axis_features = get_axis_features(new_raft)

            refined_raft = self.raftinator.aligner.refine_alignment_within_raft(
                new_raft, seams, axis_features)

        else:

            refined_raft = new_raft

        self.geometry.coords = refined_raft.coords
        self.geometry.size = refined_raft.size

        self.update_adjacency()

        ts = datetime.now().strftime('%Y%m%d-%H%M%S')
        
        path = self.next_path('matches_' + ts, 'csv')
        self.save_tab_matches(path)
        
        path = self.next_path('solver_' + ts, 'json')
        save_json(path, self)

    def score_pockets(self):

        raft = puzzler.raft.Raft(self.geometry.coords, self.geometry.size)
        pocket_finder = puzzler.commands.quads.PocketFinder(self.pieces, raft)
        pockets = pocket_finder.find_pockets_on_frontiers()

        fits = []
        for pocket in pockets:
            
            v = self.score_pocket(pocket)
            
            if len(v) > 1:
                r = v[0][0] / v[1][0]
                fits.append((r, *v[0]))
            elif v:
                fits.append((1., *v[0]))

            s = [f"{i[1]}:{i[0]:.1f}" for i in v[:3]]

            print(f"{pocket!s}: " + ', '.join(s))

        return fits

    @functools.lru_cache(maxsize=128)
    def score_pocket(self, pocket):

        pocket_raft = puzzler.raft.Raft(self.geometry.coords, self.geometry.size)
        pf = puzzler.commands.quads.PocketFitter(self.pieces, pocket_raft, pocket, 1)

        fits = []
        
        for src_label in self.pieces:
            if src_label in self.geometry.coords:
                continue
            for mse, feature_pairs in pf.measure_fit(src_label):
                fits.append((mse[-1], src_label, feature_pairs))

        return sorted(fits, key=operator.itemgetter(0))

    def save_tab_matches(self, path):

        tab_xy = []
        radii = []
        labels = []
        for k, v in self.geometry.coords.items():
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
                
    def update_adjacency(self):

        coords = self.geometry.coords
        width, height = self.geometry.size
        axes_opt = (0., width, height, 0.)
        closest_pieces = ClosestPieces(self.pieces, coords, axes_opt, self.distance_query_cache)
        adjacency = dict((i, closest_pieces(i)) for i in self.geometry.coords)
        
        frontiers = BoundaryComputer(self.pieces).find_boundaries_from_adjacency(adjacency)
        
        fe = FrontierExplorer(self.pieces)
        corners = []
        for f in frontiers:
            corners += fe.find_interesting_corners(f, coords)
        good_corners = []
        for (s, t, u), tab0, tab1 in corners:
            is_interesting = abs(s) < .5 and 50 < t < 1000 and 50 < u < 1000
            if is_interesting:
                good_corners.append((tab0, tab1))

        self.frontiers = frontiers
        self.corners = good_corners

def load_json(path, pieces):

    with open(path, 'r') as f:
        s = f.read()

    return from_json(pieces, s)

def from_json(pieces, s):

    def parse_geometry(o):
        if o is None:
            return None
        
        size = tuple(o['size'])
        coords = dict()
        for k, v in o['coords'].items():
            coords[k] = parse_affine_transform(v)

        return Geometry(coords, size)

    def parse_affine_transform(o):
        angle = o['angle']
        xy = tuple(o['xy'])
        return Coord(angle, xy)

    def parse_frontiers(o):
        if o is None:
            return None

        return [parse_frontier(i) for i in o]

    def parse_frontier(o):
        return [(a, tuple(b)) for a, b in o]

    def parse_corners(o):

        if o is None:
            return None

        return [parse_corner(i) for i in o]

    def parse_corner(o):

        return tuple((a,b) for a, b in o)

    o = json.loads(s)

    assert set(pieces.keys()) == set(o['pieces'])

    geometry = parse_geometry(o['geometry'])
    # frontiers = parse_frontiers(o['frontiers'])
    # corners = parse_corners(o['corners'])

    solver = PuzzleSolver(pieces)
    solver.geometry = geometry
    # solver.frontiers = frontiers
    # solver.corners = corners
    solver.update_adjacency()

    return solver

def save_json(path, solver):

    with open(path, 'w') as f:
        f.write(to_json(solver))

def to_json(solver):

    def format_pieces(pieces):
        return sorted(pieces.keys())

    def format_geometry(geometry):
        if geometry is None:
            return None

        coords = dict((k, format_coord(v))
                      for k, v in geometry.coords.items())

        return {'size':geometry.size, 'coords':coords}

    def format_coord(t):
        return {'angle':t.angle, 'xy':t.xy.tolist()}

    def format_frontiers(frontiers):

        return frontiers

        if frontiers is None:
            return None

        return [format_frontier(i) for i in frontiers]

    def format_frontier(f):

        return str(f)

    def format_corners(corners):

        if corners is None:
            return None

        return corners

    o = dict()
    o['pieces'] = format_pieces(solver.pieces)
    o['geometry'] = format_geometry(solver.geometry)
    o['frontiers'] = format_frontiers(solver.frontiers)
    o['corners'] = format_corners(solver.corners)

    return json.JSONEncoder(indent=0).encode(o)
