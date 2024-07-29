import puzzler
import collections
import csv
from datetime import datetime
import itertools
import json
import math
import numpy as np
import operator
import os
import re
import scipy
from dataclasses import dataclass

def pairwise_circular(iterable):
    # https://stackoverflow.com/questions/36917042/pairwise-circular-python-for-loop
    a, b = itertools.tee(iterable)
    first = next(b, None)
    return zip(a, itertools.chain(b, (first,)))

@dataclass
class BorderConstraint:
    """BorderConstraint associates an edge, identified by a (label, edge_no) tuple,
with an axis"""
    edge: "tuple[str,int]"
    axis: int

@dataclass
class TabConstraint:
    """TabConstraint associates a pair of tabs, each identified by a (piece_label, tab_no) tuple"""
    a: "tuple[str,int]"
    b: "tuple[str,int]"

Coord = puzzler.align.Coord

@dataclass
class Geometry:
    width: float
    height: float
    coords: "dict[str,Coord]"

class BorderSolver:

    def __init__(self, pieces, use_raftinator=False):
        self.pieces = pieces
        self.pred = dict()
        self.succ = dict()
        self.corners = []
        self.edges = []
        self.use_raftinator = use_raftinator
        self.raftinator = puzzler.raft.Raftinator(self.pieces)

        for p in self.pieces.values():
            n = len(p.edges)
            if n == 0:
                continue

            # HACK: drop incorrectly labeled border pieces
            if True and len(self.pieces) == 1026 and not re.fullmatch("([A-Z]+(1|38))|((A|AA)\d+)", p.label):
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

            (pred, succ) = self.compute_edge_info(p)
            self.pred[p.label] = pred
            self.succ[p.label] = succ

            if n == 2:
                self.corners.append(p.label)
            elif n == 1:
                self.edges.append(p.label)

    def init_constraints(self, border):

        # assuming a rectangular puzzle, the edges are as shown:
        #
        #    +<--- 2 ---+
        #    |          ^
        #    3          |
        #    |          1
        #    v          |
        #    +--- 0 --->+
        #
        #  0 and 3 are fixed, coincident to the X and Y axes
        #  1 and 2 are floating, parallel to their respective axes

        axis = 3
        constraints = []
        for label in border:

            constraints.append(BorderConstraint((label, self.succ[label][0]), axis))
            if label in self.corners:
                axis = (axis + 1) % 4
                constraints.append(BorderConstraint((label, self.pred[label][0]), axis))

        for a, b in pairwise_circular(border):
            constraints.append(TabConstraint((a, self.pred[a][1]), (b, self.succ[b][1])))

        return constraints

    def init_placement_X(self, border, scores):

        coords = dict()
        start = border[0] # "I1"

        p = self.pieces[start]
        e = p.edges[self.pred[start][0]]
        v = e.line.pts[0] - e.line.pts[1]
        angle = -np.arctan2(v[1], v[0])

        coords[start] = Coord(angle)

        for prev, curr in zip(border, border[1:]):
            assert prev in coords and curr not in coords

            prev_m = coords[prev].xform.matrix
            curr_m = scores[curr][prev][1].xform.matrix

            coords[curr] = Coord.from_matrix(curr_m @ prev_m)

        return Geometry(0., 0., coords)

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
        width = axes[1].value
        height = axes[2].value

        return Geometry(width, height, coords)
    
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

    def link_pieces(self, scores):

        print(f"link_pieces: corners={self.corners}, no. edges={len(self.edges)}")
        # print(f"{scores=}")

        expected_pairs = dict()

        # define expected pairs for the 1026 piece puzzle
        # automagically so we can identify problems as they occur when
        # attemping to solve it programmatically
        if True and len(self.pieces) == 1026:
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
        if True and len(self.pieces) == 1026:
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

        def raftinator_compute_alignment(dst_label, dst_desc, src_label, src_desc):

            r = self.raftinator
            
            Feature = puzzler.raft.Feature
            edge_pair = (Feature(dst_label, 'edge', dst_desc[0]), Feature(src_label, 'edge', src_desc[0]))
            tab_pair = (Feature(dst_label, 'tab', dst_desc[1]), Feature(src_label, 'tab', src_desc[1]))

            dst_raft = r.factory.make_raft_for_piece(dst_label)
            src_raft = r.factory.make_raft_for_piece(src_label)

            src_coord = r.aligner.rough_align_edge_and_tab(dst_raft, src_raft, edge_pair, tab_pair)
            raft = r.factory.merge_rafts(puzzler.raft.RaftAlignment(dst_raft, src_raft, src_coord))
            raft = r.refine_alignment_within_raft(raft)
                    
            mse = r.get_total_error_for_raft_and_seams(raft)
            src_coord = raft.coords[src]
            src_fit_pts = (None, None)
            dst_fit_pts = (None, None)

            return (mse, src_coord, src_fit_pts, dst_fit_pts)

        scores = collections.defaultdict(dict)
        
        borders = self.corners + self.edges

        for dst in borders:

            dst_piece = self.pieces[dst]

            edge_aligner = None
            if not self.use_raftinator:
                edge_aligner = puzzler.align.EdgeAligner(dst_piece)

            for src in borders:

                # while the fit might be excellent, this would prove
                # topologically difficult
                if src == dst:
                    continue

                src_piece = self.pieces[src]

                dst_desc = self.succ[dst]
                src_desc = self.pred[src]

                # tabs have to be complementary (one indent and one
                # outdent)
                if dst_piece.tabs[dst_desc[1]].indent == src_piece.tabs[src_desc[1]].indent:
                    continue

                if self.use_raftinator:
                    scores[dst][src] = raftinator_compute_alignment(
                        dst, dst_desc, src, src_desc)
                else:
                    scores[dst][src] = edge_aligner.compute_alignment(
                        dst_desc, src_piece, src_desc)

        return scores

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
        return self.find_boundaries(successors, neighbors, nodes_on_frontier)[0]

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
        
            curr = successors[head]

            while curr not in visited: # [1]

                # [1] this condition was formerly curr != head, followed by
                #
                #   assert curr not in visited
                #
                # but this assertion got tripped on simple rafts of
                # just two pieces, perhaps uniquely because of their
                # topology.  Changing the while loop as above fixes
                # the issue, without understanding why the assert was
                # failing
                
                visited.add(curr)

                # if the current node has a neighbor (is not on the
                # frontier) then jump to the neighbor and then to its
                # successor (by definition the neighbor's neighbor would
                # take us back to curr)
                if neighbor := neighbors.get(curr):

                    assert neighbor not in visited
                    visited.add(neighbor)
                    fullpath.append(neighbor)
                    
                    curr = neighbor
                else:
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
            t = np.cross(p3 - p1, v4) / np.cross(v2, v4)
            u = np.cross(p3 - p1, v2) / np.cross(v2, v4)
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

    def __init__(self, pieces, *, expected=None):
        self.pieces = pieces
        self.geometry = None
        self.constraints = None
        self.adjacency = None
        self.frontiers = None
        self.corners = []
        self.distance_query_cache = puzzler.align.DistanceQueryCache()
        self.use_raftinator = True
        self.raftinator = puzzler.raft.Raftinator(pieces)
        self.seams = []
        self.expected_tab_matches = expected

    def solve(self):
        if self.geometry:
            self.solve_field()
        else:
            self.solve_border()

    def solve_border(self):
        
        bs = BorderSolver(self.pieces, use_raftinator=self.use_raftinator)

        scores = bs.score_matches()

        if False:
            ts = datetime.now().strftime('%Y%m%d-%H%M%S')
            path = self.next_path('border_match_scores_' + ts, 'tab')

            f = open(path, 'w', newline='')
            writer = csv.DictWriter(f, dialect='excel-tab', fieldnames='dst src align rank mse raft'.split())
            writer.writeheader()
            
            for dst, sources in scores.items():
                dst_desc = bs.succ[dst]
                rows = []
                for src, score in sources.items():
                    src_desc = bs.pred[src]
                    s = f"{dst}:{dst_desc[0]},{dst_desc[1]}={src}:{src_desc[0]},{src_desc[1]}"
                    t = f"{dst}/{dst_desc[0]}={src}/{src_desc[0]},{dst}:{dst_desc[1]}={src}:{src_desc[1]}"
                    rows.append({'dst':dst, 'src':src, 'align':s, 'raft':t, 'mse':score[0], 'rank':None})
                rows.sort(key=operator.itemgetter('mse'))
                for i, row in enumerate(rows, start=1):
                    row['rank'] = i
                writer.writerows(rows)
            writer = None
            f = None

        if False:
            coords = dict()
            coords['A22'] = Coord()
            coords['A23'] = scores['A22']['A23'][1]
            print(f"{coords=} {scores['A22']['A23']}")
            self.geometry = Geometry(0., 0., coords)
            return
        
        border = bs.link_pieces(scores)
        # print(f"{border=}")
        bs.estimate_puzzle_size(border)

        self.constraints = bs.init_constraints(border)
        self.geometry = bs.init_placement(border)

        print(f"puzzle_size: width={self.geometry.width:.1f} height={self.geometry.height:.1f}")

        # print(f"{self.constraints=}")
        # print(f"{self.geometry=}")

        self.update_adjacency()

        ts = datetime.now().strftime('%Y%m%d-%H%M%S')

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

        Feature = puzzler.raft.Feature

        def get_expected_piece_and_tabs_for_corner(corner):
            if self.expected_tab_matches is None:
                return None, tuple()
            
            # corner gets processed backward by score_corner, repeat that here
            corner_features = [Feature(i, 'tab', j) for i, j in corner[::-1]]
            expected_piece_features = [self.expected_tab_matches[i] for i in corner_features]

            expected_piece = expected_piece_features[0].piece
            expected_tabs = tuple(i.index for i in expected_piece_features)

            if all(i.piece == expected_piece for i in expected_piece_features):
                return expected_piece, expected_tabs
            return None, tuple()
        
        if self.geometry is None:
            return

        if not self.corners:
            return

        self.distance_query_cache.purge()

        fits = []
        for corner in self.corners:
            v = self.score_corner(corner)
            if False and len(v) >= 2 and v[0][1] == 'O28' and v[1][1] == 'X37':
                # HACK: force X37 to fit in place of O28
                fits.append((0.1, *v[1]))
            elif len(v) > 1:
                r = v[0][0] / v[1][0]
                fits.append((r, *v[0]))
            elif v:
                fits.append((1., *v[0]))

            s = [f"{i[1]}:{i[0]:.1f}" for i in v[:3]]

            # HACK: show the scoring on the known correct piece
            if corner == (('Y4', 3), ('Z5', 0)):
                for i in v[3:]:
                    s.append(f"{i[1]}:{i[0]:.1f}")
                    if i[1] == 'Y5':
                        break
                        
            print(f"{corner}: " + ", ".join(s))

            expected_piece, expected_tabs = get_expected_piece_and_tabs_for_corner(corner)
            if expected_piece:
                for i, fit in enumerate(v):
                    if fit[1] == expected_piece: # and fit[3] == expected_tabs:
                        break
                if i == len(v):
                    print(f">>> expected match ({expected_piece}, {expected_tabs}) not scored!")
                elif i > 0:
                    print(f">>> expected match ({expected_piece}, {expected_tabs}) found at position {i}")
                    print(v[i][1], v[i][3])

        if not fits:
            return
        
        fits.sort(key=operator.itemgetter(0))

        for i, f in enumerate(fits[:10]):
            r, mse, src_label, src_coords, src_tabs, corner = f
            angle = src_coords.angle * (180. / math.pi)
            print(f"{i}: {src_label:3s} angle={angle:+6.1f} {r=:.3f} {mse=:5.1f} {src_tabs=} {corner=}")

        _, _, src_label, coords, src_tabs, corner = fits[0]

        if self.expected_tab_matches is not None:
            corner_features = [Feature(i, 'tab', j) for i, j in corner]
            piece_features = [Feature(src_label, 'tab', i) for i in src_tabs]

            expected_piece_features = [self.expected_tab_matches[i] for i in corner_features]

            if piece_features != expected_piece_features:
                print(f"\n>>> {src_label=} {src_tabs=} {corner=}")
                a = ','.join(str(f) for f in piece_features)
                e = ','.join(str(f) for f in expected_piece_features)
                print(f"    expected={e} actual={a}\n")

        dst_raft = puzzler.raft.Raft(self.geometry.coords)
        src_raft = self.raftinator.factory.make_raft_for_piece(src_label)
        seams = self.raftinator.seamstress.trim_seams(
            self.raftinator.seamstress.seams_between_rafts(dst_raft, src_raft, coords))
        self.seams = seams
        print(f"{src_label=} MSE={self.raftinator.seamstress.cumulative_error_for_seams(seams):.3f}")
        if src_label == 'Z4':
            for i, seam in enumerate(seams):
                print(f" seam[{i}]: {seam.dst.piece=} {seam.src.piece=} {seam.error=:.3f} seam.n_points={len(seam.src.indices)}")
            
        for src_tab_no, dst in zip(src_tabs, corner):
            self.constraints.append(TabConstraint((src_label, src_tab_no), dst))

        self.geometry.coords[src_label] = coords

        self.update_adjacency()

        print(self.distance_query_cache.stats | {'cache_size': len(self.distance_query_cache.cache)})

        ts = datetime.now().strftime('%Y%m%d-%H%M%S')
        
        path = self.next_path('matches_' + ts, 'csv')
        self.save_tab_matches(path)
        
        path = self.next_path('solver_' + ts, 'json')
        save_json(path, self)

    def score_corner(self, corner):

        # print(f"score_corner: {corner=}")

        class RaftHelper:

            def __init__(self, pieces, coords, raftinator):
                self.pieces = pieces
                self.coords = coords
                self.overlaps = OverlappingPieces(pieces, coords)
                self.max_dist = 256
                self.raftinator = raftinator

            def measure_fit(self, src_label, src_coord):

                src_center = src_coord.xy
                src_radius = self.pieces[src_label].radius
                
                coords = dict()
                for label in self.overlaps(src_center, src_radius + self.max_dist).tolist():
                    coords[label] = self.coords[label]
                coords[src_label] = src_coord

                magic_corner = src_label == 'B2' and all(k in coords for k in ('A1', 'A2', 'B1'))
                if magic_corner:
                    print(f"  measure_fit: {src_label=} {src_coord=} pieces={','.join(coords.keys())}")
                    coords = {k:coords[k] for k in ('A1', 'A2', 'B1', 'B2')}

                raft = puzzler.raft.Raft(coords)

                for _ in range(1):
                    raft = self.raftinator.refine_alignment_within_raft(raft)

                return self.raftinator.get_total_error_for_raft_and_seams(raft)
                
        assert 2 == len(corner)

        corner = (corner[1], corner[0])
        dst_tabs = tuple([self.pieces[p].tabs[i].indent for p, i in corner])

        allways = []
        for src_piece in self.pieces.values():
                
            if src_piece.label in self.geometry.coords:
                continue

            n = len(src_piece.tabs)
            for i in range(n):
                prev = src_piece.tabs[i-1]
                curr = src_piece.tabs[i]

                if prev.indent != dst_tabs[0] and curr.indent != dst_tabs[1]:
                    src_tabs = ((i+n-1)%n, i)
                    allways.append((src_piece.label, src_tabs))

        # HACK: only consider pieces that could definitely go in this corner
        if False and len(self.pieces) == 1026:
            p0, p1 = corner[0][0], corner[1][0]
            m0 = re.fullmatch("([A-Z]+)(\d+)", p0)
            assert m0
            m1 = re.fullmatch("([A-Z]+)(\d+)", p1)
            assert m1
            rows = (m0[1], m1[1])
            cols = (m0[2], m1[2])
            valid = set(row + col for row in rows for col in cols)
            # print(f"HACK: {corner=} {valid=}")
            allways = [(src_label, src_tabs) for src_label, src_tabs in allways if src_label in valid]

        aligner = puzzler.align.MultiAligner(
            corner, self.pieces, self.geometry, self.distance_query_cache)

        raft_helper = None
        if self.use_raftinator:
            raft_helper = RaftHelper(self.pieces, self.geometry.coords, self.raftinator)

        fits = []

        for src_label, src_tabs in allways:

            src_piece = self.pieces[src_label]
            src_coords = aligner.compute_alignment(src_piece, src_tabs)

            if raft_helper is not None:
                mse = raft_helper.measure_fit(src_label, src_coords)
            else:
                mse = aligner.measure_fit(src_piece, src_tabs, src_coords)
            
            fits.append((mse, src_label, src_coords, src_tabs, corner))

        fits.sort(key=operator.itemgetter(0))

        return fits

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
                rows.append({'dst_label':dst[0], 'dst_tab_no':dst[1], 'src_label':src[0], 'src_tab_no':src[1], 'distance':distance})

        with open(path, 'w', newline='') as f:
            field_names = 'dst_label dst_tab_no src_label src_tab_no distance'.split()
            writer = csv.DictWriter(f, field_names)
            writer.writeheader()
            writer.writerows(rows)
                
    def update_adjacency(self):

        coords = self.geometry.coords
        axes_opt = (0., self.geometry.width, self.geometry.height, 0.)
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

        self.adjacency = adjacency
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
        
        width = o['width']
        height = o['height']
        coords = dict()
        for k, v in o['coords'].items():
            coords[k] = parse_affine_transform(v)

        return Geometry(width, height, coords)

    def parse_affine_transform(o):
        angle = o['angle']
        xy = tuple(o['xy'])
        return Coord(angle, xy)

    def parse_constraints(o):
        if o is None:
            return None

        return [parse_constraint(i) for i in o]

    def parse_constraint(o):

        if o['kind'] == 'edge':
            edge = tuple(o['edge'])
            axis = o['axis']
            return BorderConstraint(edge, axis)

        if o['kind'] == 'tab':
            a = tuple(o['a'])
            b = tuple(o['b'])
            return TabConstraint(a, b)

        assert False

    def parse_adjacency(o):

        def helper(o):
            ret = collections.defaultdict(list)
            for k, v in o.items():
                ret[k] = [tuple(i) for i in v]

            return ret

        if o is None:
            return None

        return dict((k, helper(v)) for k, v in o.items())

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
    constraints = parse_constraints(o['constraints'])
    adjacency = parse_adjacency(o['adjacency'])
    frontiers = parse_frontiers(o['frontiers'])
    corners = parse_corners(o['corners'])

    solver = PuzzleSolver(pieces)
    solver.geometry = geometry
    solver.constraints = constraints
    solver.adjacency = adjacency
    solver.frontiers = frontiers
    solver.corners = corners

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

        return {'width':geometry.width,
                'height':geometry.height,
                'coords':coords}

    def format_coord(t):
        return {'angle':t.angle, 'xy':t.xy.tolist()}

    def format_constraints(constraints):
        if constraints is None:
            return None
        return [format_constraint(i) for i in constraints]

    def format_constraint(c):
        if isinstance(c, BorderConstraint):
            return {'kind':'edge', 'edge': list(c.edge), 'axis':c.axis}
        
        if isinstance(c, TabConstraint):
            return {'kind':'tab', 'a':list(c.a), 'b':list(c.b)}

        assert False

    def format_adjacency(adjacency):
        if adjacency is None:
            return None

        return dict((k,v) for k, v in adjacency.items())

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
    o['constraints'] = format_constraints(solver.constraints)
    o['adjacency'] = format_adjacency(solver.adjacency)
    o['frontiers'] = format_frontiers(solver.frontiers)
    o['corners'] = format_corners(solver.corners)

    return json.JSONEncoder(indent=0).encode(o)
