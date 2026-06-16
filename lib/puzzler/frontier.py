import collections
import itertools
import numpy as np
import operator
import puzzler
from typing import NamedTuple

import json

class ClosestPieces:

    def __init__(self, pieces, coords, axes_opt, distance_query_cache):
        self.pieces = pieces
        self.coords = coords
        self.axes_opt = axes_opt
        self.overlaps = puzzler.solver.OverlappingPieces(pieces, coords)
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
                return len(RingRange(a, b+1, n))

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
            for prev, curr in puzzler.solver.pairwise_circular(ranges):
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

class RingRange:

    def __init__(self, a, b, n):
        self.a = a
        self.b = b
        self.n = n

    def __iter__(self):
        a, b = self.a, self.b
        return itertools.chain(range(a, self.n), range(0, b)) if a >= b else range(a, b)

    def __contains__(self, i):
        a, b = self.a, self.b
        return (a <= i < self.n or 0 <= i < b) if a >= b else (a <= i < b)

    def __len__(self):
        a, b = self.a, self.b
        return (b - a) if b > a else (b + self.n - a)
    
Feature = puzzler.raft.Feature
    
class DetailedTab(NamedTuple):
    piece: str
    index: int
    indent: bool
    tab_xy: np.array
    tab_uv: np.array

class FrontierExplorer:

    def __init__(self, pieces, coords):
        self.pieces = pieces
        self.coords = coords

    def get_detailed_tabs(self, frontier):

        retval = []
        
        for piece_label, piece_range in frontier:
            retval += self.get_detailed_tabs_for_range(piece_label, piece_range)

        return retval

    def get_detailed_tabs_for_range(self, piece_label, piece_range):
        
        p = self.pieces[piece_label]
        rr = RingRange(piece_range[0], piece_range[1], len(p.points))

        retval = []
        for i, tab in enumerate(p.tabs):
            if all(j in rr for j in tab.fit_indexes):
                tab_xy, tab_uv = self.get_tab_center_and_direction(Feature(piece_label,'tab',i))
                xform = self.coords[piece_label].xform
                tab_xy = xform.apply_v2(tab_xy)
                tab_uv = xform.apply_n2(tab_uv)
                retval.append(DetailedTab(piece_label, i, tab.indent, tab_xy, tab_uv))

        def position_in_ring(x):
            tab = p.tabs[x.index]
            begin = tab.fit_indexes[0]
            if begin < rr.a:
                begin += rr.n
            return begin

        return sorted(retval, key=position_in_ring)

    def find_interesting_corners(self, frontier):

        tabs = self.get_detailed_tabs(frontier)

        scores = []
        for i, curr in enumerate(tabs):
            prev = tabs[i-1]
            p1, v2 = prev.tab_xy, prev.tab_uv
            p3, v4 = curr.tab_xy, curr.tab_uv
            t = np.cross(p3 - p1, v4)
            u = np.cross(p3 - p1, v2)
            d = np.cross(v2, v4)
            if d != 0.:
                t /= d
                u /= d
            else:
                t = 0.
                u = 0.
            scores.append(((np.dot(v2, v4), t, u), prev, curr))
            
        return scores

    def get_tab_center_and_direction(self, tab):

        p = self.pieces[tab.piece]
        t = p.tabs[tab.index]
        v = p.points[np.array(t.tangent_indexes)] - t.ellipse.center
        v = v / np.linalg.norm(v, axis=1)
        v = np.sum(v, axis=0)
        v = v / np.linalg.norm(v)
        if not t.indent:
            v = -v
        return (t.ellipse.center, v)

def pockets_from_corners(corners, adjacency):

    Pocket = puzzler.pocket.Pocket

    good_corners = []
    for (s, t, u), tab0, tab1 in corners:
        if abs(s) < .5 and 50 < t < 1000 and 50 < u < 1000:
            good_corners.append((tab0, tab1))

    pockets = []
    for a, b in good_corners:
        tabA = Feature(a.piece, 'tab', a.index)
        tabB = Feature(b.piece, 'tab', b.index)
        common = set(adjacency[tabA.piece].keys()) & set(adjacency[tabB.piece].keys())
        common.discard('none')
        print(f"{tabA=!s} {tabB=!s} {common=}")
        if len(common) == 1:
            pockets.append(Pocket(tabA, tabB, frozenset([tabA.piece, common.pop(), tabB.piece])))

    return pockets

class FrontierJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, frozenset):
            return list(obj)
        return super().default(obj)

def frontier_experiment(pieces, raft, dqc):

    o = {}

    coords = raft.coords
    width, height = raft.size
    axes_opt = (0., width, height, 0.)
    closest_pieces = ClosestPieces(pieces, coords, axes_opt, dqc)
    adjacency = dict((i, closest_pieces(i)) for i in coords)

    o['adjacency'] = adjacency

    frontiers = BoundaryComputer(pieces).find_boundaries_from_adjacency(adjacency)

    o['frontiers'] = frontiers

    fe = FrontierExplorer(pieces, coords)
    corners = []
    for f in frontiers:
        corners += fe.find_interesting_corners(f)

    o['corners'] = corners

    good_corners = []
    for (s, t, u), tab0, tab1 in corners:
        is_interesting = abs(s) < .5 and 50 < t < 1000 and 50 < u < 1000
        if is_interesting:
            good_corners.append((tab0, tab1))

    o['good_corners'] = good_corners

    pockets = pockets_from_corners(corners, adjacency)

    o['pockets'] = pockets

    return o
