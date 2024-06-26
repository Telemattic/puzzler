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

AffineTransform = puzzler.align.AffineTransform

@dataclass
class Geometry:
    width: float
    height: float
    coords: "dict[str,AffineTransform]"

class BorderSolver:

    def __init__(self, pieces):
        self.pieces = pieces
        self.pred = dict()
        self.succ = dict()
        self.corners = []
        self.edges = []

        for p in self.pieces.values():
            n = len(p.edges)
            if n == 0:
                continue

            if len(self.pieces) > 1000 and not re.fullmatch("([A-Z]+(1|38))|((A|AA)\d+)", p.label):
                print(f"Skipping {p.label}, not actually a border piece!")
                continue

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

        coords[start] = AffineTransform(angle)

        for prev, curr in zip(border, border[1:]):
            assert prev in coords and curr not in coords

            prev_m = coords[prev].get_transform().matrix
            curr_m = scores[curr][prev][1].get_transform().matrix

            coords[curr] = AffineTransform.invert_matrix(curr_m @ prev_m)

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

        coords = {k: AffineTransform(v.angle, v.center) for k, v in bodies.items()}
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

        print(f"link_pieces: corners={len(self.corners)}, edges={len(self.edges)}")
        # print(f"{scores=}")

        expected_pairs = dict()
        
        if len(self.edges) > 100:
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
            print(f"{expected_pairs=}")
            print(f"{len(expected_pairs)=}")
    
            assert set(expected_pairs.keys()) == set(expected_pairs.values())

            retval = [min(self.corners)]
            curr = expected_pairs[retval[0]]
            while curr != retval[0]:
                retval.append(curr)
                curr = expected_pairs[curr]

            return retval[::-1]

        border = set(self.corners + self.edges)

        pairs = dict()
        used = set()
        for dst in self.corners + self.edges:

            ss = scores[dst]
            best = min(ss.keys(), key=lambda x: ss[x][0])

            # print(f"{dst} <- {best} (mse={ss[best][0]})")

            expected = expected_pairs.get(dst)
            if expected_pairs and expected != best:
                if expected not in border:
                    print(f"--> {expected=} not in border set! skipping {dst=}")
                else:
                    print(f"{dst} <- {best} ({expected=})")
                    sss = sorted([(v[0], k) for k, v in ss.items()])
                    print(sss)
                continue

            if best in used:
                print(f"best match match for {dst} is {best} and it has already been used!")
                sss = sorted([(v[0], k) for k, v in ss.items()])
                print(sss)
                continue

            # greedily assume the best fit will be available, if it
            # isn't then we'll have to try harder (possibly *much*
            # harder)
            assert best not in used
            used.add(best)
            pairs[dst] = best

        print(f"{pairs=}")

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

    def score_matches(self):

        scores = collections.defaultdict(dict)
        
        borders = self.corners + self.edges

        for dst in borders:

            dst_piece = self.pieces[dst]
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
            centers.append(coords[k].dxdy)
            radii.append(v.radius)

        self.labels = labels
        self.centers = np.array(centers)
        self.radii = np.array(radii)

    def __call__(self, center, radius):
        dist = np.linalg.norm(center - self.centers, axis=1)
        ii = np.nonzero(self.radii + radius > dist)[0]
        return np.take(self.labels, ii)
                
class ClosestPieces:

    def __init__(self, pieces, geometry, distance_query_cache):
        self.pieces = pieces
        self.geometry = geometry
        self.overlaps = OverlappingPieces(pieces, geometry.coords)
        self.max_dist = 50
        self.distance_query_cache = distance_query_cache

        # print(f"ClosestPieces: width={self.geometry.width:.1f} height={self.geometry.height:.1f}")

    def __call__(self, src_label):

        src_piece = self.pieces[src_label]
        src_coords = self.geometry.coords[src_label]
        num_points = len(src_piece.points)
        
        ret_dist = np.full(num_points, self.max_dist)
        ret_no = np.zeros(num_points, dtype=np.int32)
        ret_labels = ['none']

        def piece_overlap(dst_label):
            dst_piece = self.pieces[dst_label]
            dst_coords = self.geometry.coords[dst_label]

            dst_dist = np.abs(
                self.distance_query_cache.query(dst_piece, dst_coords, src_piece, src_coords))

            ii = np.nonzero(dst_dist < ret_dist)[0]
            if 0 == len(ii):
                return
            
            ret_dist[ii] = dst_dist[ii]
            ret_no[ii] = len(ret_labels)
            ret_labels.append(dst_label)

        candidates = self.overlaps(src_coords.dxdy, src_piece.radius + self.max_dist).tolist()

        for dst_label in candidates:

            if dst_label != src_label:
                piece_overlap(dst_label)

        src_points = None

        def axis_overlap(xy, loc, label):

            center = src_coords.dxdy[xy]
            radius = src_piece.radius + self.max_dist
            overlaps = center - radius <= loc <= center + radius
            if not overlaps:
                return

            nonlocal src_points
            if src_points is None:
                transform = puzzler.render.Transform()
                transform.translate(src_coords.dxdy).rotate(src_coords.angle)
                src_points = transform.apply_v2(src_piece.points)
            
            dst_dist = np.abs(src_points[:,xy] - loc)
            ii = np.nonzero(dst_dist < ret_dist)[0]
            if 0 == len(ii):
                return
            
            ret_dist[ii] = dst_dist[ii]
            ret_no[ii] = len(ret_labels)
            ret_labels.append(label)

        axis_overlap(1, 0., 'axis0')
        axis_overlap(0, self.geometry.width, 'axis1')
        axis_overlap(1, self.geometry.height, 'axis2')
        axis_overlap(0, 0., 'axis3')

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
    
class ClosestPoints:

    def __init__(self, pieces, coords, kdtrees):
        self.pieces = pieces
        self.coords = coords
        self.kdtrees = kdtrees
        self.overlaps = OverlappingPieces(pieces, coords)
        self.max_dist = 50

    def __call__(self, src_label):

        # these are a function of the piece geometry and nothing
        # external perhaps?
        src_indexes = self.get_point_indexes(src_label)
        src_points = self.pieces[src_label].points[src_points]
        num_points = len(src_points)

        dst_labels = self.overlaps(src_coords.dxdy, src_piece.radius + self.max_dist)

        ret_dist = np.full(num_points, self.max_dist)
        ret_no = np.full(num_points, len(dst_labels))
        ret_indexes = np.zeros(num_points)

        for dst_no, dst_label in enumerate(dst_labels):

            if dst_label == src_label:
                continue

            xform = self.src_to_dst_transform(src_label, dst_label)
            dst_dist, dst_index = dst_kdtree.query(xform.apply_v2(src_points))

            ii = np.nonzero(dst_dist < ret_dist)
            if 0 == len(ii):
                continue
            
            ret_dist[ii] = dst_dist[ii]
            ret_no[ii] = dst_no
            ret_indexes[ii] = dst_index[ii]

        retval = dict()
        for dst_no, dst_label in enumerate(dst_labels):

            ii = np.nonzero(ret_no == dst_no)
            if 0 == len(ii):
                continue

            retval[(src_label, dst_label)] = (src_indexes[ii], ret_indexes[ii])

        return retval
    
class AdjacencyComputer:

    def __init__(self, pieces, constraints, geometry, distance_query_cache):
        self.pieces = pieces
        self.constraints = constraints
        self.geometry = geometry

        self.border_constraints = collections.defaultdict(list)
        self.tab_constraints = collections.defaultdict(list)

        for c in self.constraints:
            if isinstance(c, BorderConstraint):
                self.border_constraints[c.edge[0]].append(c)
            elif isinstance(c, TabConstraint):
                self.tab_constraints[c.a[0]].append(c)
                self.tab_constraints[c.b[0]].append(TabConstraint(c.b, c.a))

        self.closest = ClosestPieces(self.pieces, self.geometry, distance_query_cache)

    def compute_adjacency(self, label):

        return self.closest(label)
        
        retval = []
        
        p = self.pieces[label]

        for c in self.border_constraints[label]:
            
            edge = p.edges[c.edge[1]]
            axis = c.axis
            span = edge.fit_indexes
            
            retval.append((span, axis))

        return (retval, self.closest(label))

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

    def find_frontiers(self, successors, neighbors, nodes_on_frontier):

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
            while curr != head:

                assert curr not in visited
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

    def __init__(self, pieces, geometry):
        self.pieces = pieces
        self.geometry = geometry

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

    def find_interesting_corners(self, frontier):

        tabs = self.find_tabs(frontier)
        dirs = []
        for tab in tabs:
            p, v = self.get_tab_center_and_direction(tab)
            t = self.geometry.coords[tab[0]].get_transform()
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

    def __init__(self, pieces, geometry=None):
        self.pieces = pieces
        self.geometry = geometry
        self.constraints = None
        self.adjacency = None
        self.frontiers = None
        self.corners = []
        self.distance_query_cache = puzzler.align.DistanceQueryCache()

    def solve(self):
        if self.geometry:
            self.solve_field()
        else:
            self.solve_border()

    def solve_border(self):
        
        bs = BorderSolver(self.pieces)

        scores = bs.score_matches()

        if False:
            coords = dict()
            coords['A22'] = AffineTransform()
            coords['A23'] = scores['A22']['A23'][1]
            print(f"{coords=} {scores['A22']['A23']}")
            self.geometry = Geometry(0., 0., coords)
            return
        
        border = bs.link_pieces(scores)
        print(f"{border=}")
        bs.estimate_puzzle_size(border)

        self.constraints = bs.init_constraints(border)
        self.geometry = bs.init_placement(border)

        print(f"{self.constraints=}")
        print(f"{self.geometry=}")

        self.update_adjacency()

        ts = datetime.now()

        path = self.next_path(ts.strftime('matches_%Y%m%d-%H%M%S'), 'csv')
        self.save_tab_matches(path)
        
        path = self.next_path(ts.strftime('solver_%Y%m%d-%H%M%S'), 'json')
        save_json(path, self)

    def next_path(self, fname, ext):

        dname = r'C:\temp\puzzler\align'
        i = 0
        while True:
            path = os.path.join(dname, f"{fname}_{i}.{ext}")
            if not os.path.exists(path):
                return path
            i += 1

    def solve_field(self):
        
        if self.geometry is None:
            return

        if not self.corners:
            return

        self.distance_query_cache.purge()

        fits = []
        for corner in self.corners:
            v = self.score_corner(corner)
            if len(v) > 1:
                r = v[0][0] / v[1][0]
                fits.append((r, *v[0]))
            elif v:
                fits.append((1., *v[0]))

            s = [f"{i[1]}:{i[0]:.1f}" for i in v[:3]]

            # debugging hack to show the scoring on the known correct piece
            if corner == (('Y4', 3), ('Z5', 0)):
                for i in v[3:]:
                    s.append(f"{i[1]}:{i[0]:.1f}")
                    if i[1] == 'Y5':
                        break
                        
            print(f"{corner}: " + ", ".join(s))

        if not fits:
            return
        
        fits.sort(key=operator.itemgetter(0))

        for i, f in enumerate(fits[:10]):
            r, mse, src_label, src_coords, src_tabs, corner = f
            angle = src_coords.angle * (180. / math.pi)
            print(f"{i}: {src_label:3s} angle={angle:+6.1f} {r=:.3f} {mse=:5.1f} {src_tabs=} {corner=}")

        _, _, src_label, coords, src_tabs, corner = fits[0]
            
        for src_tab_no, dst in zip(src_tabs, corner):
            self.constraints.append(TabConstraint((src_label, src_tab_no), dst))

        self.geometry.coords[src_label] = coords

        self.update_adjacency()

        print(self.distance_query_cache.stats | {'cache_size': len(self.distance_query_cache.cache)})

        ts = datetime.now()
        
        path = self.next_path(ts.strftime('matches_%Y%m%d-%H%M%S'), 'csv')
        self.save_tab_matches(path)
        
        path = self.next_path(ts.strftime('solver_%Y%m%d-%H%M%S'), 'json')
        save_json(path, self)

    def score_corner(self, corner):

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

        aligner = puzzler.align.MultiAligner(
            corner, self.pieces, self.geometry, self.distance_query_cache)

        fits = []

        for src_label, src_tabs in allways:

            src_piece = self.pieces[src_label]
            src_coords = aligner.compute_alignment(src_piece, src_tabs)
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
            tab_xy += [xy for xy in v.get_transform().apply_v2(centers)]

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

        ac = AdjacencyComputer(self.pieces, [], self.geometry, self.distance_query_cache)
        
        adjacency = dict()
        for label in self.geometry.coords:
            adjacency[label] = ac.compute_adjacency(label)

        successors, neighbors, nodes_on_frontier = ac.compute_successors_and_neighbors(adjacency)

        frontiers, fullpaths = ac.find_frontiers(successors, neighbors, nodes_on_frontier)
        
        def flatten(i):
            a, b = i
            return f"{a}:{b[0]}-{b[1]}"

        def flatten_dict(d):
            return {flatten(k): flatten(v) for k, v in d.items()}

        def flatten_list(l):
            return [flatten(i) for i in l]

        fe = FrontierExplorer(self.pieces, self.geometry)
        corners = []
        for f in frontiers:
            corners += fe.find_interesting_corners(f)
        good_corners = []
        for (s, t, u), tab0, tab1 in corners:
            is_interesting = abs(s) < .5 and 50 < t < 1000 and 50 < u < 1000
            if is_interesting:
                good_corners.append((tab0, tab1))

        self.adjacency = adjacency
        self.frontiers = frontiers
        self.corners = good_corners

        return

        with open(r'C:\temp\puzzler\update_adjacency.txt','a') as f:
            print(f"successors={flatten_dict(successors)}", file=f)
            print(f"neighbors={flatten_dict(neighbors)}", file=f)
            print(f"nodes_on_frontier={flatten_list(nodes_on_frontier)}", file=f)
            for i, j in enumerate(frontiers):
                print(f"frontiers[{i}]={flatten_list(j)}", file=f)
            for i, j in enumerate(fullpaths):
                print(f"fullpaths[{i}]={flatten_list(j)}", file=f)
            print(f"corners={corners}", file=f)
            print(f"good_corners={good_corners}", file=f)

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
        dxdy = tuple(o['dxdy'])
        return puzzler.align.AffineTransform(angle, dxdy)

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

        coords = dict((k, format_affine_transform(v))
                      for k, v in geometry.coords.items())

        return {'width':geometry.width,
                'height':geometry.height,
                'coords':coords}

    def format_affine_transform(t):
        return {'angle':t.angle, 'dxdy':t.dxdy.tolist()}

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
