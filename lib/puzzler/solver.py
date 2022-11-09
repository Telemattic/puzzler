import puzzler
import collections
import itertools
import math
import numpy as np
import operator
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
    edge: tuple[str,int]
    axis: int

@dataclass
class TabConstraint:
    """TabConstraint associates a pair of tabs, each identified by a (piece_label, tab_no) tuple"""
    a: tuple[str,int]
    b: tuple[str,int]

AffineTransform = puzzler.align.AffineTransform

@dataclass
class Geometry:
    width: float
    height: float
    coords: dict[str,AffineTransform]

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
            icp.make_axis(np.array((0, -1), dtype=np.float), 0., True),
            icp.make_axis(np.array((1, 0), dtype=np.float)),
            icp.make_axis(np.array((0, 1), dtype=np.float)),
            icp.make_axis(np.array((-1, 0), dtype=np.float), 0., True)
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

        return Geometry(0., 0., coords)
    
    def estimate_puzzle_size(self, border):

        axis = 3
        size = np.zeros(4, dtype=np.float)
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

        pairs = dict()
        used = set()
        for dst in self.corners + self.edges:

            ss = scores[dst]
            best = min(ss.keys(), key=lambda x: ss[x][0])

            # print(f"{dst} <- {best} (mse={ss[best][0]})")

            # greedily assume the best fit will be available, if it
            # isn't then we'll have to try harder (possibly *much*
            # harder)
            assert best not in used
            used.add(best)
            pairs[dst] = best

        # make sure the border pieces for a single ring
        visited = set()
        curr = next(iter(pairs.keys()))
        while curr not in visited:
            visited.add(curr)
            curr = pairs[curr]

        assert visited == used == set(pairs.keys())

        retval = ["H1"]
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

    def __init__(self, pieces, coords):
        self.pieces = pieces
        self.coords = coords
        self.overlaps = OverlappingPieces(pieces, coords)
        self.max_dist = 50

    def __call__(self, src_label):

        src_piece = self.pieces[src_label]
        src_coords = self.coords[src_label]
        num_points = len(src_piece.points)
        
        dst_labels = self.overlaps(src_coords.dxdy, src_piece.radius + self.max_dist)

        ret_dist = np.full(num_points, self.max_dist)
        ret_no = np.full(num_points, len(dst_labels))

        for dst_no, dst_label in enumerate(dst_labels):

            if dst_label == src_label:
                continue

            dst_piece = self.pieces[dst_label]
            dst_coords = self.coords[dst_label]

            di = puzzler.align.DistanceImage(dst_piece)
            
            transform = puzzler.render.Transform()
            transform.rotate(-dst_coords.angle).translate(-dst_coords.dxdy)
            transform.translate(src_coords.dxdy).rotate(src_coords.angle)

            src_points = transform.apply_v2(src_piece.points)
            dst_dist = di.query(src_points)

            ii = np.nonzero(dst_dist < ret_dist)[0]
            if 0 == len(ii):
                continue
            
            ret_dist[ii] = dst_dist[ii]
            ret_no[ii] = dst_no

        retval = dict()
        for dst_no, dst_label in enumerate(dst_labels):

            ii = np.nonzero(ret_no == dst_no)[0]
            if 0 == len(ii):
                continue

            ranges = []
            for key, group in itertools.groupby(enumerate(ii), lambda x: x[0] - x[1]):
                group = list(map(operator.itemgetter(1), group))
                ranges.append((group[0], group[-1]))

            retval[(src_label, dst_label)] = ranges

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

    def __init__(self, pieces, constraints, geometry):
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

        self.closest = ClosestPieces(self.pieces, self.geometry.coords)

    def compute_adjacency(self, label):
        
        retval = []
        
        p = self.pieces[label]

        for c in self.border_constraints[label]:
            
            edge = p.edges[c.edge[1]]
            axis = c.axis
            span = edge.fit_indexes
            
            retval.append((span, axis))

        return (retval, self.closest(label))
