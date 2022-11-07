import puzzler
import collections
import numpy as np
from dataclasses import dataclass

@dataclass
class BorderConstraint:
    """BorderConstraint associates an edge, identified by a (label, edge_no) tuple,
with an axis"""
    edge: tuple[str,int]
    axis: int

@dataclass
class TabConstraint:
    """TabConstraint associates a pair of tabs, each identified by a (piece_label, tab_no) tuple"""
    tabs: list[tuple[str,int]]

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

            constraints.append(EdgeConstraint((label, self.succ[label][0]), axis))
            if self.is_corner(label):
                axis = (axis + 1) % 4
                constraints.append(EdgeConstraint((label, self.pred[label][0]), axis))

        for a, b in pairwise_circular(border):
            constraints.append(TabConstraint([(a, self.pred[a][1]), (b, self.succ[b][1])]))

        return constraints

    def init_placement(self, border, scores):

        coords = dict()
        start = border[0] # "I1"

        p = self.pieces[start]
        e = p.edges[self.pred[start][0]]
        v = e.line.pts[0] - e.line.pts[1]
        angle = -np.arctan2(v[1], v[0])

        coords[start] = AffineTransform(angle)
        
        for prev, curr in itertools.pairwise(border):
            assert prev in coords and curr not in coords

            prev_m = coords[prev].get_transform().matrix
            curr_m = scores[prev][curr][1].get_transform().matrix

            coords[curr] = AffineTransform.invert_matrix(prev_m @ curr_m)

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
