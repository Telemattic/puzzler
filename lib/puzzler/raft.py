import puzzler
import collections
import functools
import itertools
import operator
import numpy as np
import re
import scipy
from typing import Any, Iterable, Mapping, NamedTuple, Optional, Sequence, Tuple
from dataclasses import dataclass

Piece = 'puzzler.puzzle.Puzzle.Piece'
Pieces = Mapping[str,Piece]

Coord = puzzler.align.Coord
Coords = Mapping[str,Coord]

Boundary = Tuple[str,Tuple[int,int]]
Boundaries = Sequence[Sequence[Boundary]]

class Feature(NamedTuple):
    piece: str
    kind: str
    index: int

    def __str__(self):
        x = self.kind
        if x == 'edge':
            x = '/'
        elif x == 'tab':
            x = ':'
        else:
            x = '{' + x + '}'
        return self.piece + x + str(self.index)

Frontier = Sequence[Feature]
Features = Sequence[Feature]
Frontiers = Sequence[Frontier]

FeaturePair = Tuple[Feature,Feature]
FeaturePairs = Sequence[FeaturePair]

AxisFeatures = Mapping[int,Frontier]

@dataclass
class Raft:

    coords: Coords

@dataclass
class RaftAlignment:

    dst: Raft
    src: Raft
    # feature_pairs: Sequence[Tuple[int,int]]
    src_coord: Coord
    # error: float
    # other_stuff_to_visualize_algorithm: Any

class RaftFactory:

    def __init__(self, pieces: Pieces) -> None:
        self.pieces = pieces
        self.distance_query_cache = puzzler.align.DistanceQueryCache(purge_interval=10_000)
        
    def make_raft_for_piece(self, src: str) -> Raft:
        
        coords = {src: Coord(0.,(0.,0.))}
        return Raft(coords)

    @staticmethod
    def transform_coords(coords: Coords, xform: Coord) -> Coords:
        def helper(coord):
            return Coord.from_matrix(xform.matrix @ coord.matrix)
        return dict((k, helper(v)) for k, v in coords.items())

    def merge_rafts(self, alignment: RaftAlignment) -> Raft:

        dst_raft = alignment.dst
        src_raft = alignment.src

        coords = dst_raft.coords | self.transform_coords(src_raft.coords, alignment.src_coord)
        return Raft(coords)

class RaftFeatures(NamedTuple):
    tabs: Frontiers
    axes: Mapping[int,Frontier]

class RaftFeaturesComputer:

    def __init__(self, pieces: Pieces):
        self.pieces = pieces
        self.distance_query_cache = puzzler.align.DistanceQueryCache(purge_interval=10_000)

    def compute_features(self, coords: Coords) -> RaftFeatures:

        tabs = []
        edges = []
        for frontier in self.compute_frontiers(coords):
            if any(f.kind == 'edge' for f in frontier):
                edges.append(frontier)
            if any(f.kind == 'tab' for f in frontier):
                tabs.append(frontier)

        if len(edges) > 1:
            raise ValueError("all edges should be contained in a single frontier!")

        axes = self.split_frontier_into_axes(edges[0], coords) if edges else dict()

        return RaftFeatures(tabs, axes)

    def split_frontier_into_axes(self, frontier: Frontier, coords: Coords) -> Frontiers:

        edges = []
        
        for k, g in itertools.groupby(frontier, key=operator.attrgetter('kind')):
            if k == 'edge':
                edges += list(g)

        helper = FeatureHelper(self.pieces, coords)
        
        axes = [helper.get_edge_unit_vector(edges[0])]
        for _ in range(3):
            x, y = axes[-1]
            axes.append(np.array((-y, x)))

        axes = np.array(axes)

        axis_nos = []
        for f in edges:
            v = helper.get_edge_unit_vector(f)
            i = np.argmax(np.sum(axes * v, axis=1))
            axis_nos.append((i, f))

        # rotate the edge so we don't split an axis if we can avoid it
        if len(axis_nos) > 1 and axis_nos[0][0] == axis_nos[-1][0]:
            for i, axis_no in enumerate(axis_nos):
                if axis_no[0] != axis_nos[0][0]:
                    break
            axis_nos = axis_nos[i:] + axis_nos[:i]

        axes = dict()
        for axis_no, group in itertools.groupby(axis_nos, key=operator.itemgetter(0)):
            if axis_no in axes:
                raise ValueError("axis order is nonsense, indicative of bad edge labeling")
            axes[axis_no] = [f for _, f in group]

        return axes

    def compute_frontiers(self, coords: Coords) -> Frontiers:

        return [self.compute_frontier_for_boundary(i) for i in self.compute_boundaries(coords)]

    def compute_frontier_tabs_for_piece(self, coords: Coords, label: str) -> Features:

        closest = puzzler.solver.ClosestPieces(
            self.pieces, coords, None, self.distance_query_cache)

        features = []
        for a, b in closest(label).get('none', []):
            for f in self.compute_ordered_features_for_segment(label, a, b):
                if f.kind == 'tab':
                    features.append(f)

        return features

    def compute_boundaries(self, coords: Coords) -> Boundaries:
        
        closest = puzzler.solver.ClosestPieces(
            self.pieces, coords, None, self.distance_query_cache)
        adjacency = dict((i, closest(i)) for i in coords)

        bc = puzzler.solver.BoundaryComputer(self.pieces)
        return bc.find_boundaries_from_adjacency(adjacency)

    def compute_frontier_for_boundary(self, boundary) -> Frontier:

        frontier = []
        for label, (a, b) in boundary:
            frontier += self.compute_ordered_features_for_segment(label, a, b)
        return frontier

    def compute_ordered_features_for_segment(self, label, a, b):

        p = self.pieces[label]
        n = len(p.points)
        segment = puzzler.commands.align.RingRange(a, b, n)
        
        def midpoint(indexes):
            i, j = indexes
            if i < j:
                m = (i + j) // 2
            else:
                m = ((i + j + n) // 2) % n
            return m

        def is_edge_included(edge):
            return midpoint(edge.fit_indexes) in segment

        def is_tab_included(tab):
            return midpoint(tab.tangent_indexes) in segment

        def position_in_ring(f):
            if f.kind == 'tab':
                m = midpoint(p.tabs[f.index].tangent_indexes)
            else:
                m = midpoint(p.edges[f.index].fit_indexes)
            # adjust the midpoint so that points that occur later in
            # the segment always have a higher reported midpoint, even
            # if the segment "wraps" from high indices to low indices.
            # An alternative would be to report these points as
            # relative offsets within the segment, i.e. perform all of
            # these calculations and then subtract a, so it should
            # always be the case that 0 <= m < len(segment)
            if a > b and m < b:
                m += n
            return m
            
        features = []
        
        for i, edge in enumerate(p.edges):
            if is_edge_included(edge):
                features.append(Feature(label, 'edge', i))
                
        for i, tab in enumerate(p.tabs):
            if is_tab_included(tab):
                features.append(Feature(label, 'tab', i))

        return sorted(features, key=position_in_ring)

class FeatureHelper:

    def __init__(self, pieces: Pieces, coords: Coords):
        self.pieces = pieces
        self.coords = coords

    def get_edge_points(self, f: Feature) -> np.ndarray:
        
        assert f.kind == 'edge'
        
        coord = self.coords[f.piece]
        edge = self.pieces[f.piece].edges[f.index]

        return coord.xform.apply_v2(edge.line.pts)

    def get_edge_unit_vector(self, f: Feature) -> np.ndarray:

        pts = self.get_edge_points(f)
        return puzzler.math.unit_vector(pts[1] - pts[0])

    def get_edge_angle(self, f: Feature) -> float:
        
        pts = self.get_edge_points(f)
        vec = points[1] - points[0]
        return np.arctan2(vec[1], vec[0])

    def get_tab_center(self, f: Feature) -> np.ndarray:
        
        assert f.kind == 'tab'
        
        coord = self.coords[f.piece]
        tab = self.pieces[f.piece].tabs[f.index]

        return coord.xform.apply_v2(tab.ellipse.center)

    def get_tab_points(self, f: Feature) -> np.ndarray:
    
        assert f.kind == 'tab'
        
        coord = self.coords[f.piece]
        piece = self.pieces[f.piece]
        tab = piece.tabs[f.index]
        
        ti = list(tab.tangent_indexes)
        l, r = piece.points[ti if tab.indent else ti[::-1]]
        c = tab.ellipse.center

        return coord.xform.apply_v2(np.array((l, r, c)))

class Stitches(NamedTuple):
    piece: str
    indices: np.ndarray
    points: np.ndarray
    normals: np.ndarray

class Seam(NamedTuple):
    dst: Stitches
    src: Stitches
    error: float

Seams = Sequence[Seam]

class RaftAligner:

    def __init__(self, pieces: Pieces) -> None:
        self.pieces = pieces

    def rough_align(self, dst_raft: Raft, src_raft: Raft, feature_pairs: FeaturePairs) -> Coord:

        if len(feature_pairs) == 0:
            raise ValueError("no features to align")

        if all(self.is_tab_pair(i) for i in feature_pairs):
            return self.rough_align_multiple_tabs(dst_raft, src_raft, feature_pairs)

        if len(feature_pairs) == 2:
            a, b = feature_pairs
            if self.is_edge_pair(a) and self.is_tab_pair(b):
                return self.rough_align_edge_and_tab(dst_raft, src_raft, a, b)

        raise ValueError("don't know how to align features")

    def is_edge_pair(self, p: FeaturePair) -> bool:
        return p[0].kind == 'edge' and p[1].kind == 'edge'

    def is_tab_pair(self, p: FeaturePair) -> bool:
        return p[0].kind == 'tab' and p[1].kind == 'tab'

    def feature_helper(self, raft: Raft) -> FeatureHelper:
        return FeatureHelper(self.pieces, raft.coords)

    def rough_align_edge_and_tab(
            self,
            dst_raft: Raft,
            src_raft: Raft,
            edge_pair: FeaturePair,
            tab_pair: FeaturePair
    ) -> Coord:

        dst_helper = self.feature_helper(dst_raft)
        src_helper = self.feature_helper(src_raft)

        dst_line = dst_helper.get_edge_points(edge_pair[0])
        src_line = src_helper.get_edge_points(edge_pair[1])

        dst_edge_vec = dst_line[1] - dst_line[0]
        src_edge_vec = src_line[1] - src_line[0]

        dst_edge_angle = np.arctan2(dst_edge_vec[1], dst_edge_vec[0])
        src_edge_angle = np.arctan2(src_edge_vec[1], src_edge_vec[0])

        src_coord = Coord(dst_edge_angle - src_edge_angle)
        src_point = src_coord.xform.apply_v2(src_line[0])
        src_coord.xy = puzzler.math.vector_to_line(src_point, dst_line)

        dst_tab_center = dst_helper.get_tab_center(tab_pair[0])
        
        src_tab_center = src_helper.get_tab_center(tab_pair[1])
        src_tab_center = src_coord.xform.apply_v2(src_tab_center)

        dst_edge_vec = puzzler.math.unit_vector(dst_line[1]-dst_line[0])
        d = np.dot(dst_edge_vec, (dst_tab_center - src_tab_center))

        src_coord.xy = src_coord.xy + dst_edge_vec * d

        return src_coord
    
    def rough_align_single_tab(
            self,
            dst_raft: Raft,
            src_raft: Raft,
            tab_pair: FeaturePair
    ) -> Coord:

        dst_points = self.feature_helper(dst_raft).get_tab_points(tab_pair[0])
        src_points = self.feature_helper(src_raft).get_tab_points(tab_pair[1])

        dst_vec = dst_points[0] - dst_points[1] + dst_points[2] - dst_points[1]
        dst_angle = np.arctan2(dst_vec[1], dst_vec[0])

        src_vec = src_points[0] - src_points[1] + src_points[2] - src_points[1]
        src_angle = np.arctan2(src_vec[1], src_vec[0])

        src_points_rotated = Coord(dst_angle-src_angle).xform.apply_v2(src_points)

        r, x, y = puzzler.align.compute_rigid_transform(src_points_rotated, dst_points)
        r += dst_angle - src_angle

        return Coord(r, (x,y))
        
    def rough_align_multiple_tabs(
            self,
            dst_raft: Raft,
            src_raft: Raft,
            tab_pairs: FeaturePairs
    ) -> Coord:

        assert 0 < len(tab_pairs)
        
        if len(tab_pairs) == 1:
            return self.rough_align_single_tab(dst_raft, src_raft, tab_pairs[0])

        dst_helper = self.feature_helper(dst_raft)
        src_helper = self.feature_helper(src_raft)

        dst_points = np.array([dst_helper.get_tab_center(i) for i, _ in tab_pairs])
        src_points = np.array([src_helper.get_tab_center(j) for _, j in tab_pairs])
        
        dst_vec = dst_points[-1] - dst_points[0]
        dst_angle = np.arctan2(dst_vec[1], dst_vec[0])

        src_vec = src_points[-1] - src_points[0]
        src_angle = np.arctan2(src_vec[1], src_vec[0])

        src_points_rotated = Coord(dst_angle-src_angle).xform.apply_v2(src_points)
        
        r, x, y = puzzler.align.compute_rigid_transform(src_points_rotated, dst_points)
        r += dst_angle - src_angle

        return Coord(r, (x,y))

    def find_piece_closest_to_origin(self, raft: Raft) -> str:

        label = []
        xy = []
        for piece, coord in raft.coords.items():
            label.append(piece)
            xy.append(coord.xy)

        xy = np.array(xy)                
        dist = np.linalg.norm(xy, axis=1)
        i = np.argmin(dist)
        return label[i]

    def refine_alignment_between_rafts(self, alignment: RaftAlignment) -> RaftAlignment:

        dst_raft = alignment.dst
        src_raft = alignment.src
        src_raft_coord = alignment.src_coord

        seams = self.seamstress.trim_seams(
            self.seamstress.seams_between_rafts(dst_raft, src_raft, src_raft_coord))

        raise ValueError("oops, not implemented")

    def refine_edge_alignment_within_raft(self, raft: Raft, seams: Sequence[Seam], edges: FeaturePair, verbose=False) -> Raft:

        if len(raft.coords) != 2:
            raise ValueError("expected raft with exactly two pieces")

        dst_edge, src_edge = edges

        if verbose:
            print(f"{dst_edge=} {src_edge=}")
        
        dst_piece = dst_edge.piece

        icp = puzzler.icp.IteratedClosestPoint()
        
        pieces_with_seams = set(s.src.piece for s in seams) | set(s.dst.piece for s in seams)

        helper = FeatureHelper(self.pieces, raft.coords)

        dst_edge_points = helper.get_edge_points(dst_edge)
        dst_edge_value = puzzler.math.distance_to_line(np.array((0.,0.)), dst_edge_points)
        
        dst_edge_unit_vector = puzzler.math.unit_vector(dst_edge_points[1] - dst_edge_points[0])
        dst_edge_normal = np.array((-dst_edge_unit_vector[1], dst_edge_unit_vector[0]))

        if verbose:
            with np.printoptions(precision=4):
                print(f"{dst_edge_points=} {dst_edge_normal=} {dst_edge_value=}")

        axis = icp.make_axis(dst_edge_normal, dst_edge_value, fixed=True)

        bodies = dict()
        for piece, coord in raft.coords.items():
            bodies[piece] = icp.make_rigid_body(coord.angle, coord.xy, fixed=(piece == dst_piece))

        for i in seams:
            src_body = bodies[i.src.piece]
            if src_body.fixed:
                continue
            src_piece = self.pieces[i.src.piece]
            src_points = src_piece.points[i.src.indices]
            
            dst_body = bodies[i.dst.piece]
            dst_piece = self.pieces[i.dst.piece]
            dst_points = dst_piece.points[i.dst.indices]
            dst_normals = puzzler.align.NormalsComputer()(dst_piece.points, i.dst.indices)
            
            icp.add_body_correspondence(src_body, src_points,
                                        dst_body, dst_points, dst_normals)

            e = src_piece.edges[src_edge.index]
            icp.add_axis_correspondence(src_body, e.line.pts, axis)

        icp.solve()

        coords = dict()
        for piece, coord in raft.coords.items():
            body = bodies.get(piece)
            if body:
                coord = Coord(body.angle, body.center)
            coords[piece] = coord

        return Raft(coords)
    
    def refine_alignment_within_raft(
            self,
            raft: Raft,
            seams: Sequence[Seam],
            axis_features: Optional[Frontiers] = None) -> Raft:

        verbose = False

        icp = puzzler.icp.IteratedClosestPoint()

        pieces_with_seams = set(s.src.piece for s in seams) | set(s.dst.piece for s in seams)

        fixed_piece = None
        axes = [None] * 4

        if axis_features is None:
            fixed_piece = self.find_piece_closest_to_origin(raft)
        else:
            axes = [
                icp.make_axis(np.array((0, -1), dtype=float), 0., True),
                icp.make_axis(np.array((1, 0), dtype=float)),
                icp.make_axis(np.array((0, 1), dtype=float)),
                icp.make_axis(np.array((-1, 0), dtype=float), 0., True)
            ]

        if verbose:
            print(f"refine_alignment_within_raft: {fixed_piece=}")

        bodies = dict()
        for piece in pieces_with_seams:
            coord = raft.coords[piece]
            bodies[piece] = icp.make_rigid_body(coord.angle, coord.xy, fixed=(piece == fixed_piece))

        for i in seams:
            src_body = bodies[i.src.piece]
            if src_body.fixed:
                continue
            src_piece = self.pieces[i.src.piece]
            src_points = src_piece.points[i.src.indices]
            
            dst_body = bodies[i.dst.piece]
            dst_piece = self.pieces[i.dst.piece]
            dst_points = dst_piece.points[i.dst.indices]
            dst_normals = puzzler.align.NormalsComputer()(dst_piece.points, i.dst.indices)
            
            icp.add_body_correspondence(src_body, src_points,
                                        dst_body, dst_points, dst_normals)

        if axis_features:
            for axis_no, frontier in enumerate(axis_features):
                for f in frontier:
                    e = self.pieces[f.piece].edges[f.index]
                    icp.add_axis_correspondence(bodies[f.piece], e.line.pts, axes[axis_no])

        icp.solve()

        if verbose:
            for piece, body in bodies.items():
                old_coord = raft.coords[piece]
                new_coord = Coord(body.angle, body.center)
                with np.printoptions(precision=3):
                    print(f"{piece} {old_coord} {new_coord}")
    
        coords = dict()
        for piece, coord in raft.coords.items():
            body = bodies.get(piece)
            if body:
                coord = Coord(body.angle, body.center)
            coords[piece] = coord

        return Raft(coords)

class RaftSeamstress:

    def __init__(self, pieces):
        self.pieces = pieces
        self.stride = 10
        self.close_cutoff = 10
        self.medium_cutoff = 48

    def cumulative_error_for_seams(self, seams: Seams) -> float:
        n_points = 0
        error = 0.
        for s in seams:
            n_points += len(s.src.indices)
            error += s.error
        return error / n_points if n_points else 0.

    def seams_between_rafts(self, dst_raft: Raft, src_raft: Raft, src_raft_coord: Coord) -> Seams:

        overlaps = puzzler.solver.OverlappingPieces(self.pieces, dst_raft.coords)

        seams = []
        for src_label, src_coord in src_raft.coords.items():

            assert src_coord.angle == 0. and np.all(src_coord.xy == 0.)
            # center of the src piece in the space of the aligned rafts
            src_xy = src_raft_coord.xform.apply_v2(src_coord.xy)
            src_piece = self.pieces[src_label]
            
            for dst_label in overlaps(src_xy, src_piece.radius):
                dst_coord = dst_raft.coords[dst_label]
                # HACK: this should be the composed coordinate, but
                # we've already validated that the src_raft is just an
                # identity transform
                seam = self.seam_between_pieces(dst_label, dst_coord, src_label, src_raft_coord)
                if seam is None:
                    continue
                seams.append(seam)

        return seams

    def seams_within_raft(self, raft: Raft) -> Seams:

        overlaps = puzzler.solver.OverlappingPieces(self.pieces, raft.coords)

        seams = []
        for src_label, src_coord in raft.coords.items():
            src_piece = self.pieces[src_label]
            for dst_label in overlaps(src_coord.xy, src_piece.radius):
                if dst_label == src_label:
                    continue
                dst_coord = raft.coords[dst_label]
                seam = self.seam_between_pieces(dst_label, dst_coord, src_label, src_coord)
                if seam is None:
                    continue
                seams.append(seam)

        return seams

    def trim_seams(self, seams: Seams) -> Seams :

        def helper(src_piece, dst_seams):

            if len(dst_seams) < 2:
                return dst_seams

            min_dist = dict()
            for seam_no, seam in enumerate(dst_seams):
                distance = np.linalg.norm(seam.dst.points - seam.src.points, axis=1)
                for i, src_index in enumerate(seam.src.indices):
                    if src_index not in min_dist or min_dist[src_index][0] > distance[i]:
                        min_dist[src_index] = (distance[i], seam_no)

            by_seam = list(set() for _ in range(len(dst_seams)))
            for src_index, (_, dst_seam_no) in min_dist.items():
                by_seam[dst_seam_no].add(src_index)

            retval = []
            for seam, take_indexes in zip(dst_seams, by_seam):

                nonzero = [i for i, j in enumerate(seam.src.indices) if j in take_indexes]
                if len(nonzero) == 0:
                    continue

                dst_stitches = Stitches(seam.dst.piece,
                                        seam.dst.indices[nonzero],
                                        seam.dst.points[nonzero],
                                        seam.dst.normals[nonzero])

                src_stitches = Stitches(seam.src.piece,
                                        seam.src.indices[nonzero],
                                        seam.src.points[nonzero],
                                        seam.src.normals[nonzero])

                error = np.sum((dst_stitches.points - src_stitches.points) ** 2)

                retval.append(Seam(dst_stitches, src_stitches, error))

            return retval
                

        by_src_piece = collections.defaultdict(list)
        for s in seams:
            by_src_piece[s.src.piece].append(s)

        retval = []
        for src_piece, dst_seams in by_src_piece.items():
            retval += helper(src_piece, dst_seams)

        return retval

    def get_index_range_for_stitches(self, stitches: Stitches) -> Tuple[int, int]:

        piece = self.pieces[stitches.piece]
        n = len(piece.points)
        # sorted indices
        si = np.sort(stitches.indices)
        i = np.argmax(np.diff(si, append=si[0]+n))
        a = si[(i+1) % len(si)]
        b = si[i]
        return (a, b)

    @functools.lru_cache(maxsize=1024)
    def seam_between_pieces(self, dst_label: str, dst_coord: Coord, src_label: str, src_coord: Coord) -> Optional[Seam]:

        src_piece = self.pieces[src_label]
        src_indices = np.arange(0, len(src_piece.points), self.stride)

        src_points = src_coord.xform.apply_v2(src_piece.points[src_indices])

        dst_inverse_xform = (puzzler.render.Transform()
                             .rotate(-dst_coord.angle)
                             .translate(-dst_coord.xy))

        src_points_in_dst_frame = dst_inverse_xform.apply_v2(src_points)

        distance, dst_indices = self.get_kdtree(dst_label).query(
            src_points_in_dst_frame, distance_upper_bound=self.medium_cutoff)
        if np.all(distance > self.medium_cutoff):
            return None

        dst_piece = self.pieces[dst_label]
        dst_normals = dst_coord.xform.apply_n2(self.compute_normals(dst_piece.points, dst_indices))
        src_normals = src_coord.xform.apply_n2(self.compute_normals(src_piece.points, src_indices))

        dot_product = np.sum(dst_normals * src_normals, axis=1)

        is_close = (distance < self.close_cutoff) | ((distance < self.medium_cutoff) & (dot_product < -0.5))
        close_points = np.nonzero(is_close)
        if len(close_points[0]) == 0:
            return None

        close_dst_indices = dst_indices[close_points]
        close_dst_points = dst_coord.xform.apply_v2(dst_piece.points[close_dst_indices])
                                
        dst_stitches = Stitches(dst_label,
                                close_dst_indices,
                                close_dst_points,
                                dst_normals[close_points])

        src_stitches = Stitches(src_label,
                                src_indices[close_points],
                                src_points[close_points],
                                src_normals[close_points])

        close_distance = distance[close_points]

        error = np.sum(close_distance ** 2)

        return Seam(dst_stitches, src_stitches, error)

    @functools.lru_cache(maxsize=128)
    def get_kdtree(self, label: str) -> scipy.spatial.KDTree:
        piece = self.pieces[label]
        return scipy.spatial.KDTree(piece.points)

    def compute_normals(self, points: np.ndarray, indices: np.ndarray) -> np.ndarray:
        return puzzler.align.NormalsComputer()(points, indices)

class Raftinator:

    def __init__(self, pieces: Pieces):
        self.pieces = pieces
        self.factory = RaftFactory(pieces)
        self.aligner = RaftAligner(pieces)
        self.seamstress = RaftSeamstress(pieces)
        self.raft_error = RaftError(pieces)
        self.verbose = False

    def parse_feature(self, s) -> Feature:
        m = re.fullmatch(r"([A-Z]+\d+)([:/])(\d+)", s)
        if not m:
            raise ValueError("bad feature")
        piece, kind, index = m[1], m[2], int(m[3])
        kind = 'tab' if kind == ':' else 'edge'
        return Feature(piece, kind, index)

    def parse_feature_pair(self, s) -> FeaturePair:
        v = s.split('=')
        if len(v) != 2:
            raise ValueError("bad feature pair")

        return (self.parse_feature(v[0]), self.parse_feature(v[1]))

    def parse_feature_pairs(self, s) -> FeaturePairs:
        return [self.parse_feature_pair(i) for i in s.strip().split(',')]

    def format_feature_pair(self, feature_pair: FeaturePair) -> str:
        return str(feature_pair[0]) + '=' + str(feature_pair[1])

    def format_feature_pairs(self, feature_pairs: FeaturePairs) -> str:
        return ','.join(self.format_feature_pair(i) for i in feature_pairs)

    def get_seams_for_raft(self, raft) -> Seams:
        s = self.seamstress
        return s.trim_seams(s.seams_within_raft(raft))

    def get_cumulative_error_for_seams(self, seams) -> float:
        return self.seamstress.cumulative_error_for_seams(seams)

    def get_overlap_error_for_raft(self, raft: Raft) -> float:
        return self.raft_error.overlap_error_for_raft(raft)

    def get_total_error_for_raft_and_seams(self, raft: Raft, seams: Optional[Seams] = None) -> float:
        if seams is None:
            seams = self.get_seams_for_raft(raft)
        return self.raft_error.total_error_for_raft_and_seams(raft, seams).mse

    def align_and_merge_rafts_with_feature_pairs(self, dst_raft: Raft, src_raft: Raft, feature_pairs: FeaturePairs) -> Raft:
        
        src_coord = self.aligner.rough_align(dst_raft, src_raft, feature_pairs)
        return self.factory.merge_rafts(RaftAlignment(dst_raft, src_raft, src_coord))

    def refine_alignment_within_raft(self, raft: Raft, seams: Optional[Seams] = None) -> Raft:

        if seams is None:
            seams = self.get_seams_for_raft(raft)

        return self.aligner.refine_alignment_within_raft(raft, seams)

    def make_raft_from_string(self, s: str) -> Raft:

        feature_pairs = self.parse_feature_pairs(s)
        return self.make_raft_from_feature_pairs(feature_pairs)

    def make_raft_from_feature_pairs(self, feature_pairs: FeaturePairs) -> Raft:
        
        rafts = dict()
        next_raft_id = 0

        def next_raft_name():
            nonlocal next_raft_id
            next_raft_id += 1
            return f"raft_{next_raft_id-1}"
        
        def get_raft_for_piece(piece, make_raft_if_missing=False):
            nonlocal rafts
            for k, v in rafts.items():
                if piece in v.coords:
                    return k
            if not make_raft_if_missing:
                return None

            k = next_raft_name()
            v = self.factory.make_raft_for_piece(piece)
            rafts[k] = v

            if self.verbose:
                print(f"raftinator: {k}=Raft({piece})")
            
            return k

        def get_rafts_for_feature_pair(feature_pair, make_rafts_if_missing=False):
            
            dst_feature, src_feature = feature_pair
            dst_raft = get_raft_for_piece(dst_feature.piece, make_rafts_if_missing)
            src_raft = get_raft_for_piece(src_feature.piece, make_rafts_if_missing)

            return (dst_raft, src_raft)

        if self.verbose:
            print("raftinator:")
            for i, fp in enumerate(feature_pairs):
                print(f"  feature_pair[{i}]={fp}")

        i = 0
        new_raft = None
        while i < len(feature_pairs):

            dst_raft, src_raft = get_rafts_for_feature_pair(feature_pairs[i], True)
            if dst_raft == src_raft:
                i = i+1
                continue

            j = i+1
            while j < len(feature_pairs) and get_rafts_for_feature_pair(feature_pairs[j]) == (dst_raft, src_raft):
                j += 1

            if self.verbose:
                print(f"raftinator: {dst_raft=} {src_raft=} {i=} {j=}")

            new_raft = self.align_and_merge_rafts_with_feature_pairs(
                rafts[dst_raft], rafts[src_raft], feature_pairs[i:j])

            rafts.pop(dst_raft)
            rafts.pop(src_raft)

            new_name = next_raft_name()
            rafts[new_name] = new_raft

            if self.verbose:
                print(f"raftinator: {new_name}={dst_raft}+{src_raft}, features={feature_pairs[i:j]}")
            i = j

        assert new_raft is not None and len(rafts) == 1

        return new_raft

class FitError:

    def __init__(self, sse: float, n: int) -> None:
        self.sse = sse
        self.n = n

    @property
    def mse(self) -> float:
        return self.sse / self.n if self.n > 0 else 0.

    def __add__(self, that: 'FitError') -> 'FitError':
        return FitError(self.sse + that.sse, self.n + that.n)

    def __iadd__(self, that: 'FitError') -> None:
        self.sse += that.sse
        self.n += that.n

class RaftError:

    def __init__(self, pieces):
        self.pieces = pieces
        self.stride = 10
        self.close_cutoff = 10
        self.max_dist = 256

    def total_error_for_raft_and_seams(self, raft: Raft, seams: Seams) -> FitError:

        return self.overlap_error_for_raft(raft) + self.seam_error_for_raft(seams)

    def seam_error_for_raft(self, seams: Seams) -> FitError:

        sse = 0.
        n = 0
        for s in seams:
            sse += s.error
            n += len(s.src.indices)
        return FitError(sse, n)

    def overlap_error_for_raft(self, raft: Raft) -> FitError:
        
        overlaps = puzzler.solver.OverlappingPieces(self.pieces, raft.coords)

        sse = 0.
        num_points = 0
        
        for src_label, src_coord in raft.coords.items():
            
            src_piece = self.pieces[src_label]
            src_center = src_coord.xy
            src_radius = src_piece.radius

            src_indices = np.arange(0, len(src_piece.points), self.stride)
            src_points = src_piece.points[src_indices]
        
            for dst_label in overlaps(src_center, src_radius + self.max_dist).tolist():
                
                if dst_label == src_label:
                    continue

                dst_piece = self.pieces[dst_label]
                dst_coord = raft.coords[dst_label]
            
                transform = puzzler.render.Transform()
                transform.rotate(-dst_coord.angle).translate(-dst_coord.xy)
                transform.translate(src_coord.xy).rotate(src_coord.angle)

                di = puzzler.align.DistanceImage.Factory(dst_piece)
                distance = di.query(transform.apply_v2(src_points))

                ii = np.nonzero(distance < -self.close_cutoff)[0]
                if len(ii):
                    sse += np.sum(np.square(distance[ii]))
                    num_points += len(ii)

        return FitError(sse, num_points)
