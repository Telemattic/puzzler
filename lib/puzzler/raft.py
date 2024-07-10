import puzzler
import collections
import numpy as np
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
    
Frontier = Sequence[Feature]
Frontiers = Sequence[Frontier]

FeaturePair = Tuple[Feature,Feature]
FeaturePairs = Sequence[FeaturePair]

@dataclass
class Raft:

    coords: Coords
    frontiers: Frontiers

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
        self.distance_query_cache = puzzler.align.DistanceQueryCache()
        
    def compute_frontiers(self, coords: Coords) -> Frontiers:
        boundaries = self.compute_boundaries(coords)
        return [self.compute_frontier_for_boundary(i) for i in boundaries]

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

    def make_raft_for_piece(self, src: str) -> Raft:
        
        coords = {src: Coord(0.,(0.,0.))}
        frontiers = self.compute_frontiers(coords)
        return Raft(coords, frontiers)

    def transform_coords(self, coords: Coords, xform: Coord) -> Coords:
        def helper(coord):
            return Coord.from_matrix(xform.matrix @ coord.matrix)
        return dict((k, helper(v)) for k, v in coords.items())

    def merge_rafts(self, alignment: RaftAlignment) -> Raft:

        dst_raft = alignment.dst
        src_raft = alignment.src

        coords = dst_raft.coords | self.transform_coords(src_raft.coords, alignment.src_coord)
        frontiers = self.compute_frontiers(coords)
        return Raft(coords, frontiers)

class FeatureHelper:

    def __init__(self, pieces: Pieces, coords: Coords):
        self.pieces = pieces
        self.coords = coords

    def get_edge_points(self, f: Feature) -> np.ndarray:
        
        assert f.kind == 'edge'
        
        coord = self.coords[f.piece]
        edge = self.pieces[f.piece].edges[f.index]

        return coord.xform.apply_v2(edge.line.pts)

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

class RaftAligner:

    def __init__(self, pieces: Pieces, dst_raft: Raft) -> None:
        self.pieces = pieces
        self.dst_raft = dst_raft

    def rough_align(self, src_raft: Raft, feature_pairs: FeaturePairs) -> Optional[Coord]:

        a, b = self.choose_feature_pairs_for_alignment(feature_pairs)

        if is_edge_pair(a) and is_tab_pair(b):
            return self.rough_align_edge_tab(src_raft, a, b)

        if is_tab_pair(a) and is_tab_pair(b):
            return self.rough_align_tab_tab(src_raft, a, b)

        return None

    def feature_helper(self, raft: Raft) -> FeatureHelper:
        return FeatureHelper(self.pieces, raft.coords)

    def rough_align_edge_and_tab(
            self,
            src_raft: Raft,
            edge_pair: FeaturePair,
            tab_pair: FeaturePair
    ) -> Coord:

        dst_line = self.feature_helper(self.dst_raft).get_edge_points(edge_pair[0])
        src_line = self.feature_helper(src_raft).get_edge_points(edge_pair[1])

        dst_edge_angle = np.arctan2(dst_line[1] - dst_line[0])
        src_edge_angle = np.arctan2(src_line[1] - src_line[0])

        src_coord = AffineTransform(dst_edge_angle - src_edge_angle)
        src_point = src_coords.get_transform().apply_v2(src_line[0])
        src_coord.dxdy = puzzler.math.vector_to_line(src_point, dst_line)

        dst_center = dst.get_tab_center(tab_pair[0])
        
        src_center = src.get_tab_center(tab_pair[1])
        src_center = src_coords.get_transform().apply_v2(src_center)

        dst_edge_vec = puzzler.math.unit_vector(dst_line[1]-dst_line[0])
        d = np.dot(dst_edge_vec, (dst_center - src_center))

        src_coord.dxdy = src_coord.dxdy + dst_edge_vec * d

        return src_coord
    
    def rough_align_single_tab(
            self,
            src_raft: Raft,
            tab_pair: FeaturePair
    ) -> Coord:

        dst_points = self.feature_helper(self.dst_raft).get_tab_points(tab_pair[0])
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
            src_raft: Raft,
            tab_pairs: FeaturePairs
    ) -> Coord:

        assert 0 < len(tab_pairs)
        
        if len(tab_pairs) == 1:
            return self.rough_align_single_tab(src_raft, tab_pairs[0])

        dst_helper = self.feature_helper(self.dst_raft)
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

    def refine_alignment_within_raft(self, raft: Raft, seams: Sequence[Seam]):

        verbose = False

        icp = puzzler.icp.IteratedClosestPoint()

        pieces_with_seams = set(s.src.piece for s in seams) | set(s.dst.piece for s in seams)

        fixed_piece = None
        for piece, coord in raft.coords.items():
            if coord.angle == 0. and np.all(coord.xy == 0.):
                fixed_piece = piece

        if fixed_piece is None:
            fixed_piece = next(iter(pieces_with_seams))

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

        return Raft(coords, raft.frontiers)

class RaftSeamstress:

    def __init__(self, pieces):
        self.pieces = pieces
        self.stride = 10
        self.close_cutoff = 10
        self.medium_cutoff = 48

    def seams_within_raft(self, raft):

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

    def trim_seams(self, seams):

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

    def seam_between_pieces(self, dst_label, dst_coord, src_label, src_coord):

        src_piece = self.pieces[src_label]
        src_indices = np.arange(0, len(src_piece.points), self.stride)

        src_points = src_coord.xform.apply_v2(src_piece.points[src_indices])

        dst_inverse_xform = (puzzler.render.Transform()
                             .rotate(-dst_coord.angle)
                             .translate(-dst_coord.xy))

        src_points_in_dst_frame = dst_inverse_xform.apply_v2(src_points)

        distance, dst_indices = self.get_kdtree(dst_label).query(src_points_in_dst_frame)
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

    def get_kdtree(self, label):
        piece = self.pieces[label]
        return scipy.spatial.KDTree(piece.points)

    def compute_normals(self, points, indices):
        return puzzler.align.NormalsComputer()(points, indices)
