import puzzler
import collections
from typing import Iterable, Mapping, NamedTuple, Sequence
from dataclasses import dataclass

Piece = puzzler.puzzle.Puzzle.Piece
Pieces = Mapping[str,Piece]

Coord = puzzler.align.Coord
Coords = Mapping[str,Coord]

Feature = NamedTuple('Feature', ['piece', 'kind', 'index'])
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
    feature_pairs: Sequence[Tuple[int,int]]
    src_coord: Coord
    error: float
    other_stuff_to_visualize_algorithm: Any

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
        for segment in boundary:
            frontier += compute_ordered_features_for_segment(*segment)
        return frontier

    def compute_ordered_features_for_segment(self, label, a, b):

        p = self.pieces[label]
        n = len(piece.points)
        segment = RingRange(a, b, n)
        
        def midpoint(indexes):
            i, j = indexes
            if i < j:
                return (i + j) // 2
            return ((i + j + n) // 2) % n

        def is_edge_included(edge):
            return midpoint(edge.fit_indexes) in segment

        def is_tab_included(tab):
            return midpoint(tab.tangent_indexes) in segment

        def position_in_ring(f):
            if f.kind == 'tab':
                return midpoint(p.tabs[f.index].tangent_indexes)
            else:
                return midpoint(p.edges[f.index].fit_indexes)

        features = []
        
        for i, edge in enumerate(p.edges):
            if is_edge_included(edge):
                features.append(Feature(label, 'edge', i))
                
        for i, tab in enumerate(p.tabs):
            if is_tab_included(tab):
                features.append(Feature(label, 'tab', i))

        return sorted(features, key=position_in_ring)

    def make_raft_for_piece(self, src: str) -> Raft:
        
        coords = {src: Coord(0.,0.,0.)}
        frontiers = self.compute_frontiers(coords)
        return Raft(coords, frontiers)

    def merge_rafts(self, alignment: RaftAlignment) -> Raft:

        dst_raft = alignment.dst
        src_raft = alignment.src

        coords = dst_raft.coords | self.transform_coords(src_raft, alignment.src_coord)
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

        r, x, y = compute_rigid_transform(src_points_rotated, dst_points)
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

        dst_points = np.array(dst_helper.get_tab_center(i) for i, _ in tab_pairs)
        src_points = np.array(src_helper.get_tab_center(j) for _, j in tab_pairs)
        
        dst_vec = dst_points[-1] - dst_points[0]
        dst_angle = np.arctan2(dst_vec[1], dst_vec[0])

        src_vec = src_points[-1] - src_points[0]
        src_angle = np.arctan2(src_vec[1], src_vec[0])

        src_points_rotated = Coord(dst_angle-src_angle).xform.apply_v2(src_points)
        
        r, x, y = compute_rigid_transform(src_points_rotated, dst_points)
        r += dst_angle - src_angle

        return Coord(r, (x,y))
