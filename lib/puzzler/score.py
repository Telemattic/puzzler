import puzzler
import shapely

Coord = puzzler.align.Coord

class ScoreComputer:

    def __init__(self, pieces):
        self.pieces = pieces

    def score(self, dst_label, dst_coord, src_label, src_coord):

        dst_piece = self.pieces[dst_label]
        src_piece = self.pieces[src_label]

        dst_points = dst_coord.xform.apply_v2(dst_piece.points)
        dst_poly = shapely.polygons(shapely.LinearRing(dst_points))

        src_points = src_coord.xform.apply_v2(src_piece.points)
        src_poly = shapely.polygons(shapely.LinearRing(src_points))

        isect_poly = shapely.intersection(dst_poly, src_poly)

        print(f"{shapely.area(dst_poly)=:.1f} {shapely.area(src_poly)=:.1f} {shapely.area(isect_poly)=:.1f}")

        return isect_poly
        
