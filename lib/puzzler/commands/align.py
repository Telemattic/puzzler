import bisect
import collections
import csv
import cv2 as cv
import itertools
import math
import numpy as np
import re
import scipy
import puzzler.feature
import puzzler
import puzzler.solver

from tkinter import *
from tkinter import font
from tkinter import ttk

from dataclasses import dataclass, field

class Camera:

    def __init__(self, center, zoom, viewport):
        self._center = center.copy()
        self._zoom = zoom
        self._viewport = viewport
        self._matrix = None
        self.__update_matrix()

    @property
    def center(self):
        return self._center

    @center.setter
    def center(self, v):
        self._center = v.copy()
        self.__update_matrix()

    @property
    def zoom(self):
        return self._zoom

    @zoom.setter
    def zoom(self, v):
        self._zoom = v
        self.__update_matrix()

    @property
    def viewport(self):
        return self._viewport

    @viewport.setter
    def viewport(self, v):
        self._viewport = v
        self.__update_matrix()

    @property
    def matrix(self):
        return self._matrix

    def fixed_point_zoom(self, f, xy):
        xy  = np.array((*xy, 1))
        inv = np.linalg.inv(self._matrix)
        xy  = (xy @ inv.T)[:2]

        self._center = self._zoom * (f - 1) * xy + self._center
        self._zoom   = self._zoom * f
        self.__update_matrix()

    def __update_matrix(self):
        w, h = self.viewport
        
        vp = np.array(((1,  0, w/2),
                       (0, -1, h/2),
                       (0,  0, 1)), dtype=np.float64)

        x, y = self.center
        z = self.zoom
        
        lookat = np.array(((z, 0, -x),
                           (0, z, -y),
                           (0, 0,  1)))

        self._matrix = vp @ lookat

class Draggable:

    def start(self, xy):
        pass

    def drag(self, xy):
        raise NotImplementedError

    def commit(self):
        pass

class TransformDraggable(Draggable):

    def __init__(self, camera_matrix):
        self.matrix = np.linalg.inv(camera_matrix)

    def transform(self, xy):
        uv = np.hstack((xy, np.ones(1))) @ self.matrix.T
        return uv[:2]

class MovePiece(TransformDraggable):

    def __init__(self, piece, camera_matrix):
        self.piece = piece
        self.init_piece_dxdy = piece.coords.dxdy.copy()
        super().__init__(camera_matrix)

    def start(self, xy):
        self.origin = self.transform(xy)

    def drag(self, xy):
        self.piece.coords.dxdy = self.init_piece_dxdy + (self.transform(xy) - self.origin)

class RotatePiece(TransformDraggable):
    
    def __init__(self, piece, camera_matrix):
        self.piece = piece
        self.init_piece_angle = piece.coords.angle
        super().__init__(camera_matrix)

    def start(self, xy):
        self.origin = self.transform(xy)

    def drag(self, xy):
        self.piece.coords.angle = (self.init_piece_angle
                                   + self.to_angle(self.transform(xy))
                                   - self.to_angle(self.origin))

    def to_angle(self, xy):
        dx, dy = xy - self.piece.coords.dxdy
        return math.atan2(dy, dx) if dx or dy else 0.
            
class MoveCamera(Draggable):

    def __init__(self, camera):
        self.camera = camera
        self.init_camera_center = camera.center.copy()

    def start(self, xy):
        self.origin = xy

    def drag(self, xy):
        delta = (xy - self.origin) * np.array((-1, 1))
        self.camera.center = self.init_camera_center + delta

class AffineTransform:

    def __init__(self, angle=0., xy=(0.,0.)):
        self.angle = angle
        self.dxdy  = np.array(xy, dtype=np.float64)

    def invert_matrix(m):
        angle = math.atan2(m[1,0], m[0,0])
        x, y = m[0,2], m[1,2]
        return AffineTransform(angle, (x,y))

    def get_transform(self):
        return (puzzler.render.Transform()
                .translate(self.dxdy)
                .rotate(self.angle))

    def rot_matrix(self):
        c, s = np.cos(self.angle), np.sin(self.angle)
        return np.array(((c, -s),
                         (s,  c)))

    def copy(self):
        return AffineTransform(self.angle, tuple(self.dxdy))

class Perimeter:

    def __init__(self, points):
        self.points = points
        self.index = dict((tuple(xy), i) for i, xy in enumerate(points))

class ApproxPoly:

    def __init__(self, perimeter, epsilon):
        approx = cv.approxPolyDP(perimeter.points, epsilon, True)
        poly = list(perimeter.index[tuple(xy)] for xy in np.squeeze(approx))

        reset = None
        for i in range(1,len(poly)):
            if poly[i-1] > poly[i]:
                assert reset is None
                reset = i
            else:
                assert poly[i-1] < poly[i]
                
        if reset:
            poly = poly[reset:] + poly[:reset]

        self.perimeter = perimeter
        self.indexes = poly

    def normal_at_index(self, perimeter_index):

        j = bisect.bisect_left(self.indexes, perimeter_index)

        n = len(self.indexes)
        a = self.indexes[j-1]
        b = self.indexes[j%n]

        # print(f"normal_at_index: {perimeter_index=} -> {j=} {n=} {a=} {b=}")
        
        p1 = self.perimeter.points[a]
        p2 = self.perimeter.points[b]
        uv = p2 - p1
        # print(f"{p1=} {p2=} {uv=} {np.linalg.norm(uv)=} {uv / np.linalg.norm(uv)=}")
        uv = uv / np.linalg.norm(uv)
        return uv @ np.array(((0., 1.), (-1., 0.)))

    def distance_to_line_equation(self, perimeter_index):

        j = bisect.bisect_left(self.indexes, i)
        
        x1, y1 = self.perimeter.points[j]
        x2, y2 = self.perimeter.points[(j+1) % len(self.indexes)]

        a = y1 - y2
        b = x2 - x1
        c = x1 * y2 - x2 * y1
        inv_l = 1. / math.hypot(a, b)

        assert a * x1 + b * y1 + c == 0
        assert a * x2 + b * y2 + c == 0

        return (a * inv_l, b * inv_l, c * inv_l)

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
        
def ring_slice(data, a, b):
    return np.concatenate((data[a:], data[:b])) if a >= b else data[a:b]

def ring_range(a, b, n):
    return itertools.chain(range(a, n), range(0, b)) if a >= b else range(a, b)

class Piece:

    def __init__(self, piece):

        self.piece = piece
        self.bbox = (np.min(self.piece.points, axis=0), np.max(self.piece.points, axis=0))
        self.radius = np.max(np.linalg.norm(self.piece.points, axis=1))
        self.perimeter = Perimeter(self.piece.points)
        self.approx = ApproxPoly(self.perimeter, 10)
        self.coords = AffineTransform()
        self.info   = None

    def normal_at_index(self, i):
        uv = self.approx.normal_at_index(i)
        return uv @ self.coords.rot_matrix().T

@dataclass
class BorderInfo:

    pred: tuple
    succ: tuple
    scores: dict[str,float] = field(default_factory=dict)

def make_border_info(piece):

    if piece.label == "I1":
        piece.edges = piece.edges[::-1]
        
    edges = piece.edges

    edge_next = len(edges) - 1
    tab_next = 0
    for i, tab in enumerate(piece.tabs):
        if edges[edge_next].fit_indexes < tab.fit_indexes:
            tab_next = i
            break

    edge_prev = 0
    tab_prev = len(piece.tabs) - 1
    for i, tab in enumerate(piece.tabs):
        if edges[edge_prev].fit_indexes < tab.fit_indexes:
            break
        tab_prev = i

    return BorderInfo((edge_prev, tab_prev), (edge_next, tab_next))

class FrontierComputer:

    def __init__(self, pieces):
        self.pieces = pieces

    def compute_from_border(self, border):

        assert self.is_valid_border(border)

        n = len(border)
        start = [None] * n
        end = [None] * n
        for i in range(n):
            start[i-1], end[i] = self.find_cut(border[i-1], border[i])

        return list(zip(border, start, end))[::-1]

    def is_valid_border(self, border):

        for i in border:
            p = self.pieces.get(i)
            assert p and isinstance(p.info, BorderInfo)

        # should also check that each piece's tab_next connects to the
        # tab_prev on the next piece
        
        return True

    def find_cut(self, prev_label, curr_label):

        prev, curr = self.pieces[prev_label], self.pieces[curr_label]

        di = puzzler.align.DistanceImage(prev.piece)

        t = puzzler.render.Transform()
        t.rotate(-prev.coords.angle).translate(-prev.coords.dxdy)
        t.translate(curr.coords.dxdy).rotate(curr.coords.angle)

        curr_points = t.apply_v2(curr.piece.points)
        # with np.printoptions(precision=1):
        #    print(f"{curr.piece.points}")
        #    print(f"{curr_points=}")
            
        d = di.query(curr_points)

        a, b = curr.info.succ[1], curr.info.pred[1]
        a, b = curr.piece.tabs[a].tangent_indexes[1], curr.piece.tabs[b].tangent_indexes[0]
        n = len(curr_points)

        thresh = 5

        print(f"{prev_label=} {curr_label=}")
        print(f"  {a=} {b=} {n=}")
        curr_start = next(itertools.dropwhile(lambda i: d[i] > thresh, ring_range(a, b, n)), b)
        prev_end = np.argmin(np.linalg.norm(prev.piece.points - curr_points[curr_start], axis=1))
        print(f"  {prev_end=} {curr_start=}")

        return (prev_end, curr_start)

class FrontierExplorer:

    def __init__(self, pieces):
        self.pieces = pieces

    def find_tabs(self, frontier):

        retval = []
        
        for l, a, b in frontier:
            p = self.pieces[l].piece
            rr = RingRange(a, b, len(p.points))
            for i, tab in enumerate(p.tabs):
                if all(j in rr for j in tab.tangent_indexes):
                    retval.append((l, i))

        return retval

    def find_interesting_corners(self, frontier):

        tabs = self.find_tabs(frontier)
        dirs = [self.get_tab_direction(tab) for tab in tabs]
        scores = [np.dot(dirs[i-1], curr_dir) for i, curr_dir in enumerate(dirs)]
        return [(scores[i], tabs[i-1], tabs[i]) for i in range(len(tabs))]

    def get_tab_direction(self, tab, rotate=True):

        p = self.pieces[tab[0]]
        t = p.piece.tabs[tab[1]]
        v = p.piece.points[np.array(t.tangent_indexes)] - t.ellipse.center
        v = v / np.linalg.norm(v, axis=1)
        v = np.sum(v, axis=0)
        v = v / np.linalg.norm(v)
        if not t.indent:
            v = -v
        return puzzler.math.rotate(v, p.coords.angle) if rotate else v

class Autofit:

    def __init__(self, pieces):

        self.pieces = pieces
        self.src_fit_piece = None
        self.src_fit_indexes = None
        self.dst_fit_piece = None
        self.dst_fit_indexes = None

    def align_border(self):

        borders = []
        edges = []
        corners = []
        piece_dict = dict()
        for i in self.pieces:
            if isinstance(i.info, BorderInfo):
                borders.append(i)
                piece_dict[i.piece.label] = i
                if len(i.piece.edges) == 2:
                    corners.append(i)
                elif len(i.piece.edges) == 1:
                    edges.append(i)

        print(f"{len(corners)} corners and {len(edges)} edges")

        for dst in borders:

            edge_aligner = puzzler.align.EdgeAligner(dst.piece)
            
            for src in borders:
                
                # while the fit might be excellent, this would prove
                # topologically difficult
                if dst is src:
                    continue
                
                dst_desc = dst.info.succ
                src_desc = src.info.pred

                # tabs have to be complementary (one indent and one
                # outdent)
                if dst.piece.tabs[dst_desc[1]].indent == src.piece.tabs[src_desc[1]].indent:
                    continue

                dst.info.scores[src.piece.label] = edge_aligner.compute_alignment(
                    dst_desc, src.piece, src_desc)

        if False:
            for dst in borders:
                for src, score in dst.info.scores.items():
                    print(f"{dst.piece.label},{src},{score[0]:.1f}")

        dst = None
        if corners:
            dst = corners[0].piece.label
        elif edges:
            dst = edges[0].piece.label

        if False:
            with np.printoptions(precision=1):
                for p in self.pieces:
                    c = p.coords
                    print(f"align_border: {p.piece.label}: angle={c.angle:.3f} xy={c.dxdy}")

        pairs = []
        mapped = set()
        while dst not in mapped:

            mapped.add(dst)

            best_src = None
            best_score = None
            for k, v in piece_dict[dst].info.scores.items():
                if k in mapped:
                    continue
                if math.isnan(v[0]):
                    continue
                if best_score is None or v[0] < best_score[0]:
                    best_src = k
                    best_score = v

            if best_src is not None:
                print(f"Fit {best_src} to follow {dst}")
                src = piece_dict[best_src]
                dst_m = piece_dict[dst].coords.get_transform().matrix
                src_m = best_score[1].get_transform().matrix
                src.coords = AffineTransform.invert_matrix(dst_m @ src_m)
                pairs.append((best_src, dst))
                dst = best_src
            else:
                print(f"No piece found to follow {dst}!")
                best_src = pairs[0][1]
                print(f"Assume {best_src} follows {dst}")
                pairs.append((best_src, dst))

        if False:
            with np.printoptions(precision=1):
                for p in self.pieces:
                    c = p.coords
                    print(f"align_border: {p.piece.label}: angle={c.angle:.3f} xy={c.dxdy}")

        return pairs

    def global_icp(self, pairs):

        icp = puzzler.icp.IteratedClosestPoint()

        pieces_dict = dict((i.piece.label, i) for i in self.pieces)
            
        def add_correspondence(dst, src, src_coords, src_fit_points):

            src_indexes, dst_indexes = puzzler.align.EdgeAligner(dst.piece).get_correspondence(
                src.piece, src_coords, src_fit_points)

            src_vertex = src.piece.points[src_indexes]
            dst_vertex = dst.piece.points[dst_indexes]
            dst_normal = np.array(list(dst.approx.normal_at_index(i) for i in dst_indexes))

            if bodies.get(src.piece.label) is None:
                bodies[src.piece.label] = icp.make_rigid_body(src.coords.angle)
            src_body = bodies[src.piece.label]

            if src_body.fixed:
                return (None, None)
                
            if bodies.get(dst.piece.label) is None:
                bodies[dst.piece.label] = icp.make_rigid_body(dst.coords.angle)
            dst_body = bodies[dst.piece.label]

            icp.add_body_correspondence(src_body, src_vertex, dst_body, dst_vertex, dst_normal)

            return (src_indexes, dst_indexes)

        bodies = dict()

        dst = pieces_dict[pairs[0][1]]
        bodies[dst.piece.label] = icp.make_rigid_body(
            dst.coords.angle, dst.coords.dxdy, fixed=True)

        for i, j in pairs:

            src = pieces_dict[i]
            dst = pieces_dict[j]

            _, src_coords, src_fit_indexes, dst_fit_indexes = \
                dst.info.scores[src.piece.label]
            
            s, d = add_correspondence(dst, src, src_coords, src_fit_indexes)
            if s is not None and d is not None:
                self.src_fit_piece   = i
                self.src_fit_indexes = s
                self.dst_fit_piece   = j
                self.dst_fit_indexes = d
            else:
                m1 = src_coords.get_transform().matrix
                m2 = np.linalg.inv(m1)
                with np.printoptions(precision=3):
                    print(f"{m1=}")
                    print(f"{m2=}")
                dst_coords = AffineTransform(
                    np.arctan2(m2[1][0], m2[0][0]),
                    np.array((m2[0][2], m2[1][2])))
                with np.printoptions(precision=3):
                    print(f"src_coords: angle={src_coords.angle:.3f} xy={src_coords.dxdy}")
                    print(f"dst_coords: angle={dst_coords.angle:.3f} xy={dst_coords.dxdy}")
                # dst_coords = AffineTransform(-src_coords.angle, -src_coords.dxdy)
                s, d = add_correspondence(src, dst, dst_coords, dst_fit_indexes)
                self.src_fit_piece   = i
                self.src_fit_indexes = d
                self.dst_fit_piece   = j
                self.dst_fit_indexes = s

        for _ in range(2):
            icp.solve()

        with np.printoptions(precision=1):
            for k, v in bodies.items():
                print(f"global_icp:  {k}: angle={v.angle:.3f} xy={v.center}")

        for k, v in bodies.items():
            p = pieces_dict[k]
            p.coords.angle = v.angle
            p.coords.dxdy = v.center

        # return

        axes = [
            icp.make_axis(np.array((0, -1), dtype=np.float), 0., True),
            icp.make_axis(np.array((1, 0), dtype=np.float)),
            icp.make_axis(np.array((0, 1), dtype=np.float)),
            icp.make_axis(np.array((-1, 0), dtype=np.float), 0., True)
        ]

        def add_axis_correspondence(p):
            for edge in p.piece.edges:
                v = edge.line.pts[0] - edge.line.pts[1]
                angle = np.arctan2(v[1], v[0]) - p.coords.angle
                q = int((angle + 2 * math.pi) * 2 / math.pi + .5) % 4
                print(f"{p.piece.label}: {angle=:.3f} {q=}")

                src = bodies[p.piece.label]
                src_vertex = p.piece.points[np.array(edge.fit_indexes)]
                dst = axes[q]

                icp.add_axis_correspondence(src, src_vertex, dst)

        # make the initially fixed rigid body float, so that the fixed
        # axes are the only fixed constraints
        fixed_piece = pairs[0][1]
        bodies[fixed_piece].fixed = False
        bodies[fixed_piece].index = icp.n_cols
        icp.n_cols += 3

        for p in pieces_dict.values():
            add_axis_correspondence(p)

        icp.solve()

        with np.printoptions(precision=1):
            for k, v in bodies.items():
                print(f"global_icp2: {k}: angle={v.angle:.3f} xy={v.center}")
            for i, v in enumerate(axes):
                print(f"global_icp2: axis={i}: value={v.value:.1f} fixed={v.fixed}")

        for k, v in bodies.items():
            p = pieces_dict[k]
            p.coords.angle = v.angle
            p.coords.dxdy = v.center

    def choose_anchor(self):

        corners = self.find_corners()
        if corners:
            return corners[0]

        edges = self.find_edges()
        if edges:
            return edges[0]

        return self.pieces[0]

    def find_corners(self):
        return [p for p in self.pieces if len(p.piece.edges) >= 2]

    def find_edges(self):
        return [p for p in self.pieces if len(p.piece.edges) != 0]

    def find_field(self):
        return [p for p in self.pieces if len(p.piece.edges) == 0]

class PuzzleRenderer:

    def __init__(self, canvas, camera, pieces):
        self.canvas = canvas
        self.camera = camera
        self.pieces = pieces
        self.selection = None
        self.frontier = []
        self.renderer = None
        self.render_fast = None
        self.canvas_w = int(self.canvas.cget('width'))
        self.canvas_h = int(self.canvas.cget('height'))

    def render(self, render_fast):
        
        self.canvas.delete('all')
        self.renderer = puzzler.render.Renderer(self.canvas)
        self.renderer.transform.multiply(self.camera.matrix)

        self.render_fast = render_fast

        colors = ['red', 'green', 'blue']
        for i, piece in enumerate(self.pieces):

            color = colors[i%len(colors)]
            self.draw_piece(piece, color, f"piece_{i}")

        if self.selection is not None:
            self.draw_rotate_handles(self.selection)

        if self.frontier:
            self.draw_frontier(self.frontier)

    def test_bbox(self, bbox):

        ll, ur = bbox
        x0, y0 = ll
        x1, y1 = ur
        points = np.array([(x0,y0), (x1,y0), (x1,y1), (x0,y1)])

        screen = self.renderer.transform.apply_v2(points)
        x = screen[:,0]
        y = screen[:,1]

        if np.max(x) < 0 or np.min(x) > self.canvas_w:
            return False

        if np.max(y) < 0 or np.min(y) > self.canvas_h:
            return False

        return True

    def draw_piece(self, p, color, tag):

        r = self.renderer
            
        with puzzler.render.save_matrix(r.transform):
                
            r.transform.translate(p.coords.dxdy).rotate(p.coords.angle)

            if not self.test_bbox(p.bbox):
                return
            
            if p.piece.edges and False:
                for edge in p.piece.edges:
                    r.draw_lines(edge.line.pts, fill='pink', width=8)
                    r.draw_points(edge.line.pts[0], fill='purple', radius=8)
                    r.draw_points(edge.line.pts[1], fill='green', radius=8)

            if p.piece.tabs and False:
                for tab in p.piece.tabs:
                    pts = puzzler.geometry.get_ellipse_points(tab.ellipse, npts=40)
                    r.draw_polygon(pts, fill='cyan', outline='')

            if self.render_fast:
                points = p.perimeter.points[p.approx.indexes]
            else:
                points = p.piece.points
                
            r.draw_polygon(points, outline=color, fill='', width=2, tag=tag)

            r.draw_text(np.array((0,0)), p.piece.label)

    def draw_rotate_handles(self, piece_id):

        p = self.pieces[piece_id]

        r = self.renderer
        with puzzler.render.save_matrix(r.transform):

            r.transform.translate(p.coords.dxdy).rotate(p.coords.angle)

            r1  = 250
            r2  = 300
            phi = np.linspace(0, math.pi/2, num=20)
            cos = np.cos(phi)
            sin = np.sin(phi)
            x   = np.concatenate((r1 * cos, r2 * np.flip(cos)))
            y   = np.concatenate((r1 * sin, r2 * np.flip(sin)))
            points = np.vstack((x, y)).T
            tags = ('rotate', f'piece_{piece_id}')

            for i in range(4):
                with puzzler.render.save_matrix(r.transform):
                    r.transform.rotate(i * math.pi / 2)
                    r.draw_polygon(points, outline='black', fill='', width=1, tags=tags)

            if p.info:
                tabs = p.piece.tabs
                edges = p.piece.edges
                r.draw_text(tabs[p.info.succ[1]].ellipse.center, "n")
                r.draw_text(tabs[p.info.pred[1]].ellipse.center, "p")
                if p.info.succ[0] != p.info.pred[0]:
                    r.draw_text(np.mean(edges[p.info.succ[0]].line.pts, axis=0), "en")
                    r.draw_text(np.mean(edges[p.info.pred[0]].line.pts, axis=0), "ep")
                else:
                    r.draw_text(np.mean(edges[p.info.succ[0]].line.pts, axis=0), "e")
                    
    def draw_frontier(self, frontier):

        piece_dict = dict((i.piece.label, i) for i in self.pieces)
        
        fe = FrontierExplorer(piece_dict)

        tabs = collections.defaultdict(list)
        for label, tab_no in fe.find_tabs(frontier):
            tabs[label].append(tab_no)

        r = self.renderer
        
        for l, tab_nos in tabs.items():
            p = piece_dict[l]
            with puzzler.render.save_matrix(r.transform):
                r.transform.translate(p.coords.dxdy).rotate(p.coords.angle)
                for tab_no in tab_nos:
                    v = fe.get_tab_direction((l, tab_no), False)
                    p0 = p.piece.tabs[tab_no].ellipse.center
                    p1 = p0 + v * 100
                    r.draw_lines(np.array((p0, p1)), fill='red', width=1, arrow='last')

        for l, a, b in frontier:
            p = piece_dict[l]
            with puzzler.render.save_matrix(r.transform):
                r.transform.translate(p.coords.dxdy).rotate(p.coords.angle)
                r.draw_points(p.piece.points[a], fill='pink', radius=8)
                
        for l, a, b in frontier:
            p = piece_dict[l]
            with puzzler.render.save_matrix(r.transform):
                r.transform.translate(p.coords.dxdy).rotate(p.coords.angle)
                r.draw_points(p.piece.points[b], fill='purple', radius=5)
            


class AlignTk:

    def __init__(self, parent, pieces):
        self.pieces = pieces

        self.draggable = None
        self.selection = None

        self.keep0 = None
        self.keep1 = None

        self.frontier = None

        self._init_ui(parent)

    def canvas_press(self, event):

        piece_no  = None
        drag_type = 'move'
        for tag in self.canvas.gettags(self.canvas.find('withtag', 'current')):
            m = re.fullmatch("piece_(\d+)", tag)
            if m:
                piece_no = int(m[1])
            if tag == 'rotate':
                drag_type = 'turn'

        if piece_no is None:
            self.draggable = MoveCamera(self.camera)
            self.selection = None
        else:
            piece = self.pieces[piece_no]
            if drag_type == 'move':
                self.draggable = MovePiece(piece, self.camera.matrix)
            else:
                self.draggable = RotatePiece(piece, self.camera.matrix)
            self.selection = piece_no

        self.draggable.start(np.array((event.x, event.y)))

        self.render()
        
    def canvas_drag(self, event):

        if self.draggable:
            self.draggable.drag(np.array((event.x, event.y)))
            self.render()

    def canvas_release(self, event):
                
        if self.draggable:
            self.draggable.commit()
            self.draggable = None
            self.render()

    def render(self):
        
        r = PuzzleRenderer(self.canvas, self.camera, self.pieces)
        r.selection = self.selection
        r.frontier = self.frontier
        r.render(True)

        self.render_full += 1
        if 1 == self.render_full:
            self.parent.after_idle(self.full_render)

    def full_render(self):
        
        self.render_full = -1

        r = PuzzleRenderer(self.canvas, self.camera, self.pieces)
        r.selection = self.selection
        r.frontier = self.frontier
        r.render(False)

        self.render_full = 0

    @staticmethod
    def umeyama(P, Q):
        assert P.shape == Q.shape
        n, dim = P.shape

        centeredP = P - P.mean(axis=0)
        centeredQ = Q - Q.mean(axis=0)

        C = np.dot(np.transpose(centeredP), centeredQ) / n

        V, S, W = np.linalg.svd(C)
        d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

        if d:
            S[-1] = -S[-1]
            V[:, -1] = -V[:, -1]

        R = np.dot(V, W)

        varP = np.var(P, axis=0).sum()
        c = 1/varP * np.sum(S) # scale factor

        t = Q.mean(axis=0) - P.mean(axis=0).dot(c*R)

        return c, R, t

    @staticmethod
    def eldridge(P, Q):
        m = P.shape[0]
        assert P.shape == Q.shape == (m, 2)

        # want the optimized rotation and translation to map P -> Q

        Px, Py = np.sum(P, axis=0)
        Qx, Qy = np.sum(Q, axis=0)

        A00 =np.sum(np.square(P))
        A = np.array([[A00, -Py, Px], 
                      [-Py,  m , 0.],
                      [ Px,  0., m ]], dtype=np.float64)

        u0 = np.sum(P[:,0]*Q[:,1]) - np.sum(P[:,1]*Q[:,0])
        u = np.array([u0, Qx-Px, Qy-Py], dtype=np.float64)

        return np.linalg.lstsq(A, u, rcond=None)

    @staticmethod
    def icp(d, n, s):
        m = d.shape[0]
        assert d.shape == n.shape == s.shape == (m, 2)

        a = s[:,0]*n[:,1] - s[:,1]*n[:,0]
        a = a.reshape((m,1))
        A = np.hstack((a, n))

        # n dot (d-s)
        v = np.sum(n * (d-s), axis=1)

        # print(f"icp: {d=} {n=} {s=}")
        # print(f"  {A=}")
        # print(f"  {v=}")

        return np.linalg.lstsq(A, v, rcond=None)

    def do_fit_tabs(self):

        piece_dict = dict((i.piece.label, i) for i in self.pieces)

        dst_piece = piece_dict.get("B2")
        dst_tab_no = 2
        
        src_piece = piece_dict.get("B3")
        src_tab_no = 0

        if not dst_piece or not src_piece:
            return
        
        tab_aligner = puzzler.align.TabAligner(dst_piece.piece)
        
        (mse, src_coords, sfp, dfp) = tab_aligner.compute_alignment(
            dst_tab_no, src_piece.piece, src_tab_no)
        
        t = dst_piece.coords.get_transform()
        m = t.translate(src_coords.dxdy).rotate(src_coords.angle).matrix
        
        src_piece.coords.angle = np.arctan2(m[1][0], m[0][0])
        src_piece.coords.dxdy = np.array([m[0][2], m[1][2]])

        self.render()

        c = puzzler.render.Renderer(self.canvas)
        c.transform.multiply(self.camera.matrix)

        src_points = src_piece.coords.get_transform().apply_v2(
            src_piece.piece.points[list(sfp)])
        c.draw_points(src_points, fill='pink', radius=6)

        print(f"{sfp=} {dfp=}")

        dst_points = dst_piece.coords.get_transform().apply_v2(
            dst_piece.piece.points[list(dfp)])
        c.draw_points(src_points, fill='purple', radius=3)

    def do_fit(self):
        
        print("Fit!")

        piece0 = self.pieces[0]
        piece1 = self.pieces[1]

        points0 = piece0.coords.get_transform().apply_v2(piece0.piece.points)
        kdtree0 = scipy.spatial.KDTree(points0)
        
        points1 = piece1.coords.get_transform().apply_v2(piece1.piece.points)
        kdtree1 = scipy.spatial.KDTree(points1)

        indexes = kdtree0.query_ball_tree(kdtree1, r=15)
        matches = [i for i, v in enumerate(indexes) if len(v)]

        print(f"{len(matches)} points in piece0 have a correspondence with piece1")

        n = len(matches)
        keep = [matches[i] for i in range(n // 8, n, n // 5)]
        print(f"{keep=}")

        data0 = points0[keep]
        d, i = kdtree1.query(data0)
        data1 = points1[i]

        self.keep0 = keep
        self.keep1 = i

        print(f"{data0=}")
        print(f"{data1=}")

        print(f"umeyama: {self.umeyama(data0, data1)}")
        print(f"eldridge: {self.eldridge(data0, data1)}")

        c = puzzler.render.Renderer(self.canvas)
        c.transform.multiply(self.camera.matrix)
        c.draw_circle(data0, 17, fill='purple', outline='')

    def do_refit(self):

        if self.keep0 is None or self.keep1 is None:
            return

        print("Refit!")
        
        piece0 = self.pieces[0]
        piece1 = self.pieces[1]

        points0 = piece0.coords.get_transform().apply_v2(piece0.piece.points)
        points1 = piece1.coords.get_transform().apply_v2(piece1.piece.points)

        # we're solving with an assumption that the rotation is about
        # the center of piece, so make that the origin
        data0 = points0[self.keep0] - piece1.coords.dxdy
        data1 = points1[self.keep1] - piece1.coords.dxdy
        
        normal0 = np.array([piece0.normal_at_index(i) for i in self.keep0])
        normal1 = np.array([piece1.normal_at_index(i) for i in self.keep1])

        theta, tx, ty = self.icp(data0, normal0, data1)[0]
        dxdy = np.array((tx,ty))
        with np.printoptions(precision=3):
            print(f"icp: theta={math.degrees(theta):.3f} degrees, {dxdy=}")

        c = piece1.coords
        print(f"  before: {c.angle=} {c.dxdy=}")

        c.angle += theta
        c.dxdy  += dxdy

        print(f"  after:  {c.angle=} {c.dxdy=}")
        
        self.render()

        c = puzzler.render.Renderer(self.canvas)
        c.transform.multiply(self.camera.matrix)

        for i, xy in enumerate(data0):
            c.draw_lines(np.array((xy, xy+normal0[i]*50)), fill='red', width=1, arrow='last')
        
        for i, xy in enumerate(data1):
            c.draw_lines(np.array((xy, xy+normal1[i]*50)), fill='green', width=1, arrow='last')

    def do_tab_alignment(self):

        print("Tab alignment!")

        num_indents = 0
        num_outdents = 0
        for p in self.pieces:
            for t in p.piece.tabs:
                if t.indent:
                    num_indents += 1
                else:
                    num_outdents += 1
        
        print(f"{len(self.pieces)} pieces: {num_indents} indents, {num_outdents} outdents")

        with open('tab_alignment.csv', 'w', newline='') as f:
            field_names = 'dst_label dst_tab_no src_label src_tab_no mse'.split()
            writer = csv.DictWriter(f, field_names)
            writer.writeheader()

            for dst in self.pieces:
                rows = []
                tab_aligner = puzzler.align.TabAligner(dst.piece)
                for src in self.pieces:
                    if src is dst:
                        continue
                    for dst_tab_no, dst_tab in enumerate(dst.piece.tabs):
                        for src_tab_no, src_tab in enumerate(src.piece.tabs):
                            if dst_tab.indent == src_tab.indent:
                                continue
                            mse = tab_aligner.compute_alignment(dst_tab_no, src.piece, src_tab_no)[0]
                            rows.append({'dst_label': dst.piece.label,
                                         'dst_tab_no': dst_tab_no,
                                         'src_label': src.piece.label,
                                         'src_tab_no': src_tab_no,
                                         'mse': mse})
                print(f"{dst.piece.label}: {len(rows)} rows")
                writer.writerows(rows)

    def do_tab_alignment_B2(self):

        dsts = [("A2", 2), ("B1", 1)]
        scores = []

        piece_dict = dict((i.piece.label, i) for i in self.pieces)

        for dst_label, dst_tab_no in dsts:

            dst_piece = piece_dict[dst_label]
            dst_tab   = dst_piece.piece.tabs[dst_tab_no]

            the_scores = dict()
            tab_aligner = puzzler.align.TabAligner(dst_piece.piece)
            
            for src_piece in self.pieces:
                if src_piece is dst_piece:
                    continue
                src_label = src_piece.piece.label
                for src_tab_no, src_tab in enumerate(src_piece.piece.tabs):
                    if dst_tab.indent == src_tab.indent:
                        continue
                    mse, _, sfp, _ = tab_aligner.compute_alignment(dst_tab_no, src_piece.piece, src_tab_no)
                    a, b = sfp
                    n = b-a if b > a else len(src_piece.piece.points)-a+b
                    the_scores[(src_label,src_tab_no)] = (mse, n)

            scores.append(the_scores)

        allways = []
        for k0, v0 in scores[0].items():
            for k1, v1 in scores[1].items():
                if k1[0] == k0[0]:
                    mse = (v0[0] * v0[1] + v1[0] * v1[1]) / (v0[1] + v1[1])
                    allways.append((mse, (*k0, *v0), (*k1, *v1)))

        allways.sort()

        aligner = puzzler.align.MultiAligner([(piece_dict[i].piece, j, piece_dict[i].coords) for i, j in dsts])

        for i, j in enumerate(allways):
            
            print(i, j)
            if i == 20:
                break

            _, ii, jj = j

            src_piece = piece_dict[ii[0]]
            src_tabs  = [ii[1], jj[1]]
            
            src_coords = aligner.compute_alignment(src_piece.piece, src_tabs)

            mse = aligner.measure_fit(src_piece.piece, src_tabs, src_coords)

            print(f"  fit: angle={src_coords.angle:3f} xy={src_coords.dxdy} {mse=:.1f}")
            
        print(f"{len(scores[0])=} {len(scores[1])=} {len(allways)=}")
                
    def do_autofit(self):

        pieces_dict = dict((i.piece.label, i) for i in self.pieces)
        
        bs = puzzler.solver.BorderSolver({i.piece.label: i.piece for i in self.pieces})

        scores = bs.score_matches()
        border = bs.link_pieces(scores)
        print(f"{border=}")
        bs.estimate_puzzle_size(border)

        print("Autofit!")

        af = Autofit(self.pieces)
        pairs = af.align_border()
        af.global_icp(pairs)

        pieces_dict = dict((i.piece.label, i) for i in self.pieces)

        border = [i for i, _ in pairs]
        
        fc = FrontierComputer(pieces_dict)

        self.frontier = fc.compute_from_border(border)

        fe = FrontierExplorer(pieces_dict)
        corners = fe.find_interesting_corners(self.frontier)
        for s, t0, t1 in sorted(corners, key=lambda x: abs(x[0])):
            print(f"{t0}, {t1}: {s:.3f}")

        self.render()
        
    def mouse_wheel(self, event):
        f = pow(1.2, 1 if event.delta > 0 else -1)
        xy = (event.x, event.y)
        self.camera.fixed_point_zoom(f, xy)
        self.motion(event)
        self.render()

    def motion(self, event):
        xy0 = np.array((event.x, event.y, 1))
        xy1 = xy0 @ np.linalg.inv(self.camera.matrix).T
        tags = self.canvas.find('overlapping', event.x-1, event.y-1, event.x+1, event.y+1)
        tags = [self.canvas.gettags(i) for i in tags]
        self.var_label.set(f"{xy1[0]:.0f},{xy1[1]:.0f} " + ','.join(str(i) for i in tags))

    def keypress(self, event):
        print(event)

    def _init_ui(self, parent):

        w, h = parent.winfo_screenwidth(), parent.winfo_screenheight()
        viewport = (min(w-32,1024), min(h-128,1024))

        self.parent = parent
        self.camera = Camera(np.array((0,0), dtype=np.float64), 1/3, viewport)
        
        self.frame = ttk.Frame(parent, padding=5)
        self.frame.grid(sticky=(N, W, E, S))

        self.canvas = Canvas(self.frame, width=viewport[0], height=viewport[1],
                             background='white', highlightthickness=0)
        self.canvas.grid(column=0, row=0, sticky=(N, W, E, S))
        self.canvas.bind("<Button-1>", self.canvas_press)
        self.canvas.bind("<B1-Motion>", self.canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.canvas_release)
        self.canvas.bind("<MouseWheel>", self.mouse_wheel)
        self.canvas.bind("<Motion>", self.motion)

        parent.bind("<Key>", self.keypress)

        self.controls = ttk.Frame(self.frame)
        self.controls.grid(row=1, sticky=(W,E))

        b1 = ttk.Button(self.controls, text='Fit', command=self.do_fit)
        b1.grid(column=0, row=0, sticky=W)

        b2 = ttk.Button(self.controls, text='Refit', command=self.do_refit)
        b2.grid(column=1, row=0, sticky=W)

        b3 = ttk.Button(self.controls, text='Autofit', command=self.do_autofit)
        b3.grid(column=3, row=0, sticky=W)

        b4 = ttk.Button(self.controls, text='Fit Tabs', command=self.do_fit_tabs)
        b4.grid(column=4, row=0, sticky=W)

        b5 = ttk.Button(self.controls, text='Tab Alignment', command=self.do_tab_alignment)
        b5.grid(column=5, row=0, sticky=W)

        self.var_label = StringVar(value="x,y")
        l1 = ttk.Label(self.controls, textvariable=self.var_label, width=40)
        l1.grid(column=6, row=0, sticky=(E))

        self.render_full = False
        self.render()
                
def align_ui(args):

    puzzle = puzzler.file.load(args.puzzle)

    by_label = dict()
    for p in puzzle.pieces:
        by_label[p.label] = p

    labels = set(args.labels)

    if args.edges:
        labels |= set('A1 A2 A3 A4 A5 A6 A7 A8 A9 A10 A11 B1 B11 C1 C11 D1 D11 E1 E12 F1 F11 G1 G11 H1 H11 I1 I2 I3 I4 I5 I6 I7 I8 I9 I10 I11'.split())

    if not labels:
        labels |= set(by_label.keys())

    pieces = [Piece(by_label[l]) for l in sorted(labels)]
    for piece in pieces:
        if piece.piece.edges:
            piece.info = make_border_info(piece.piece)

    rows = set()
    cols = set()
    for piece in pieces:
        m = re.fullmatch("([a-zA-Z]+)(\d+)", piece.piece.label)
        if m:
            rows.add(m[1])
            cols.add(int(m[2]))

    rows = dict((r, i) for i, r in enumerate(sorted(rows)))
    cols = dict((c, i) for i, c in enumerate(sorted(cols)))

    for piece in pieces:
        m = re.fullmatch("([a-zA-Z]+)(\d+)", piece.piece.label)
        if m:
            row, col = m[1], int(m[2])
            x = cols[col] * 1000.
            y = rows[row] * -1000.
            piece.coords.dxdy = np.array((x, y))

    root = Tk()
    ui = AlignTk(root, pieces)
    root.bind('<Key-Escape>', lambda e: root.destroy())
    root.title("Puzzler: align")
    root.wm_resizable(0, 0)
    root.mainloop()

def add_parser(commands):

    parser_align = commands.add_parser("align", help="UI to experiment with aligning pieces")
    parser_align.add_argument("labels", nargs='*')
    parser_align.add_argument("-e", "--edges", help="add all edges", action='store_true')
    parser_align.set_defaults(func=align_ui)
