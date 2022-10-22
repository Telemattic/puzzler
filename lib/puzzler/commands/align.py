import bisect
import cv2 as cv
import math
import numpy as np
import re
import scipy
import puzzler.feature
import puzzler

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

def ring_slice(a, begin, end):
    if begin < end:
        return a[begin:end]
    return np.concatenate((a[begin:], a[:end]))
        
class Piece:

    def __init__(self, piece):

        self.piece  = piece
        self.perimeter = Perimeter(self.piece.points)
        self.approx = ApproxPoly(self.perimeter, 10)
        self.coords = AffineTransform()
        self.info   = None

    def normal_at_index(self, i):
        uv = self.approx.normal_at_index(i)
        return uv @ self.coords.rot_matrix().T

@dataclass
class BorderInfo:

    edge: puzzler.feature.Edge
    tab_prev: puzzler.feature.Tab
    tab_next: puzzler.feature.Tab
    scores: dict[str,float] = field(default_factory=dict)

def make_border_info(piece):

    if piece.label == "I1":
        piece.edges = piece.edges[::-1]
        
    edges = piece.edges

    tab_next = piece.tabs[0]
    for tab in piece.tabs:
        if edges[-1].fit_indexes < tab.fit_indexes:
            tab_next = tab
            break

    tab_prev = piece.tabs[-1]
    for tab in piece.tabs:
        if edges[0].fit_indexes < tab.fit_indexes:
            break
        tab_prev = tab

    return BorderInfo(edges, tab_prev, tab_next)

def compute_rigid_transform(P, Q):
        
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

    return np.linalg.lstsq(A, u, rcond=None)[0]

class TabAligner:

    def __init__(self, dst):
        self.dst = dst
        self.kdtree = scipy.spatial.KDTree(dst.piece.points)

    def compute_alignment(self, dst_tab_no, src, src_tab_no):

        src_points = self.get_tab_points(src.piece, src_tab_no)
        dst_points = self.get_tab_points(self.dst.piece, dst_tab_no)

        r, x, y = compute_rigid_transform(src_points, dst_points)

        src_coords = AffineTransform(r, (x,y))

        mse, sfp, dfp = self.measure_fit(src, src_tab_no, src_coords)

        return (mse, src_coords, sfp, dfp)

    def measure_fit(self, src, src_tab_no, src_coords):
        
        d, i = self.kdtree.query(src_coords.get_transform().apply_v2(src.piece.points))

        l, r = src.piece.tabs[src_tab_no].tangent_indexes

        n = len(src.piece.points)

        thresh = 5
        
        for j in range(l+n//2, l+n, 1):
            if d[j%n] < thresh:
                break
        l = j % n

        for j in range(r+n//2, r, -1):
            if d[j%n] < thresh:
                break
        r = j % n

        d = ring_slice(d, l, r)

        mse = np.sum(d ** 2) / len(d)

        src_fit_points = (l, r)
        dst_fit_points = (i[r], i[l])

        return (mse, src_fit_points, dst_fit_points)

    @staticmethod
    def get_tab_points(piece, tab_no):

        tab = piece.tabs[tab_no]
        ti = list(tab.tangent_indexes)
        l, r = piece.points[ti if tab.indent else ti[::-1]]
        c = tab.ellipse.center
        return np.array([l, c, r])

class EdgeAligner:

    def __init__(self, dst):
        assert isinstance(dst.info, BorderInfo)
        self.dst = dst
        self.kdtree = scipy.spatial.KDTree(dst.piece.points)

    def compute_alignment(self, src):
        
        # print(f"align_edge_src_to_dst: dst={dst.piece.label} src={src.piece.label}")

        assert isinstance(src.info, BorderInfo)

        dst_edge = self.dst.piece.edges[-1]

        dst_edge_vec = dst_edge.line.pts[1] - dst_edge.line.pts[0]
        dst_edge_angle = np.arctan2(dst_edge_vec[1], dst_edge_vec[0])

        src_edge = src.piece.edges[0]

        src_edge_vec = src_edge.line.pts[1] - src_edge.line.pts[0]
        src_edge_angle = np.arctan2(src_edge_vec[1], src_edge_vec[0])

        src_coords = AffineTransform()
        src_coords.angle = dst_edge_angle - src_edge_angle

        dst_line = dst_edge.line.pts
        src_point = src_coords.get_transform().apply_v2(src_edge.line.pts[0])
        
        src_coords.dxdy = puzzler.math.vector_to_line(src_point, dst_line)

        dst_center = self.dst.info.tab_next.ellipse.center
        src_center = src_coords.get_transform().apply_v2(
            src.info.tab_prev.ellipse.center)

        dst_edge_vec = puzzler.math.unit_vector(dst_edge_vec)
        d = np.dot(dst_edge_vec, (dst_center - src_center))
        src_coords.dxdy = src_coords.dxdy + dst_edge_vec * d

        src_fit_pts = (src.info.tab_prev.tangent_indexes[0],
                       src.info.edge[0].fit_indexes[0])

        dst_fit_pts = (self.dst.info.edge[-1].fit_indexes[1],
                       self.dst.info.tab_next.tangent_indexes[1])

        # with np.printoptions(precision=3):
        #     print(f"src_coords: angle={src_coords.angle:.3f} xy={src_coords.dxdy}")
        #     print(f"  matrix={src_coords.get_transform().matrix}")

        points = ring_slice(src.piece.points, *src_fit_pts)

        d, _ = self.kdtree.query(src_coords.get_transform().apply_v2(points))

        mse = np.sum(d ** 2) / len(d)

        # print(f"  MSE={mse:.1f}")

        return (mse, src_coords, src_fit_pts, dst_fit_pts)
        
    def get_correspondence(self, src, src_coords, src_fit_indexes):

        a, b = src_fit_indexes

        n = len(src.piece.points)
        if a < b:
            src_indexes = list(range(a,b))
        else:
            src_indexes = list(range(a,n)) + list(range(0,b))

        n = len(src_indexes)
        src_indexes = [src_indexes[i] for i in range(n // 10, n, n // 5)]

        src_points = src.piece.points[src_indexes]

        dst_dist, dst_indexes = self.kdtree.query(src_coords.get_transform().apply_v2(src_points))

        return (src_indexes, dst_indexes)

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

            edge_aligner = EdgeAligner(dst)
            
            for src in borders:
                
                # while the fit might be excellent, this would prove
                # topologically difficult
                if dst is src:
                    continue
                
                # tabs have to be complementary (one indent and one
                # outdent)
                if dst.info.tab_next.indent == src.info.tab_prev.indent:
                    continue

                dst.info.scores[src.piece.label] = edge_aligner.compute_alignment(src)

        for dst in borders:
            for src, score in dst.info.scores.items():
                print(f"{dst.piece.label},{src},{score[0]:.1f}")

        dst = None
        if corners:
            dst = corners[0].piece.label
        elif edges:
            dst = edges[0].piece.label

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
        
        with np.printoptions(precision=1):
            for p in self.pieces:
                c = p.coords
                print(f"align_border: {p.piece.label}: angle={c.angle:.3f} xy={c.dxdy}")

        return pairs

    def global_icp(self, pairs):

        icp = puzzler.icp.IteratedClosestPoint()

        pieces_dict = dict((i.piece.label, i) for i in self.pieces)
            
        def add_correspondence(dst, src, src_coords, src_fit_points):

            src_indexes, dst_indexes = EdgeAligner(dst).get_correspondence(
                src, src_coords, src_fit_points)

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

            icp.add_correspondence(src_body, src_vertex, dst_body, dst_vertex, dst_normal)

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

class AlignTk:

    def __init__(self, parent, pieces):
        self.pieces = pieces

        self.draggable = None
        self.selection = None

        self.keep0 = None
        self.keep1 = None

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

        canvas = self.canvas
        canvas.delete('all')

        colors = ['red', 'green', 'blue']
        for i, piece in enumerate(self.pieces):

            color = colors[i%len(colors)]

            r = puzzler.render.Renderer(canvas)
            r.transform.multiply(self.camera.matrix)

            self.draw_piece(r, piece, color, f"piece_{i}")

        if self.selection is not None:
            self.draw_rotate_handles(self.selection)
            
    def draw_piece(self, r, p, color, tag):
            
        with puzzler.render.save_matrix(r.transform):
                
            r.transform.translate(p.coords.dxdy)
            r.transform.rotate(p.coords.angle)
            
            if p.piece.edges and False:
                for edge in p.piece.edges:
                    r.draw_lines(edge.line.pts, fill='pink', width=8)
                    r.draw_points(edge.line.pts[0], fill='purple', radius=8)
                    r.draw_points(edge.line.pts[1], fill='green', radius=8)

            if p.piece.tabs and False:
                for tab in p.piece.tabs:
                    pts = puzzler.geometry.get_ellipse_points(tab.ellipse, npts=40)
                    r.draw_polygon(pts, fill='cyan', outline='')

            r.draw_polygon(p.piece.points, outline=color, fill='', width=2, tag=tag)

            r.draw_text(np.array((0,0)), p.piece.label)

    def draw_rotate_handles(self, piece_id):

        p = self.pieces[piece_id]

        c = puzzler.render.Renderer(self.canvas)
        c.transform.multiply(self.camera.matrix)
            
        c.transform.translate(p.coords.dxdy)
        c.transform.rotate(p.coords.angle)

        r1  = 250
        r2  = 300
        phi = np.linspace(0, math.pi/2, num=20)
        cos = np.cos(phi)
        sin = np.sin(phi)
        x   = np.concatenate((r1 * cos, r2 * np.flip(cos)))
        y   = np.concatenate((r1 * sin, r2 * np.flip(sin)))
        points = np.vstack((x, y)).T
        # print(f"{points=}")
        tags = ('rotate', f'piece_{piece_id}')

        for i in range(4):
            with puzzler.render.save_matrix(c.transform):
                c.transform.rotate(i * math.pi / 2)
                c.draw_polygon(points, outline='black', fill='', width=1, tags=tags)

        if p.info:
            c.draw_text(p.info.tab_next.ellipse.center, "n")
            c.draw_text(p.info.tab_prev.ellipse.center, "p")
            for i, e in enumerate(p.info.edge):
                c.draw_text(np.mean(e.line.pts, axis=0), f"e{i}")

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
        
        tab_aligner = TabAligner(dst_piece)
        
        (mse, src_coords, sfp, dfp) = tab_aligner.compute_alignment(
            dst_tab_no, src_piece, src_tab_no)
        
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

    def do_autofit(self):
        print("Autofit!")
        af = Autofit(self.pieces)
        pairs = af.align_border()
        af.global_icp(pairs)
        self.render()

        pieces_dict = dict((i.piece.label, i) for i in self.pieces)

        if af.src_fit_indexes is not None and af.dst_fit_indexes is not None:
            c = puzzler.render.Renderer(self.canvas)
            c.transform.multiply(self.camera.matrix)

            src = pieces_dict[af.src_fit_piece]
            src_points = src.piece.points[af.src_fit_indexes]
            c.draw_points(src.coords.get_transform().apply_v2(src_points), fill='pink', radius=6)

            dst = pieces_dict[af.dst_fit_piece]
            dst_points = dst.piece.points[af.dst_fit_indexes]
            c.draw_points(dst.coords.get_transform().apply_v2(dst_points), fill='purple', radius=3)

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

        dx = dy = 0
        if event.keycode == 38:
            dy = 1
        elif event.keycode == 40:
            dy = -1
        elif event.keycode == 37:
            dx = -1
        elif event.keycode == 39:
            dx = 1

        if dx or dy:
            dx *= 100 * self.camera.zoom
            dy *= 100 * self.camera.zoom

            self.camera.center = self.camera.center + np.array((dx, dy))
            self.render()

    def _init_ui(self, parent):

        w, h = parent.winfo_screenwidth(), parent.winfo_screenheight()
        viewport = (min(w-32,1024), min(h-128,1024))
        
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

        self.var_label = StringVar(value="x,y")
        l1 = ttk.Label(self.controls, textvariable=self.var_label, width=40)
        l1.grid(column=5, row=0, sticky=(E))

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
