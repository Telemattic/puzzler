import bisect
import cv2 as cv
import math
import numpy as np
import scipy
import puzzler

from tkinter import *
from tkinter import font
from tkinter import ttk

class AffineTransform:

    def __init__(self, angle=0., xy=(0.,0.)):
        self.angle = angle
        self.dxdy  = np.array(xy, dtype=np.float64)

    def to_global_xy(self, points):
        return points @ self.rot_matrix() + self.dxdy

    def to_local_xy(self, points):
        return (points - self.dxdy) @ self.rot_matrix().T

    def rot_matrix(self):
        c, s = np.cos(self.angle), np.sin(self.angle)
        return np.array(((c, s),
                         (-s, c)))

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
        
class Piece:

    def __init__(self, piece):

        self.piece  = piece
        self.center = np.array(np.mean(self.piece.points, axis=0), dtype=np.int32)
        self.perimeter = Perimeter(self.piece.points)
        self.approx = ApproxPoly(self.perimeter, 10)
        self.coords = AffineTransform()
        self.kdtree = scipy.spatial.KDTree(self.piece.points)

    def get_transform(self):
        return (puzzler.render.Transform()
                .translate(*self.coords.dxdy)
                .rotate(self.coords.angle)
                .translate(*-self.center))        

    def dist(self, xy):
        xy = self.coords.to_local_xy(xy) + self.center
        return self.kdtree.query(xy, distance_upper_bound=20)

    def normal_at_index(self, i):
        uv = self.approx.normal_at_index(i)
        return uv @ self.coords.rot_matrix()

class Autofit:

    def __init__(self, pieces):

        self.pieces = pieces

        anchor = self.choose_anchor()

        self.align_edges(self.pieces[0], self.pieces[1])
        
    def align_edges(self, dst, src):

        dst_edge = self.get_edge(dst)

        dst_edge_vec = dst_edge.line.pts[1] - dst_edge.line.pts[0]
        dst_edge_angle = dst.coords.angle + np.arctan2(dst_edge_vec[1], dst_edge_vec[0])

        src_edge = self.get_edge(src)

        src_edge_vec = src_edge.line.pts[1] - src_edge.line.pts[0]
        src_edge_angle = src.coords.angle + np.arctan2(src_edge_vec[1], src_edge_vec[0])

        src.coords.angle += dst_edge_angle - src_edge_angle

        dst_line = dst.get_transform().apply_v2(dst_edge.line.pts)
        src_point = src.get_transform().apply_v2(src_edge.line.pts[0])
        
        src.coords.dxdy = src.coords.dxdy + puzzler.math.vector_to_line(src_point, dst_line)

    def get_edge(self, piece):

        return piece.piece.edges[0]
        
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

class DragHandle:

    def __init__(self, coord):
        self.coord = coord
        self.color = 'black'
        self.drag_radius = 50
        self.knob_radius = 20

        self.drag_kind = None
        self.drag_start = None

    def hit_test(self, xy):
        
        dragxy = self.coord.dxdy
        knobxy = np.float64((0.,self.drag_radius)) @ self.coord.rot_matrix() + dragxy

        if np.linalg.norm(knobxy - xy) <= self.knob_radius:
            return 'turn'

        if np.linalg.norm(dragxy - xy) <= self.drag_radius:
            return 'move'

        return None

    def start_drag(self, xy, kind):

        self.drag_kind = kind
        self.drag_start = xy
        self.drag_coord = self.coord.copy()

    def drag(self, xy):

        if self.drag_kind == 'move':
            dx, dy = xy[0] - self.drag_start[0], xy[1] - self.drag_start[1]
            self.coord.dxdy = self.drag_coord.dxdy + (dx, dy)
            
            #newloc = self.drag_coord.dxdy + (dx, dy)
            #print(f"move: xy={tuple(newloc)}")

        elif self.drag_kind == 'turn':
            dx, dy = xy[0] - self.drag_coord.dxdy[0], xy[1] - self.drag_coord.dxdy[1]
            if dx or dy:
                angle = math.atan2(dy, dx)
                angle -= math.pi / 2
                self.coord.angle = angle
                # print(f"turn: angle={angle:.3f} ({angle*180./math.pi:.3f} deg)")

    def end_drag(self):

        self.drag_kind = None
        self.drag_start = None
        self.drag_coord = None

    def render(self, ctx):

        dragxy = np.float64((0,0))
        ctx.draw_circle(dragxy, self.drag_radius,
                        outline=self.color, fill='', width=3)

        knobxy = np.float64((0.,self.drag_radius))
        ctx.draw_circle(knobxy, self.knob_radius,
                        fill=self.color, outline='white', width=2)

        # graph.draw_line(tuple(dragxy), tuple(knobxy), color=self.color, width=2)

class AlignTk:

    def __init__(self, parent, pieces):
        self.pieces = pieces

        self.drag_handle = None
        self.drag_id     = None

        self.keep0 = None
        self.keep1 = None

        self._init_ui(parent)

    def find_nearest_piece(self, xy):
        return min((p.dist(xy)[0], i) for i, p in enumerate(self.pieces))

    def canvas_press(self, event):

        cm = self.camera_matrix
        inv = np.linalg.inv(cm)
        wxy = np.array((event.x, event.y, 1), dtype=np.float64) @ inv.T

        xy = wxy[:2]

        dist, id = self.find_nearest_piece(xy)
        if dist < 20:
            self.drag_id = id
            self.drag_handle = DragHandle(self.pieces[id].coords)
            self.drag_handle.start_drag(xy, 'move')
        else:
            for i, p in enumerate(self.pieces):
                h = DragHandle(p.coords)
                t = h.hit_test(xy)
                if t:
                    self.drag_id = i
                    self.drag_handle = h
                    h.start_drag(xy, t)
                    break

        self.render()
        
    def canvas_drag(self, event):

        cm = self.camera_matrix
        inv = np.linalg.inv(cm)
        wxy = np.array((event.x, event.y, 1), dtype=np.float64) @ inv.T

        xy = wxy[:2]

        if self.drag_handle:
            self.drag_handle.drag(xy)
            self.render()

    def canvas_release(self, event):
                
        if self.drag_handle:
            self.drag_handle.end_drag()
                
        self.drag_handle = None
        self.drag_id     = None
            
        self.render()

    def render(self):

        canvas = self.canvas
        canvas.delete('all')

        colors = ['red', 'green', 'blue']
        for i, p in enumerate(self.pieces):

            color = colors[i%len(colors)]

            c = puzzler.render.Renderer(canvas)
            c.transform.multiply(self.camera_matrix)
            
            c.transform.translate(*p.coords.dxdy)
            c.transform.rotate(p.coords.angle)

            with puzzler.render.save_matrix(c.transform):
                
                c.transform.translate(*-p.center)
            
                if p.piece.edges and False:
                    for edge in p.piece.edges:
                        c.draw_lines(edge.line.pts, fill='pink', width=8)
                        c.draw_points(edge.line.pts[0], fill='purple', radius=8)
                        c.draw_points(edge.line.pts[1], fill='green', radius=8)

                c.draw_polygon(p.piece.points, outline=color, fill='', width=2)

            h = DragHandle(p.coords)
            h.render(c)

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

    def do_fit(self):
        
        print("Fit!")

        piece0 = self.pieces[0]
        piece1 = self.pieces[1]

        points0 = piece0.coords.to_global_xy(piece0.piece.points) - piece0.center
        kdtree0 = scipy.spatial.KDTree(points0)
        
        points1 = piece1.coords.to_global_xy(piece1.piece.points) - piece1.center
        kdtree1 = scipy.spatial.KDTree(points1)

        indexes = kdtree0.query_ball_tree(kdtree1, r=15)
        matches = [i for i, v in enumerate(indexes) if len(v)]

        print(f"{len(matches)} points in piece0 have a correspondence with piece1")

        keep = [matches[i] for i in range(0, len(matches), (len(matches)-4) // 4)]
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
        c.transform.multiply(self.camera_matrix)
        c.draw_circle(data0, 17, fill='purple', outline='')

    def do_refit(self):

        if self.keep0 is None or self.keep1 is None:
            return
        
        piece0 = self.pieces[0]
        piece1 = self.pieces[1]

        points0 = piece0.coords.to_global_xy(piece0.piece.points) - piece0.center
        points1 = piece1.coords.to_global_xy(piece1.piece.points) - piece1.center

        data0 = points0[self.keep0]
        data1 = points1[self.keep1]
        
        normal0 = np.array([piece0.normal_at_index(i) for i in self.keep0])
        normal1 = np.array([piece1.normal_at_index(i) for i in self.keep1])

        # data1 = data1 - piece1.coords.dxdy

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
        c.transform.multiply(self.camera_matrix)

        with puzzler.render.save_matrix(c.transform):
            p = piece0
            c.transform.translate(*p.coords.dxdy)
            c.transform.rotate(p.coords.angle)
            c.transform.translate(*-p.center)

            for i, k in enumerate(self.keep0):
                xy = p.piece.points[k]
                c.draw_lines(np.array((xy, xy+normal0[i]*50)), fill='red', width=1, arrow='last')
        
        with puzzler.render.save_matrix(c.transform):
            p = piece1
            c.transform.translate(*p.coords.dxdy)
            c.transform.rotate(p.coords.angle)
            c.transform.translate(*-p.center)

            for i, k in enumerate(self.keep1):
                xy = p.piece.points[k]
                c.draw_lines(np.array((xy, xy+normal1[i]*50)), fill='green', width=1, arrow='last')
        

    def do_autofit(self):
        print("Autofit!")
        af = Autofit(self.pieces)
        self.render()

    def mouse_wheel(self, event):
        self.camera_scale *= pow(1.05, 1 if event.delta > 0 else -1)
        self.init_camera_matrix()
        self.render()
        # print(event)

    def motion(self, event):
        self.var_label.set(f"{event.x},{event.y}")
        print(self.canvas.find('overlapping', event.x-1, event.y-1, event.x+1, event.y+1))

    def init_camera_matrix(self):
        
        w, h, s = self.canvas_width, self.canvas_height, self.camera_scale

        self.camera_matrix = np.array(
            ((1/s,    0,   0),
             (  0, -1/s, h-1),
             (  0,    0,   1)), dtype=np.float64)
        
    def _init_ui(self, parent):

        self.canvas_width = 1024
        self.canvas_height = 1024
        self.camera_scale = 3.
        self.init_camera_matrix()
        
        self.frame = ttk.Frame(parent, padding=5)
        self.frame.grid(sticky=(N, W, E, S))

        self.canvas = Canvas(self.frame, width=self.canvas_width, height=self.canvas_height,
                             background='white', highlightthickness=0)
        self.canvas.grid(column=0, row=0, sticky=(N, W, E, S))
        self.canvas.bind("<Button-1>", self.canvas_press)
        self.canvas.bind("<B1-Motion>", self.canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.canvas_release)
        self.canvas.bind("<MouseWheel>", self.mouse_wheel)
        self.canvas.bind("<Motion>", self.motion)

        self.controls = ttk.Frame(self.frame)
        self.controls.grid(row=1, sticky=(W,E))

        b1 = ttk.Button(self.controls, text='Fit', command=self.do_fit)
        b1.grid(column=0, row=0, sticky=W)

        b2 = ttk.Button(self.controls, text='Refit', command=self.do_refit)
        b2.grid(column=1, row=0, sticky=W)

        b3 = ttk.Button(self.controls, text='Autofit', command=self.do_autofit)
        b3.grid(column=3, row=0, sticky=W)

        self.var_label = StringVar(value="x,y")
        l1 = ttk.Label(self.controls, textvariable=self.var_label, width=20)
        l1.grid(column=4, row=0, sticky=(E))

        #self.controls.columnconfigure(4, weight=1)

        self.render()
                
def align_ui(args):

    puzzle = puzzler.file.load(args.puzzle)

    by_label = dict()
    for p in puzzle.pieces:
        by_label[p.label] = p

    pieces = [Piece(by_label[l]) for l in args.labels]

    root = Tk()
    ui = AlignTk(root, pieces)
    root.bind('<Key-Escape>', lambda e: root.destroy())
    root.title("Puzzler: align")
    root.wm_resizable(0, 0)
    root.mainloop()

def add_parser(commands):

    parser_align = commands.add_parser("align", help="UI to experiment with aligning pieces")
    parser_align.add_argument("labels", nargs='+')
    parser_align.set_defaults(func=align_ui)
