import bisect
import cv2 as cv
import math
import numpy as np
import PySimpleGUI as sg
import scipy
import puzzler

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

    def __init__(self, data):

        points = data.points
        self.points = points - np.array(np.mean(points, axis=0), dtype=np.int32)
        self.perimeter = Perimeter(self.points)
        self.approx = ApproxPoly(self.perimeter, 10)
        self.coords = AffineTransform()
        self.kdtree = scipy.spatial.KDTree(self.points)

    def dist(self, xy):
        xy = self.coords.to_local_xy(xy)
        return self.kdtree.query(xy, distance_upper_bound=20)

    def normal_at_index(self, i):
        uv = self.approx.normal_at_index(i)
        return uv @ self.coords.rot_matrix()

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

    def render(self, graph):

        dragxy = self.coord.dxdy
        graph.draw_circle(tuple(dragxy), self.drag_radius, line_color=self.color, line_width=3)

        knobxy = np.float64((0.,self.drag_radius)) @ self.coord.rot_matrix() + dragxy
        graph.draw_circle(tuple(knobxy), self.knob_radius,
                          fill_color=self.color, line_color='white', line_width=2)

        # graph.draw_line(tuple(dragxy), tuple(knobxy), color=self.color, width=2)

class AlignUI:

    def __init__(self, pieces):
        self.pieces = pieces

        self.drag_handle = None
        self.drag_id     = None

        self.keep0 = None
        self.keep1 = None

    def find_nearest_piece(self, xy):
        return min((p.dist(xy)[0], i) for i, p in enumerate(self.pieces))

    def graph_drag(self, xy, end):

        if end:

            if self.drag_handle:
                self.drag_handle.end_drag()
                
            self.drag_handle = None
            self.drag_id     = None
            
        elif self.drag_handle:

            self.drag_handle.drag(xy)
                
        else:

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

    def render(self):

        graph = self.window['graph']
        graph.erase()

        colors = ['red', 'green', 'blue']
        for i, p in enumerate(self.pieces):

            color = colors[i%len(colors)]
            dxdy  = p.coords.dxdy
            
            points = np.squeeze(p.points) @ p.coords.rot_matrix() + dxdy
            graph.draw_lines(points, color=color, width=2)

            h = DragHandle(p.coords)
            h.render(graph)

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

        graph = self.window['graph']

        piece0 = self.pieces[0]
        piece1 = self.pieces[1]

        points0 = piece0.coords.to_global_xy(piece0.points)
        kdtree0 = scipy.spatial.KDTree(points0)
        
        points1 = piece1.coords.to_global_xy(piece1.points)
        kdtree1 = scipy.spatial.KDTree(points1)

        indexes = kdtree0.query_ball_tree(kdtree1, r=15)
        matches = [i for i, v in enumerate(indexes) if len(v)]

        print(f"{len(matches)} points in piece0 have a correspondence with piece1")

        keep = [matches[i] for i in range(0, len(matches), (len(matches)-4) // 4)]
        print(f"{keep=}")

        data0 = points0[keep]
        d, i = kdtree1.query(data0)
        data1 = points1[i]

        for xy in data0:
            graph.draw_point(tuple(xy), color='purple', size=17)

        self.keep0 = keep
        self.keep1 = i

        print(f"{data0=}")
        print(f"{data1=}")

        print(f"umeyama: {self.umeyama(data0, data1)}")
        print(f"eldridge: {self.eldridge(data0, data1)}")

    def do_refit(self):

        if self.keep0 is None or self.keep1 is None:
            return
        
        graph = self.window['graph']

        piece0 = self.pieces[0]
        piece1 = self.pieces[1]

        points0 = piece0.coords.to_global_xy(piece0.points)
        points1 = piece1.coords.to_global_xy(piece1.points)

        data0 = points0[self.keep0]
        data1 = points1[self.keep1]
        
        normal0 = np.array([piece0.normal_at_index(i) for i in self.keep0])
        normal1 = np.array([piece1.normal_at_index(i) for i in self.keep1])

        for i, xy in enumerate(data0):
            graph.draw_point(tuple(xy), color='purple', size=17)
            graph.draw_text(f"{i}", tuple(xy), color='purple')
            uv = np.array(normal0[i])
            graph.draw_line(tuple(xy), tuple(xy + uv*50), color='black')
            
        for i, xy in enumerate(data1):
            graph.draw_point(tuple(xy), color='purple', size=17)
            graph.draw_text(f"{i}", tuple(xy), color='purple')
            uv = np.array(normal1[i])
            graph.draw_line(tuple(xy), tuple(xy + uv*50), color='black')

        if False:
            for i in range(len(data0)):
                uv0 = normal0[i]
                uv1 = normal1[i]
                print(f"{i}: p0_xy={data0[i]} p0_uv={uv0[0]:+6.3f},{uv0[1]:+6.3f} p1_xy={data1[i]} p1_uv={uv1[0]:+6.3f},{uv1[1]:+6.3f}")

        # print(f"{data0=}")
        # print(f"{data1=}")

        # print(f"umeyama: {self.umeyama(data0, data1)}")
        # print(f"eldridge: {self.eldridge(data0, data1)}")

        data1 = data1 - piece1.coords.dxdy

        theta, tx, ty = self.icp(data0, normal0, data1)[0]
        dxdy = np.array((tx,ty))
        print(f"icp: theta={theta*180/math.pi:.3f} degrees, {dxdy=}")

        c = piece1.coords
        print(f"  before: {c.angle=} {c.dxdy=}")

        c.angle += theta
        c.dxdy   = dxdy

        print(f"  after:  {c.angle=} {c.dxdy=}")
        
        self.render()
        
    def _init_ui(self):
        
        w, h = 1024, 1024
        s = 3

        layout = [
            [sg.Graph(canvas_size=(w, h),
                      graph_bottom_left = (0, 0),
                      graph_top_right = (w * s, h * s),
                      background_color='white',
                      key='graph',
                      drag_submits=True,
                      enable_events=True)],
            [sg.Button('Fit', key='button_fit'),
             sg.Button('Refit', key='button_refit')]
        ]
        
        self.window = sg.Window('Align', layout, finalize=True, return_keyboard_events=True)
        self.render()

    def run(self):
        
        self._init_ui()

        while True:
            event, values = self.window.read()
            if event == sg.WIN_CLOSED:
                break
            elif event == 'graph':
                self.graph_drag(values['graph'], False)
            elif event == 'graph+UP':
                self.graph_drag(values['graph'], True)
            elif event == 'button_fit':
                self.do_fit()
            elif event == 'button_refit':
                self.do_refit()
            else:
                print(event, values)

def align_ui(args):

    puzzle = puzzler.file.load(args.puzzle)

    by_label = dict()
    for p in puzzle.pieces:
        by_label[p.label] = p

    pieces = [Piece(by_label[l]) for l in args.labels]
    
    ui = AlignUI(pieces)
    ui.run()

def add_parser(commands):

    parser_align = commands.add_parser("align", help="UI to experiment with aligning pieces")
    parser_align.add_argument("labels", nargs='+')
    parser_align.set_defaults(func=align_ui)
