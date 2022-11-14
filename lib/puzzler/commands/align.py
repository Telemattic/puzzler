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

    def __len__(self):
        a, b = self.a, self.b
        return (b - a) if b > a else (b + self.n - a)
        
def ring_slice(data, a, b):
    return np.concatenate((data[a:], data[:b])) if a >= b else data[a:b]

def ring_range(a, b, n):
    return itertools.chain(range(a, n), range(0, b)) if a >= b else range(a, b)

class Piece:

    def __init__(self, piece):

        self.piece = piece
        self.perimeter = Perimeter(self.piece.points)
        self.approx = ApproxPoly(self.perimeter, 10)
        self.coords = AffineTransform()

    def normal_at_index(self, i):
        uv = self.approx.normal_at_index(i)
        return uv @ self.coords.rot_matrix().T

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

class PuzzleRenderer:

    def __init__(self, canvas, camera, pieces):
        self.canvas = canvas
        self.camera = camera
        self.pieces = pieces
        self.selection = None
        self.frontier = []
        self.adjacency = dict()
        self.renderer = None
        self.render_fast = None
        self.canvas_w = int(self.canvas.cget('width'))
        self.canvas_h = int(self.canvas.cget('height'))

    def render(self, render_fast):
        
        self.canvas.delete('all')
        self.renderer = puzzler.render.Renderer(self.canvas)
        self.renderer.transform.multiply(self.camera.matrix)

        self.render_fast = render_fast

        if self.adjacency:
            self.draw_adjacency(self.adjacency)

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

            if not self.test_bbox(p.piece.bbox):
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

    def draw_adjacency(self, adjacency):
        
        self.piece_dict = dict((i.piece.label, i) for i in self.pieces)
        
        fills = ['purple', 'pink', 'yellow', 'orange', 'cyan']
        i = 0
        
        for k1, v1 in adjacency.items():
            for k2, v2 in v1.items():
                self.draw_adjacency_list((k1, k2), v2, fills[i])
                i = (i + 1) % len(fills)

    def draw_adjacency_list(self, k, v, fill):

        src, dst = k
        p = self.piece_dict[src]

        tag = src + ":" + dst

        r = self.renderer
        with puzzler.render.save_matrix(r.transform):
            r.transform.translate(p.coords.dxdy).rotate(p.coords.angle)
            for a, b in v:
                points = ring_slice(p.piece.points, a, b+1)
                if len(points) < 2:
                    # print(f"{src=} {dst=} {a=} {b=} {points=}")
                    r.draw_points(points, fill=fill, radius=8, tag=tag)
                else:
                    r.draw_lines(points, fill=fill, width=8, tag=tag)

class AlignTk:

    def __init__(self, parent, pieces):
        self.pieces = pieces

        self.draggable = None
        self.selection = None

        self.keep0 = None
        self.keep1 = None

        self.frontier = None
        self.adjacency = None

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
        r.adjacency = self.adjacency
        r.render(True)

        self.render_full += 1
        if 1 == self.render_full:
            self.parent.after_idle(self.full_render)

    def full_render(self):
        
        self.render_full = -1

        r = PuzzleRenderer(self.canvas, self.camera, self.pieces)
        r.selection = self.selection
        r.frontier = self.frontier
        r.adjacency = self.adjacency
        r.render(False)

        self.render_full = 0

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

        best_fit = None

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

            if best_fit is None or mse < best_fit[0]:
                best_fit = (mse, src_piece.piece.label, src_coords)
            
        print(f"{len(scores[0])=} {len(scores[1])=} {len(allways)=}")

        print(f"{best_fit=}")

        if best_fit:
            piece_dict[best_fit[1]].coords = best_fit[2]
            self.render()
                
    def do_solve(self):

        pieces_dict = dict((i.piece.label, i) for i in self.pieces)
        
        bs = puzzler.solver.BorderSolver({i.piece.label: i.piece for i in self.pieces})

        scores = bs.score_matches()
        border = bs.link_pieces(scores)
        print(f"{border=}")
        bs.estimate_puzzle_size(border)

        constraints = bs.init_constraints(border)
        geometry = bs.init_placement(border)

        print(f"{constraints=}")
        print(f"{geometry=}")

        for i in self.pieces:
            coords = geometry.coords.get(i.piece.label)
            if coords:
                print(i.piece.label)
                i.coords = coords

        pieces_dict = dict((i.piece.label, i) for i in self.pieces)

        # border = [i for i, _ in pairs]

        # kdtrees = {i.piece.label: scipy.spatial.KDTree(i.piece.points) for i in self.pieces}
        ac = puzzler.solver.AdjacencyComputer(bs.pieces, constraints, geometry)
        self.adjacency = dict()
        for label in geometry.coords:
            self.adjacency[label] = ac.compute_adjacency(label)
        print(self.adjacency)
        
        self.frontier = []
        for k, v in self.adjacency.items():
            for r in v.get('none', []):
                self.frontier.append((k, *r))

        print(self.frontier)

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

        b1 = ttk.Button(self.controls, text='Solve!', command=self.do_solve)
        b1.grid(column=0, row=0, sticky=W)

        b2 = ttk.Button(self.controls, text='Tab Alignment', command=self.do_tab_alignment_B2)
        b2.grid(column=1, row=0, sticky=W)

        self.var_label = StringVar(value="x,y")
        l1 = ttk.Label(self.controls, textvariable=self.var_label, width=40)
        l1.grid(column=2, row=0, sticky=(E))

        self.render_full = False
        self.render()
                
def align_ui(args):

    puzzle = puzzler.file.load(args.puzzle)

    by_label = dict()
    for p in puzzle.pieces:
        by_label[p.label] = p

    if 'I1' in by_label:
        p = by_label['I1']
        p.edges = p.edges[::-1]

    labels = set(args.labels)

    if args.edges:
        labels |= set('A1 A2 A3 A4 A5 A6 A7 A8 A9 A10 A11 B1 B11 C1 C11 D1 D11 E1 E12 F1 F11 G1 G11 H1 H11 I1 I2 I3 I4 I5 I6 I7 I8 I9 I10 I11'.split())

    if not labels:
        labels |= set(by_label.keys())

    pieces = [Piece(by_label[l]) for l in sorted(labels)]

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
