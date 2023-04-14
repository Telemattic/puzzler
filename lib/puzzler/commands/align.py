import collections
import csv
import cv2 as cv
import itertools
import math
import numpy as np
import operator
import re
import scipy
import puzzler.feature
import puzzler
import puzzler.renderer.canvas
import puzzler.solver

from tkinter import *
from tkinter import ttk
from tqdm import tqdm

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
        self.coords = puzzler.align.AffineTransform()

class PuzzleRenderer:

    def __init__(self, canvas, camera, pieces):
        self.canvas = canvas
        self.camera = camera
        self.pieces = pieces
        self.selection = None
        self.frontiers = []
        self.adjacency = dict()
        self.renderer = None
        self.render_fast = None
        self.canvas_w = self.camera.viewport[0]
        self.canvas_h = self.camera.viewport[1]
        self.normals = dict()
        self.vertexes = dict()

    def render(self, render_fast):
        
        self.canvas.delete('all')
        self.renderer = puzzler.renderer.canvas.CanvasRenderer(self.canvas)
        self.renderer.transform(self.camera.matrix)

        self.render_fast = render_fast

        if self.adjacency:
            self.draw_adjacency(self.adjacency)

        colors = ['red', 'green', 'blue']
        for i, piece in enumerate(self.pieces):

            color = colors[i%len(colors)]
            self.draw_piece(piece, color, f"piece_{i}")

        if self.selection is not None:
            self.draw_rotate_handles(self.selection)

        if self.frontiers:
            for f in self.frontiers:
                self.draw_frontier(f)

    def test_bbox(self, bbox):

        ll, ur = bbox
        x0, y0 = ll
        x1, y1 = ur
        points = np.array([(x0,y0), (x1,y0), (x1,y1), (x0,y1)])

        screen = self.renderer.user_to_device(points)
        x = screen[:,0]
        y = screen[:,1]

        if np.max(x) < 0 or np.min(x) > self.canvas_w:
            return False

        if np.max(y) < 0 or np.min(y) > self.canvas_h:
            return False

        return True

    def draw_piece(self, p, color, tag):

        r = self.renderer
            
        with puzzler.render.save(r):
                
            r.translate(p.coords.dxdy)
            r.rotate(p.coords.angle)

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

            if p.piece.tabs and False:
                for i in range(len(p.piece.tabs)):
                    lcr = puzzler.align.TabAligner.get_tab_points(p.piece, i)
                    r.draw_points(lcr, fill='purple', radius=4)

            if self.render_fast:
                points = p.perimeter.points[p.approx.indexes]
            else:
                points = p.piece.points
                
            r.draw_polygon(points, outline=color, fill='', width=2, tag=tag)

            r.draw_text(np.array((0,0)), p.piece.label)

            normals = self.normals.get(p.piece.label)
            if normals is not None:
                for n in normals:
                    r.draw_lines(np.array((n[0], n[0] + n[1]*10)), fill='black', width=1)

            vertexes = self.vertexes.get(p.piece.label)
            if vertexes is not None:
                r.draw_points(vertexes, fill='', outline=color, radius=6)

    def draw_rotate_handles(self, piece_id):

        p = self.pieces[piece_id]

        r = self.renderer
        with puzzler.render.save(r):

            r.translate(p.coords.dxdy)
            r.rotate(p.coords.angle)

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
                with puzzler.render.save(r):
                    r.rotate(i * math.pi / 2)
                    r.draw_polygon(points, outline='black', fill='', width=1, tags=tags)
                    
    def draw_frontier(self, frontier):

        piece_dict = dict((i.piece.label, i.piece) for i in self.pieces)
        
        fe = puzzler.solver.FrontierExplorer(piece_dict, None)

        tabs = collections.defaultdict(list)
        for label, tab_no in fe.find_tabs(frontier):
            tabs[label].append(tab_no)

        r = self.renderer
        
        piece_dict = dict((i.piece.label, i) for i in self.pieces)
        
        for l, tab_nos in tabs.items():
            p = piece_dict[l]
            with puzzler.render.save(r):
                r.translate(p.coords.dxdy)
                r.rotate(p.coords.angle)
                for tab_no in tab_nos:
                    p0, v = fe.get_tab_center_and_direction((l, tab_no))
                    p1 = p0 + v * 100
                    r.draw_lines(np.array((p0, p1)), fill='red', width=1, arrow='last')

        return

        for l, a, b in frontier:
            p = piece_dict[l]
            with puzzler.render.save(r):
                r.translate(p.coords.dxdy)
                r.rotate(p.coords.angle)
                r.draw_points(p.piece.points[a], fill='pink', radius=8)
                
        for l, a, b in frontier:
            p = piece_dict[l]
            with puzzler.render.save(r):
                r.translate(p.coords.dxdy)
                r.rotate(p.coords.angle)
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
        with puzzler.render.save(r):
            r.translate(p.coords.dxdy)
            r.rotate(p.coords.angle)
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
        self.solver = puzzler.solver.PuzzleSolver({i.piece.label: i.piece for i in self.pieces}, None)

        self.draggable = None
        self.selection = None
        self.render_normals = None
        self.render_vertexes = None

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
        r.frontiers = self.solver.frontiers
        if self.var_render_adjacency.get():
            r.adjacency = self.solver.adjacency
        if self.render_normals:
            r.normals = self.render_normals
        if self.render_vertexes:
            r.vertexes = self.render_vertexes
        r.render(True)

        self.render_full += 1
        if 1 == self.render_full:
            self.parent.after_idle(self.full_render)

    def full_render(self):
        
        self.render_full = -1

        r = PuzzleRenderer(self.canvas, self.camera, self.pieces)
        r.selection = self.selection
        r.frontiers = self.solver.frontiers
        if self.var_render_adjacency.get():
            r.adjacency = self.solver.adjacency
        if self.render_normals:
            r.normals = self.render_normals
        if self.render_vertexes:
            r.vertexes = self.render_vertexes
        r.render(False)

        self.render_full = 0

    def do_tab_alignment(self):

        self.solver.solve_field()
        self.update_coords()
                    
        self.render()

        if self.var_solve_continuous.get():
            if self.solver.corners:
                self.parent.after_idle(self.do_tab_alignment)
            else:
                self.var_solve_continuous.set(0)

    def update_coords(self):
        
        g = self.solver.geometry
        if not g:
            return
        
        for p in self.pieces:
            if c := g.coords.get(p.piece.label):
                p.coords = c

    def load_geometry(self, path):

        self.solver.load_geometry(path)
        self.solver.update_adjacency()
        self.update_coords()

    def do_solve(self):

        self.solver.solve_border()
        self.update_coords()
                    
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

    def _init_ui(self, parent):

        w, h = parent.winfo_screenwidth(), parent.winfo_screenheight()
        viewport = (min(w-32,1024), min(h-128,1024))

        self.parent = parent
        self.camera = Camera(np.array((0,0), dtype=np.float64), 1/3, viewport)
        
        self.frame = ttk.Frame(parent, padding=5)
        self.frame.grid(column=0, row=0, sticky=(N, W, E, S))
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(0, weight=1)
        self.frame.grid_columnconfigure(0, weight=1)
        self.frame.grid_rowconfigure(0, weight=1)

        self.canvas = Canvas(self.frame, width=viewport[0], height=viewport[1],
                             background='white', highlightthickness=0)
        self.canvas.grid(column=0, row=0, sticky=(N, W, E, S))
        self.canvas.bind("<Button-1>", self.canvas_press)
        self.canvas.bind("<B1-Motion>", self.canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.canvas_release)
        self.canvas.bind("<MouseWheel>", self.mouse_wheel)
        self.canvas.bind("<Motion>", self.motion)
        self.canvas.bind("<Configure>", self.resize)

        self.controls = ttk.Frame(self.frame)
        self.controls.grid(row=1, sticky=(W,E))

        b1 = ttk.Button(self.controls, text='Solve!', command=self.do_solve)
        b1.grid(column=0, row=0, sticky=W)

        b2 = ttk.Button(self.controls, text='Tab Alignment', command=self.do_tab_alignment)
        b2.grid(column=1, row=0, sticky=W)

        self.var_render_adjacency = IntVar(value=1)
        b3 = ttk.Checkbutton(self.controls, text="Adjacency", command=self.render,
                             variable=self.var_render_adjacency)
        b3.grid(column=2, row=0, sticky=W)

        self.var_solve_continuous = IntVar(value=0)
        b4 = ttk.Checkbutton(self.controls, text="Continuous", variable=self.var_solve_continuous)
        b4.grid(column=3, row=0, sticky=W)

        self.var_label = StringVar(value="x,y")
        l1 = ttk.Label(self.controls, textvariable=self.var_label, width=80)
        l1.grid(column=4, row=0, sticky=(E))

        cf2 = ttk.Frame(self.frame)
        cf2.grid(row=2, sticky=(W,E))

        b5 = ttk.Button(cf2, text='Show Tab Alignment', command=self.show_tab_alignment)
        b5.grid(column=0, row=0, sticky=W)

        self.var_show_tab_alignment = StringVar(value='')
        e1 = ttk.Entry(cf2, width=16, textvariable=self.var_show_tab_alignment)
        e1.grid(column=1, row=0)

        self.render_full = False
        self.render()

    def resize(self, e):
        viewport = (e.width, e.height)
        if self.camera.viewport != viewport:
            self.camera.viewport = viewport
            self.render()

    def show_tab_alignment(self):
        s = self.var_show_tab_alignment.get().strip()
        m = re.fullmatch("([a-zA-Z]+\d+):(\d+)-([a-zA-Z]+\d+):(\d+)", s)
        if not m:
            print(f"bad input: \"{s}\"")
            return
        
        dst_label = m[1]
        dst_tab_no = int(m[2])
        src_label = m[3]
        src_tab_no = int(m[4])
        
        print(f"{dst_label=} {dst_tab_no=} {src_label=} {src_tab_no=}")

        pieces = dict([(i.piece.label, i) for i in self.pieces])
        dst = pieces[dst_label]
        src = pieces[src_label]
        
        tab_aligner = puzzler.align.TabAligner(dst.piece)

        mse, src_coords, sfp, dfp = tab_aligner.compute_alignment(dst_tab_no, src.piece, src_tab_no)
        print(f"{mse=} {src_coords=} {sfp=} {dfp=}")

        src_mid = tab_aligner.get_tab_midpoint(src.piece, src_tab_no)

        mse, src_coords, sfp, dfp = tab_aligner.refine_alignment(src.piece, src_coords, src_mid)
        print(f"{mse=} {src_coords=} {sfp=} {dfp=}")
        
        mse, src_coords, sfp, dfp = tab_aligner.refine_alignment(src.piece, src_coords, src_mid)
        print(f"{mse=} {src_coords=} {sfp=} {dfp=}")

        mse, src_coords, sfp, dfp = tab_aligner.compute_alignment(dst_tab_no, src.piece, src_tab_no, refine=2)
        print(f"refine=2: {mse=} {src_coords=} {sfp=} {dfp=}")
        
        self.render_vertexes = dict()
        self.render_vertexes[src_label] = tab_aligner.src_vertexes
        self.render_vertexes[dst_label] = tab_aligner.dst_vertexes
        
        self.render_normals = dict()
        self.render_normals[dst_label] = list(zip(tab_aligner.dst_vertexes, tab_aligner.dst_normals))

        dst.coords = puzzler.align.AffineTransform(0., (0., 2000.))
        src.coords = puzzler.align.AffineTransform(src_coords.angle, src_coords.dxdy + dst.coords.dxdy)

        self.render()
                
def align_ui(args):

    puzzle = puzzler.file.load(args.puzzle)

    by_label = dict()
    for p in puzzle.pieces:
        by_label[p.label] = p

    if 'I1' in by_label:
        p = by_label['I1']
        if len(p.edges) == 2:
            p.edges = p.edges[::-1]

    labels = set(args.labels)

    if args.edges:
        labels |= set('A1 A2 A3 A4 A5 A6 A7 A8 A9 A10 A11 B1 B11 C1 C11 D1 D11 E1 E12 F1 F11 G1 G11 H1 H11 I1 I2 I3 I4 I5 I6 I7 I8 I9 I10 I11'.split())

    if not labels:
        labels |= set(by_label.keys())

    pieces = [Piece(by_label[l]) for l in sorted(labels)]

    def to_row(s):
        row = 0
        for i in s.upper():
            row *= 26
            row += ord(i) + 1 - ord('A')
        return row

    def to_col(s):
        return int(s)

    def to_row_col(label):
        m = re.fullmatch("([a-zA-Z]+)(\d+)", label)
        return (to_row(m[1]), to_col(m[2])) if m else (None, None)

    rows = set()
    cols = set()
    for piece in pieces:
        r, c = to_row_col(piece.piece.label)
        rows.add(r)
        cols.add(c)

    rows = dict((r, i) for i, r in enumerate(sorted(rows)))
    cols = dict((c, i) for i, c in enumerate(sorted(cols)))

    for piece in pieces:
        r, c = to_row_col(piece.piece.label)
        x = cols[c] * 1000.
        y = rows[r] * -1000.
        piece.coords.dxdy = np.array((x, y))

    root = Tk()
    ui = AlignTk(root, pieces)
    root.bind('<Key-Escape>', lambda e: root.destroy())
    root.title("Puzzler: align")
    # root.wm_resizable(0, 0)

    if args.geometry:
        ui.load_geometry(args.geometry)
    
    root.mainloop()

def output_tabs(args):

    puzzle = puzzler.file.load(args.puzzle)

    print("Tab alignment!")

    def to_row(s):
        row = 0
        for i in s.upper():
            row *= 26
            row += ord(i) + 1 - ord('A')
        return row

    def to_col(s):
        return int(s)

    def to_row_col(label):
        m = re.fullmatch("([a-zA-Z]+)(\d+)", label)
        return (to_row(m[1]), to_col(m[2])) if m else (None, None)

    def sort_key(piece):
        row, col = to_row_col(piece.label)
        return (row, col, piece.label)

    pieces = sorted([i for i in puzzle.pieces], key=sort_key)
    
    num_indents = 0
    num_outdents = 0
    for p in pieces:
        for t in p.tabs:
            if t.indent:
                num_indents += 1
            else:
                num_outdents += 1
    
    print(f"{len(pieces)} pieces: {num_indents} indents, {num_outdents} outdents")

    with open(args.output, 'w', newline='') as f:
        field_names = 'dst_label dst_tab_no dst_col_no dst_row_no src_label src_tab_no src_col_no src_row_no src_coord_x src_coord_y src_coord_angle src_index_0 src_index_1 mse neighbor rank'.split()
        writer = csv.DictWriter(f, field_names)
        writer.writeheader()

        for dst in tqdm(pieces, ascii=True):
            tab_aligner = puzzler.align.TabAligner(dst)
            for dst_tab_no, dst_tab in enumerate(dst.tabs):
                rows = []
                for src in pieces:
                    if src is dst:
                        continue
                    for src_tab_no, src_tab in enumerate(src.tabs):
                        if dst_tab.indent == src_tab.indent:
                            continue
                        dst_row_no, dst_col_no = to_row_col(dst.label)
                        src_row_no, src_col_no = to_row_col(src.label)
                        neighbor = None
                        if dst_row_no == src_row_no:
                            if dst_col_no == src_col_no - 1:
                                neighbor = 'E'
                            if dst_col_no == src_col_no + 1:
                                neighbor = 'W'
                        elif dst_col_no == src_col_no:
                            if dst_row_no == src_row_no - 1:
                                neighbor = 'S'
                            if dst_row_no == src_row_no + 1:
                                neighbor = 'N'
                            
                        mse, src_coords, sfp, dfp = tab_aligner.compute_alignment(dst_tab_no, src, src_tab_no, refine=args.refine)
                        rows.append({'dst_label': dst.label,
                                     'dst_tab_no': dst_tab_no,
                                     'dst_col_no': dst_col_no,
                                     'dst_row_no': dst_row_no,
                                     'src_label': src.label,
                                     'src_tab_no': src_tab_no,
                                     'src_col_no': src_col_no,
                                     'src_row_no': src_row_no,
                                     'src_coord_x': src_coords.dxdy[0],
                                     'src_coord_y': src_coords.dxdy[1],
                                     'src_coord_angle': src_coords.angle,
                                     'src_index_0': sfp[0],
                                     'src_index_1': sfp[1],
                                     'mse': mse,
                                     'neighbor': neighbor})

                for i, row in enumerate(sorted(rows, key=lambda row: 10000 if row['mse'] is None else row['mse']), start=1):
                    row['rank'] = i
                writer.writerows(rows)

def add_parser(commands):

    parser_align = commands.add_parser("align", help="UI to experiment with aligning pieces")
    parser_align.add_argument("labels", nargs='*')
    parser_align.add_argument("-e", "--edges", help="add all edges", action='store_true')
    parser_align.add_argument("-g", "--geometry", help="geometry file")
    parser_align.set_defaults(func=align_ui)

    parser_tabs = commands.add_parser("tabs", help="Output a CSV enumerating all possible tab matches")
    parser_tabs.add_argument("-o", "--output", help="output csv path")
    parser_tabs.add_argument("-r", "--refine", help="number of refinement passes", default=0, type=int)
    parser_tabs.set_defaults(func=output_tabs)
