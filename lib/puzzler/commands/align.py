import bisect
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

class FrontierExplorer:

    def __init__(self, pieces, geometry):
        self.pieces = pieces
        self.geometry = geometry

    def find_tabs(self, frontier):

        retval = []
        
        for l, a, b in frontier:
            p = self.pieces[l]
            rr = RingRange(a, b, len(p.points))
            
            included_tabs = [i for i, tab in enumerate(p.tabs) if all(j in rr for j in tab.tangent_indexes)]

            def position_in_ring(i):
                tab = p.tabs[i]
                begin = tab.tangent_indexes[0]
                if begin < rr.a:
                    begin += rr.n
                return begin

            included_tabs.sort(key=position_in_ring)

            retval += [(l, i) for i in included_tabs]

        return retval

    def find_interesting_corners(self, frontier):

        tabs = self.find_tabs(frontier)
        dirs = []
        for tab in tabs:
            p, v = self.get_tab_center_and_direction(tab)
            t = self.geometry.coords[tab[0]].get_transform()
            dirs.append((t.apply_v2(p), t.apply_n2(v)))
            
        scores = []
        for i, curr_dir in enumerate(dirs):
            p1, v2 = dirs[i-1]
            p3, v4 = curr_dir
            t = np.cross(p3 - p1, v4) / np.cross(v2, v4)
            u = np.cross(p3 - p1, v2) / np.cross(v2, v4)
            scores.append((np.dot(v2, v4), t, u))
            
        return [(scores[i], tabs[i-1], tabs[i]) for i in range(len(tabs))]

    def get_tab_center_and_direction(self, tab):

        p = self.pieces[tab[0]]
        t = p.tabs[tab[1]]
        v = p.points[np.array(t.tangent_indexes)] - t.ellipse.center
        v = v / np.linalg.norm(v, axis=1)
        v = np.sum(v, axis=0)
        v = v / np.linalg.norm(v)
        if not t.indent:
            v = -v
        return (t.ellipse.center, v)

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

        piece_dict = dict((i.piece.label, i.piece) for i in self.pieces)
        
        fe = FrontierExplorer(piece_dict, None)

        tabs = collections.defaultdict(list)
        for label, tab_no in fe.find_tabs(frontier):
            tabs[label].append(tab_no)

        r = self.renderer
        
        piece_dict = dict((i.piece.label, i) for i in self.pieces)
        
        for l, tab_nos in tabs.items():
            p = piece_dict[l]
            with puzzler.render.save_matrix(r.transform):
                r.transform.translate(p.coords.dxdy).rotate(p.coords.angle)
                for tab_no in tab_nos:
                    p0, v = fe.get_tab_center_and_direction((l, tab_no))
                    p1 = p0 + v * 100
                    r.draw_lines(np.array((p0, p1)), fill='red', width=1, arrow='last')

        return

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
        self.geometry = None

        self.draggable = None
        self.selection = None

        self.frontier = None
        self.adjacency = None
        self.corners = []

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
        if self.var_render_adjacency.get():
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
        if self.var_render_adjacency.get():
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

    def score_corner(self, corner):

        print(f"score_corner: {corner[0]} {corner[1]}")
        
        scores = []

        piece_dict = dict((i.piece.label, i.piece) for i in self.pieces)

        for dst_label, dst_tab_no in corner:

            dst_piece = piece_dict[dst_label]
            dst_tab   = dst_piece.tabs[dst_tab_no]

            the_scores = dict()
            tab_aligner = puzzler.align.TabAligner(dst_piece)
            
            for src_piece in piece_dict.values():
                
                if src_piece.label in self.geometry.coords:
                    continue
                
                if src_piece is dst_piece:
                    continue
                
                scores_for_src = []
                for src_tab_no, src_tab in enumerate(src_piece.tabs):
                    if dst_tab.indent == src_tab.indent:
                        continue
                    mse, _, sfp, _ = tab_aligner.compute_alignment(dst_tab_no, src_piece, src_tab_no)
                    a, b = sfp
                    n = b-a if b > a else len(src_piece.points)-a+b
                    scores_for_src.append((mse, n, src_tab_no))
                    
                the_scores[src_piece.label] = scores_for_src

            scores.append(the_scores)

        allways = []
        for src_label, scores0 in scores[0].items():
            scores1 = scores[1][src_label]
            for a, b in itertools.product(scores0, scores1):
                mse0, n0, tab0 = a
                mse1, n1, tab1 = b
                if tab0 == tab1:
                    continue
                mse = (mse0 * n0 + mse1 * n1) / (n0 + n1)
                allways.append((mse, src_label, tab0, tab1))

        allways.sort()
        
        aligner = puzzler.align.MultiAligner(corner, piece_dict, self.geometry)

        fits = []

        for mse, src_label, src_tab0, src_tab1 in allways:
            
            src_tabs  = [src_tab0, src_tab1]
            
            src_piece = piece_dict[src_label]
            
            src_coords = aligner.compute_alignment(src_piece, src_tabs)

            mse = aligner.measure_fit(src_piece, src_tabs, src_coords)

            fits.append((mse, src_label, src_coords))

        fits.sort(key=operator.itemgetter(0))

        return fits

    def do_tab_alignment_B2(self):

        if self.geometry is None:
            return

        if not self.corners:
            return

        fits = []
        for corner in self.corners:
            fits += self.score_corner(corner)[:10]

        fits.sort(key=operator.itemgetter(0))

        for i, f in enumerate(fits[:10]):
            mse, src_label, src_coords = f
            with np.printoptions(precision=1):
                print(f"{i}: {src_label} angle={src_coords.angle:.3f} xy={src_coords.dxdy} {mse=:.1f}")

        if fits:
            piece_dict = dict((i.piece.label, i) for i in self.pieces)
            _, label, coords = fits[0]
            piece_dict[label].coords = coords
            self.geometry.coords[label] = coords
            self.update_adjacency()
            self.render()

    def update_adjacency(self):
        
        pieces_dict = {i.piece.label: i.piece for i in self.pieces}
        
        ac = puzzler.solver.AdjacencyComputer(pieces_dict, [], self.geometry)
        
        adjacency = dict()
        for label in self.geometry.coords:
            adjacency[label] = ac.compute_adjacency(label)

        successors, neighbors, nodes_on_frontier = ac.compute_successors_and_neighbors(adjacency)

        frontiers, fullpaths = ac.find_frontiers(successors, neighbors, nodes_on_frontier)
        
        frontier = [(l, *ab) for l, ab in frontiers[0]]

        def flatten(i):
            a, b = i
            return f"{a}:{b[0]}-{b[1]}"

        def flatten_dict(d):
            return {flatten(k): flatten(v) for k, v in d.items()}

        def flatten_list(l):
            return [flatten(i) for i in l]

        fe = FrontierExplorer(pieces_dict, self.geometry)
        corners = fe.find_interesting_corners(frontier)
        good_corners = []
        for (s, t, u), tab0, tab1 in corners:
            is_interesting = abs(s) < .5 and 50 < t < 1000 and 50 < u < 1000
            # print(f"{tab0}, {tab1}: {s=:.3f} {t=:.1f} {u=:.1f}", end='')
            # if is_interesting:
            #     print(" ***")
            # else:
            #     print()
            if is_interesting:
                good_corners.append((tab0, tab1))

        self.adjacency = adjacency
        self.frontier = frontier
        self.corners = good_corners

        with open(r'C:\temp\puzzler\update_adjacency.txt','a') as f:
            print(f"successors={flatten_dict(successors)}", file=f)
            print(f"neighbors={flatten_dict(neighbors)}", file=f)
            print(f"nodes_on_frontier={flatten_list(nodes_on_frontier)}", file=f)
            for i, j in enumerate(frontiers):
                print(f"frontiers[{i}]={flatten_list(j)}", file=f)
            for i, j in enumerate(fullpaths):
                print(f"fullpaths[{i}]={flatten_list(j)}", file=f)
            print(f"tabs={fe.find_tabs(frontier)}", file=f)
            print(f"corners={corners}", file=f)
            print(f"good_corners={good_corners}", file=f)

    def do_solve(self):

        pieces_dict = dict((i.piece.label, i) for i in self.pieces)
        
        bs = puzzler.solver.BorderSolver({i.piece.label: i.piece for i in self.pieces})

        scores = bs.score_matches()
        border = bs.link_pieces(scores)
        print(f"{border=}")
        bs.estimate_puzzle_size(border)

        constraints = bs.init_constraints(border)
        self.geometry = bs.init_placement(border)

        print(f"{constraints=}")
        print(f"{self.geometry=}")

        for i in self.pieces:
            coords = self.geometry.coords.get(i.piece.label)
            if coords:
                print(i.piece.label)
                i.coords = coords

        self.update_adjacency()

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
        self.frame.grid(sticky=(N, W, E, S))

        self.canvas = Canvas(self.frame, width=viewport[0], height=viewport[1],
                             background='white', highlightthickness=0)
        self.canvas.grid(column=0, row=0, sticky=(N, W, E, S))
        self.canvas.bind("<Button-1>", self.canvas_press)
        self.canvas.bind("<B1-Motion>", self.canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.canvas_release)
        self.canvas.bind("<MouseWheel>", self.mouse_wheel)
        self.canvas.bind("<Motion>", self.motion)

        self.controls = ttk.Frame(self.frame)
        self.controls.grid(row=1, sticky=(W,E))

        b1 = ttk.Button(self.controls, text='Solve!', command=self.do_solve)
        b1.grid(column=0, row=0, sticky=W)

        b2 = ttk.Button(self.controls, text='Tab Alignment', command=self.do_tab_alignment_B2)
        b2.grid(column=1, row=0, sticky=W)

        self.var_render_adjacency = IntVar(value=1)
        b3 = ttk.Checkbutton(self.controls, text="Adjacency", command=self.render,
                             variable=self.var_render_adjacency)
        b3.grid(column=2, row=0, sticky=W)

        self.var_label = StringVar(value="x,y")
        l1 = ttk.Label(self.controls, textvariable=self.var_label, width=80)
        l1.grid(column=3, row=0, sticky=(E))

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
