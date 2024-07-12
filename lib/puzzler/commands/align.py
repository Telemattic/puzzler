import collections
import csv
import cv2 as cv
import decimal
import itertools
import math
import numpy as np
import operator
import palettable.tableau
import re
import scipy
import time
import traceback
import puzzler.feature
import puzzler
import puzzler.raft
import puzzler.renderer.canvas
import puzzler.solver
from tqdm import tqdm

from tkinter import *
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

    def __init__(self, piece, epsilon=None):

        self.piece = piece
        if epsilon is not None:
            self.perimeter = Perimeter(self.piece.points)
            self.approx = ApproxPoly(self.perimeter, epsilon)
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
        self.font = None
        self.render_fast = None
        self.canvas_w = self.camera.viewport[0]
        self.canvas_h = self.camera.viewport[1]
        self.normals = dict()
        self.vertexes = dict()
        self.use_cairo = False

    def render(self, render_fast):
        
        if self.use_cairo:
            self.renderer = puzzler.renderer.cairo.CairoRenderer(self.canvas)
        else:
            self.canvas.delete('all')
            self.renderer = puzzler.renderer.canvas.CanvasRenderer(self.canvas)
            
        self.renderer.transform(self.camera.matrix)

        # create the font *after* applying the camera transformation so that it is sized in screen space!
        self.font = self.renderer.make_font('Courier New', 18)
        
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

        return self.renderer.commit()

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
                # ll, ur = p.piece.bbox
                # x0, y0 = ll
                # x1, y1 = ur
                # points = np.array([(x0,y0), (x1,y0), (x1,y1), (x0,y1)])
                points = p.perimeter.points[p.approx.indexes]
            else:
                points = p.piece.points
                
            r.draw_polygon(points, outline=color, fill='', width=1, tags=tag)

            normals = self.normals.get(p.piece.label)
            if normals is not None:
                for n in normals:
                    r.draw_lines(np.array((n[0], n[0] + n[1]*10)), fill='black', width=1)

            vertexes = self.vertexes.get(p.piece.label)
            if vertexes is not None:
                r.draw_points(vertexes, fill='', outline=color, radius=6)

        # we don't want the text rotated
        r.draw_text(p.coords.dxdy, p.piece.label, font=self.font, tags=tag)

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
        
        fe = puzzler.solver.FrontierExplorer(piece_dict)

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
                    
class PuzzleSGFactory:

    # HACK: initialized as singleton on first construction of PuzzleSGFactory
    piece_factory = None

    def __init__(self, pieces):
        self.pieces = pieces
        self.selection = None
        self.frontiers = []
        self.render_frontier_details = False
        self.render_vertex_details = False
        self.adjacency = dict()
        self.renderer = None
        self.font = None
        self.normals = dict()
        self.vertexes = dict()
        self.seams = []
        self.props = {'tabs.render':False, 'edges.render':False, 'tabs.ellipse.fill':''}

        if PuzzleSGFactory.piece_factory is None:
            pieces_dict = dict((i.piece.label, i.piece) for i in pieces)
            PuzzleSGFactory.piece_factory = puzzler.sgbuilder.PieceSceneGraphFactory(
                pieces_dict)

    def build(self):

        self.scenegraphbuilder = puzzler.sgbuilder.SceneGraphBuilder()
            
        if self.adjacency:
            self.draw_adjacency(self.adjacency)

        colors = [tuple(c/255 for c in color) for color in palettable.tableau.Tableau_20.colors]
        for i, piece in enumerate(self.pieces):

            color = colors[i%len(colors)]
            color = colors[hash(piece.piece.label)%len(colors)]
            self.draw_piece(piece, color, f"piece_{i}")

        if self.selection is not None:
            self.draw_rotate_handles(self.selection)

        if self.frontiers:
            for f in self.frontiers:
                self.draw_frontier(f)

        sg = self.scenegraphbuilder.commit(None, None)

        lodf = puzzler.sgbuilder.LevelOfDetailFactory()
        sg.root_node = lodf.visit_node(sg.root_node)

        return sg

    def draw_piece(self, p, color, tag):

        sgb = self.scenegraphbuilder
            
        with puzzler.sgbuilder.insert_sequence(sgb):
                
            sgb.add_translate(p.coords.dxdy)

            sgb.add_rotate(p.coords.angle)

            props = self.props | {'points.outline':color, 'points.fill':color+(0.25,), 'tabs.ellipse.outline':color, 'tags':(tag,)}
            sgb.add_node(PuzzleSGFactory.piece_factory(p.piece.label, props))

        # global coordinate system!
        self.draw_normals_and_vertexes_and_seams(p, color, tag)

    def draw_normals_and_vertexes_and_seams(self, p, color, tag):

        sgb = self.scenegraphbuilder

        def draw_vertexes(vertexes, color):
            vmap = collections.defaultdict(list)
            for i, v in enumerate(vertexes):
                vmap[tuple(v)].append(i)
            for i, v in enumerate(vertexes):
                indices = vmap[tuple(v)]
                if indices[0] != i:
                    continue
                label = ','.join(str(i) for i in indices)
                sgb.add_points([v], radius=6, fill='', outline=color, width=1,
                               tags=(tag, f'vertex:{label}'))
                sgb.add_text(v, label, font=('Courier New', 12))

        def draw_normals(normals, color):
            for n in normals:
                sgb.add_lines(np.array((n[0], n[0] + n[1]*10)), fill=color, width=1)

        def draw_stitches(stitches, color, normals_flag):
            xy = np.array((0., 2000.))
            if self.render_vertex_details:
                draw_vertexes(stitches.points + xy, color)
            if normals_flag:
                normals = list(zip(stitches.points + xy, stitches.normals))
                draw_normals(normals, color)

        if self.seams:
            for seam in self.seams:
                if seam.src.piece == p.piece.label:
                    draw_stitches(seam.src, color, False)
                if seam.dst.piece == p.piece.label:
                    draw_stitches(seam.dst, color, True)
        
        normals = self.normals.get(p.piece.label)
        if normals is not None:
            draw_normals(normals, color)

        vertexes = self.vertexes.get(p.piece.label)
        if vertexes is not None:
            draw_vertexes(vertexes, color)

    def draw_rotate_handles(self, piece_id):

        p = self.pieces[piece_id]
        sgb = self.scenegraphbuilder

        with puzzler.sgbuilder.insert_sequence(sgb):

            sgb.add_translate(p.coords.dxdy)
            sgb.add_rotate(p.coords.angle)

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
                with puzzler.sgbuilder.insert_sequence(sgb):
                    sgb.add_rotate(i * math.pi / 2)
                    sgb.add_polygon(points, outline='black', fill='', width=1, tags=tags)
                    
    def draw_frontier(self, frontier):

        piece_dict = dict((i.piece.label, i.piece) for i in self.pieces)
        
        fe = puzzler.solver.FrontierExplorer(piece_dict)

        tabs = collections.defaultdict(list)
        for label, tab_no in fe.find_tabs(frontier):
            tabs[label].append(tab_no)

        piece_dict = dict((i.piece.label, i) for i in self.pieces)

        sgb = self.scenegraphbuilder
        
        for l, tab_nos in tabs.items():
            p = piece_dict[l]
            with puzzler.sgbuilder.insert_sequence(sgb):
                sgb.add_translate(p.coords.dxdy)
                sgb.add_rotate(p.coords.angle)
                for tab_no in tab_nos:
                    p0, v = fe.get_tab_center_and_direction((l, tab_no))
                    p1 = p0 + v * 100
                    sgb.add_lines(np.array((p0, p1)), fill='red', width=1, arrow='last')

        if not self.render_frontier_details:
            return

        for l, (a, b) in frontier:
            p = piece_dict[l]
            with puzzler.sgbuilder.insert_sequence(sgb):
                sgb.add_translate(p.coords.dxdy)
                sgb.add_rotate(p.coords.angle)
                sgb.add_points([p.piece.points[b]], radius=10, outline='purple')
                
        for i, (l, (a, b)) in enumerate(frontier):
            p = piece_dict[l]
            with puzzler.sgbuilder.insert_sequence(sgb):
                sgb.add_translate(p.coords.dxdy)
                sgb.add_rotate(p.coords.angle)
                v = p.piece.points[a]
                sgb.add_points([v], radius=8, fill='pink')
                label = str(i)
                sgb.add_text(v, label, font=('Courier New', 12))

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

        sgb = self.scenegraphbuilder

        with puzzler.sgbuilder.insert_sequence(sgb):
            sgb.add_translate(p.coords.dxdy)
            sgb.add_rotate(p.coords.angle)
            for a, b in v:
                points = ring_slice(p.piece.points, a, b+1)
                if len(points) < 2:
                    # print(f"{src=} {dst=} {a=} {b=} {points=}")
                    sgb.add_points(points, radius=8, fill=fill, tags=(tag,))
                else:
                    sgb.add_lines(points, width=8, fill=fill, tags=(tag,))

class CanvasHitTester:

    def __init__(self, canvas):
        self.canvas = canvas

    def __call__(self, xy):
        x, y = xy
        oids = self.canvas.find('overlapping', x-1, y-1, x+1, y+1)
        return [self.canvas.gettags(i) for i in oids]

class AlignTk:

    def __init__(self, parent, pieces, renderer, mode):
        self.pieces = pieces
        self.solver = puzzler.solver.PuzzleSolver({i.piece.label: i.piece for i in self.pieces}, None)

        self.draggable = None
        self.selection = None
        self.render_normals = None
        self.render_vertexes = None
        self.render_seams = None
        self.use_cairo = renderer == 'cairo'
        self.use_scenegraph = mode == 'scenegraph'
        self.scenegraph = None
        self.hittester = None

        self._init_ui(parent)

    def device_to_user(self, xy):
        xy = np.array((*xy, 1)) @ np.linalg.inv(self.camera.matrix).T
        return xy[:2]
    
    def canvas_press(self, event):

        piece_no  = None
        drag_type = 'move'

        if self.hittester:
            xy = self.device_to_user((event.x, event.y))
            tags = self.hittester(xy)
            tags = [i for _, subtags in tags for i in subtags]
        else:
            tags = self.canvas.gettags(self.canvas.find('withtag', 'current'))

        for tag in tags:
            
            m = re.fullmatch("piece_(\d+)", tag)
            if m:
                piece_no = int(m[1])
            if tag == 'rotate':
                drag_type = 'turn'

        had_selection = self.selection is not None

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

        # HACK: assume hittester invalidated
        # self.hittester = None
        if self.selection is not None or had_selection:
            self.scenegraph = None
        self.render()

    def canvas_drag(self, event):

        if not self.draggable:
            return
        
        self.draggable.drag(np.array((event.x, event.y)))
        
        if self.selection is not None:
            self.scenegraph = None
            
        self.render()

    def canvas_release(self, event):

        if not self.draggable:
            return
        
        self.draggable.commit()
        self.draggable = None

        # HACK: assume hittester invalidated
        # self.hittester = None
        if self.selection is not None:
            self.scenegraph = None
        self.render()

    def render(self):

        if self.use_scenegraph:
            return self.sg_render()
        
        r = PuzzleRenderer(self.canvas, self.camera, self.pieces)
        r.selection = self.selection
        r.frontiers = self.solver.frontiers
        if self.var_render_adjacency.get():
            r.adjacency = self.solver.adjacency
        if self.render_normals:
            r.normals = self.render_normals
        if self.render_vertexes:
            r.vertexes = self.render_vertexes
        if self.use_cairo:
            r.use_cairo = True
            
        self.displayed_image = r.render(True)

        self.render_full += 1
        if 1 == self.render_full:
            self.parent.after_idle(self.full_render)

    def full_render(self):

        if self.use_scenegraph:
            return self.sg_render()
        
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
        if self.use_cairo:
            r.use_cairo = True
            
        self.displayed_image = r.render(False)

        self.render_full = 0

    def sg_render(self):

        # ideally the scenegraph doesn't have to get rebuilt for a
        # simple change in camera position

        t_start = time.perf_counter()

        if self.scenegraph is None:
            self.scenegraph = self.build_scenegraph()
            self.hittester = None

        t_build = time.perf_counter()
        
        if self.hittester is None:
            self.hittester = self.build_hittester()
            
        t_hittest = time.perf_counter()

        if self.use_cairo:
            r = puzzler.renderer.cairo.CairoRenderer(self.canvas)
        else:
            self.canvas.delete('all')
            r = puzzler.renderer.canvas.CanvasRenderer(self.canvas)

        r.transform(self.camera.matrix)

        sgr = puzzler.scenegraph.SceneGraphRenderer(
            r, self.camera.viewport, scale=self.camera.zoom)
        self.scenegraph.root_node.accept(sgr)

        if False:
            for o in self.hittester.objects:
                points = sgr.bbox_corners(o.bbox)
                r.draw_polygon(points, outline='gray', width=1, dashes=(3,6))

        self.displayed_image = r.commit()

        t_render = time.perf_counter()
        
        if False and self.var_render_adjacency.get():
            with open('scenegraph_foo.json','w') as f:
                f.write(puzzler.scenegraph.to_json(self.scenegraph))

        if False:
            print("-------------------------------------------------------")
            print(f"sg={t_build-t_start:.3f} ht={t_hittest-t_build:.3f} render={t_render-t_hittest:.3f} viewport={self.camera.viewport} scale={self.camera.zoom}")
            for f in traceback.extract_stack():
                filename = f.filename
                prefix = 'C:\\home\\eldridge\\proj\\puzzler\\'
                if filename.startswith(prefix):
                    filename = '.\\' + filename[len(prefix):]
                print(f"{filename}:{f.lineno}  {f.name}")

    def build_scenegraph(self):

        f = PuzzleSGFactory(self.pieces)

        f.selection = self.selection
        f.frontiers = self.solver.frontiers
        if self.var_render_frontier.get():
            f.render_frontier_details = True
        if self.var_render_adjacency.get():
            f.adjacency = self.solver.adjacency
            # HACK, just show adjacency for this single piece
            if False and 'B1' in f.adjacency:
                f.adjacency = {'B1':f.adjacency['B1']}
        if self.render_normals:
            f.normals = self.render_normals
        if self.var_render_vertexes.get():
            if self.render_vertexes:
                f.vertexes = self.render_vertexes
            f.render_vertex_details = True
        if self.render_seams:
            f.seams = self.render_seams
        f.props['tabs.render'] = self.var_render_tabs.get() != 0

        return f.build()

    def build_hittester(self):

        return puzzler.scenegraph.BuildHitTester()(self.scenegraph.root_node)

    def do_render_tabs(self):

        self.scenegraph = None
        self.render()

    def do_render_vertexes(self):

        self.scenegraph = None
        self.render()

    def do_render_frontier(self):
        
        self.scenegraph = None
        self.render()

    def do_render_adjacency(self):
        
        self.scenegraph = None
        self.render()

    def do_tab_alignment(self):

        self.solver.solve_field()
        self.update_coords()

        self.scenegraph = None
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

    def load_solver(self, path):
        self.solver = puzzler.solver.load_json(path, self.solver.pieces)
        self.update_coords()
        
        self.scenegraph = None
        self.render()

    def do_solve(self):

        self.solver.solve_border()
        self.update_coords()

        self.scenegraph = None
        self.render()
        
    def mouse_wheel(self, event):
        f = pow(1.2, 1 if event.delta > 0 else -1)
        xy = (event.x, event.y)
        self.camera.fixed_point_zoom(f, xy)
        self.motion(event)
        self.render()

    def motion(self, event):
        xy = self.device_to_user((event.x, event.y))
        if self.hittester:
            tags = self.hittester(xy)
            tags = [i for _, i in tags]
        else:
            tags = CanvasHitTester(self.canvas)((event.x, event.y))
        tags = ','.join(str(i) for i in tags)
        self.var_label.set(f"{xy[0]:.0f},{xy[1]:.0f} {tags}")

    def canvas_map(self, event):
        # it's a hack (because I don't know why it works) but without
        # it I get a blank canvas until I do something to cause the
        # canvas to get repainted (e.g. a mouse drag)
        self.render()

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
        self.canvas.bind("<Map>", self.canvas_map)

        self.controls = ttk.Frame(self.frame)
        self.controls.grid(row=1, sticky=(W,E))

        b1 = ttk.Button(self.controls, text='Solve!', command=self.do_solve)
        b1.grid(column=0, row=0, sticky=W)

        b2 = ttk.Button(self.controls, text='Tab Alignment', command=self.do_tab_alignment)
        b2.grid(column=1, row=0, sticky=W)

        self.var_render_adjacency = IntVar(value=0)
        b3 = ttk.Checkbutton(self.controls, text="Adjacency",
                             command=self.do_render_adjacency,
                             variable=self.var_render_adjacency)
        b3.grid(column=2, row=0, sticky=W)

        self.var_render_frontier = IntVar(value=0)
        b4 = ttk.Checkbutton(self.controls, text="Frontier",
                             command=self.do_render_frontier,
                             variable=self.var_render_frontier)
        b4.grid(column=3, row=0, sticky=W)

        self.var_render_tabs = IntVar(value=0)
        b3 = ttk.Checkbutton(self.controls, text="Tabs",
                             command=self.do_render_tabs,
                             variable=self.var_render_tabs)
        b3.grid(column=4, row=0, sticky=W)

        self.var_render_vertexes = IntVar(value=0)
        b4 = ttk.Checkbutton(self.controls, text="Vertexes",
                             command=self.do_render_vertexes,
                             variable=self.var_render_vertexes)
        b4.grid(column=5, row=0, sticky=W)

        self.var_solve_continuous = IntVar(value=0)
        b5 = ttk.Checkbutton(self.controls, text="Continuous", variable=self.var_solve_continuous)
        b5.grid(column=6, row=0, sticky=W)

        self.var_label = StringVar(value="x,y")
        l1 = ttk.Label(self.controls, textvariable=self.var_label, width=80)
        l1.grid(column=7, row=0, sticky=E)

        cf2 = ttk.Frame(self.frame)
        cf2.grid(row=2, sticky=(W,E))

        b5 = ttk.Button(cf2, text='Reset Layout', command=self.reset_layout)
        b5.grid(column=0, row=0, sticky=W)

        b6 = ttk.Button(cf2, text='Show Tab Alignment', command=self.show_tab_alignment)
        b6.grid(column=1, row=0, sticky=W)

        self.var_show_tab_alignment = StringVar(value='')
        e1 = ttk.Entry(cf2, width=16, textvariable=self.var_show_tab_alignment)
        e1.grid(column=2, row=0)

        b7 = ttk.Button(cf2, text='Show Edge Alignment', command=self.show_edge_alignment)
        b7.grid(column=3, row=0)

        self.var_show_edge_alignment = StringVar(value='')
        e2 = ttk.Entry(cf2, width=16, textvariable=self.var_show_edge_alignment)
        e2.grid(column=4, row=0)

        b8 = ttk.Button(cf2, text='Show Raft Alignment', command=self.show_raft_alignment)
        b8.grid(column=5, row=0)

        self.var_show_raft_alignment = StringVar(value='')
        e3 = ttk.Entry(cf2, width=32, textvariable=self.var_show_raft_alignment)
        e3.grid(column=6, row=0)

        self.var_refine_raft_alignment = IntVar(value=0)
        b9 = ttk.Checkbutton(cf2, text='Refine Raft Alignment', variable=self.var_refine_raft_alignment)
        b9.grid(column=7, row=0)

        self.render_full = False

    def resize(self, e):
        viewport = (e.width, e.height)
        if self.camera.viewport != viewport:
            self.camera.viewport = viewport
            self.render()

    def reset_layout(self):
        
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
        for p in self.pieces:
            r, c = to_row_col(p.piece.label)
            rows.add(r)
            cols.add(c)

        rows = dict((r, i) for i, r in enumerate(sorted(rows)))
        cols = dict((c, i) for i, c in enumerate(sorted(cols)))

        for p in self.pieces:
            r, c = to_row_col(p.piece.label)
            x = cols[c] * 1000.
            y = rows[r] * -1000.
            p.coords.angle = 0.
            p.coords.dxdy = np.array((x, y))

        self.scenegraph = None
        self.render_normals = None
        self.render_vertexes = None
        self.render_seams = None
        self.render()

    def show_edge_alignment(self):
        s = self.var_show_edge_alignment.get().strip()
        m = re.fullmatch("([a-zA-Z]+\d+):(\d+),(\d+)=([a-zA-Z]+\d+):(\d+),(\d+)", s)
        if not m:
            print(f"bad input: \"{s}\", should be <dst_label>:<edge>,<tab>=<src_label>:<edge>,<tab>")
            return

        dst_label, dst_desc = m[1], (int(m[2]), int(m[3]))
        src_label, src_desc = m[4], (int(m[5]), int(m[6]))

        print(f"{dst_label=} {dst_desc=} {src_label=} {src_desc=}")

        pieces = dict([(i.piece.label, i) for i in self.pieces])
        dst = pieces[dst_label]
        src = pieces[src_label]

        edge_aligner = puzzler.align.EdgeAligner(dst.piece)
        edge_aligner.return_table = True

        mse, src_coords, sfp, dfp = edge_aligner.compute_alignment(dst_desc, src.piece, src_desc)

        print(f"edge_aligner: {mse=:.2f} {sfp=} {dfp=}")

        t = edge_aligner.table
        self.render_vertexes = dict()
        self.render_vertexes[src_label] = t['src_vertex']

        dst.coords = puzzler.align.AffineTransform(0., (0., 2000.))
        src.coords = puzzler.align.AffineTransform(src_coords.angle, src_coords.dxdy + dst.coords.dxdy)

        self.scenegraph = None
        self.render()
        
    def show_tab_alignment(self):
        s = self.var_show_tab_alignment.get().strip()
        m = re.fullmatch("([a-zA-Z]+\d+):(\d+)=([a-zA-Z]+\d+):(\d+)", s)
        if not m:
            print(f"bad input: \"{s}\", should be <dst_label>:<dst_tab_no>=<src_label>:<src_tab_no>")
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
        tab_aligner.sample_interval = 10
        tab_aligner.return_table = True

        mse, src_coords, sfp, dfp = tab_aligner.compute_alignment(dst_tab_no, src.piece, src_tab_no, refine=2)
        print(f"{mse=} {src_coords=} {sfp=} {dfp=}")

        if False:
            src_mid = tab_aligner.get_tab_midpoint(src.piece, src_tab_no)

            mse, src_coords, sfp, dfp = tab_aligner.refine_alignment(src.piece, src_coords, src_mid)
            print(f"{mse=} {src_coords=} {sfp=} {dfp=}")
            
            mse, src_coords, sfp, dfp = tab_aligner.refine_alignment(src.piece, src_coords, src_mid)
            print(f"{mse=} {src_coords=} {sfp=} {dfp=}")

        # mse, src_coords, sfp, dfp = tab_aligner.compute_alignment(dst_tab_no, src.piece, src_tab_no, refine=2)
        # print(f"refine=2: {mse=} {src_coords=} {sfp=} {dfp=}")

        t = tab_aligner.table
        
        self.render_vertexes = dict()
        self.render_vertexes[src_label] = t['src_vertex']
        self.render_vertexes[dst_label] = t['dst_vertex']
        
        self.render_normals = dict()
        self.render_normals[dst_label] = list(zip(t['dst_vertex'], t['dst_normal']))
        self.render_normals[src_label] = list(zip(t['src_vertex'], t['src_normal']))

        def format_value(x):
            if isinstance(x,str):
                return x
            if isinstance(x,int):
                return str(x)
            if isinstance(x,float):
                return f"{x:.3f}"
            if isinstance(x,np.ndarray):
                with np.printoptions(precision=3):
                    return str(x)
            return str(x)

        keys = list(t.keys())
        n_rows = len(t[keys[0]])
        print('vertex_no,',','.join(keys))
        for i in range(n_rows):
            print(i,',',','.join(format_value(t[k][i]) for k in keys))

        dst.coords = puzzler.align.AffineTransform(0., (0., 2000.))
        src.coords = puzzler.align.AffineTransform(src_coords.angle, src_coords.dxdy + dst.coords.dxdy)

        self.scenegraph = None
        self.render()

    def show_raft_alignment(self):

        s = self.var_show_raft_alignment.get()

        pieces = dict([(i.piece.label, i.piece) for i in self.pieces])

        def print_coords(raft):
            for k, v in raft.coords.items():
                x, y = v.xy
                print(f"{k}: angle={v.angle:.3f} xy=({x:.3f},{y:.3f})")
        
        r = puzzler.raft.Raftinator(pieces)
        raft = r.make_raft_from_string(s)
        
        seams = r.get_seams_for_raft(raft)
        mse = r.get_cumulative_error_for_seams(seams)
        mse2 = r.get_total_error_for_raft_and_seams(raft, seams)
        print_coords(raft)
        print(f"MSE={mse:.3f} MSE2={mse2:.3f}")

        if self.var_refine_raft_alignment.get():
            for pass_no in range(5):
                print(f"Taking *REFINED* alignment! {pass_no=}")
                raft = r.refine_alignment_within_raft(raft)
                seams = r.get_seams_for_raft(raft)
                mse = r.get_cumulative_error_for_seams(seams)
                mse2 = r.get_total_error_for_raft_and_seams(raft, seams)
                print_coords(raft)
                print(f"MSE={mse:.3f} MSE2={mse2:.3f}")
            
        pieces = dict([(i.piece.label, i) for i in self.pieces])
        
        dst_coord = puzzler.align.AffineTransform(0., (0., 2000.))
        for label, coord in raft.coords.items():
            curr_m = dst_coord.get_transform().matrix
            prev_m = coord.matrix
            pieces[label].coords = puzzler.align.AffineTransform.invert_matrix(curr_m @ prev_m)

        self.render_vertexes = dict()
        self.render_normals = dict()
        self.render_seams = seams

        self.scenegraph = None
        self.render()
            
    def show_legacy_raft_alignment(self):
        
        s = self.var_show_raft_alignment.get().strip()
        v = s.split(',')
        if len(v) != 4:
            print(f"bad input: \"{s}\", should be <dst_raft>,<trace_no>,<src_raft>,<trace_no>")
            return
        
        dst_raft_name = v[0]
        dst_trace_no = int(v[1])
        src_raft_name = v[2]
        src_trace_no = int(v[3])

        print(f"{dst_raft_name=} {dst_trace_no=} {src_raft_name=} {src_trace_no=}")

        def frob_raft_name(s):
            m = re.fullmatch("([A-Z]+\d+):(\d+)=([A-Z]+\d+):(\d+)", s)
            if not m:
                print(f"bad raft name: \"{s}\", should be <dst_piece>:<tab_no>=<src_piece>:<tab_no>")
            return ((m[1], int(m[2])), (m[3], int(m[4])))

        pieces = dict([(i.piece.label, i.piece) for i in self.pieces])

        bb = puzzler.align.BosomBuddies(pieces, [])

        dst_raft = bb.make_raft(*frob_raft_name(dst_raft_name))
        src_raft = bb.make_raft(*frob_raft_name(src_raft_name))

        print(f"{dst_raft=}")
        print(f"{src_raft=}")

        align_rafts = puzzler.align.RaftAligner(pieces, dst_raft, dst_trace_no)
        src_coords = align_rafts.compute_alignment_for_trace(src_raft, src_trace_no)
        mse = align_rafts.measure_fit_for_trace(src_raft, src_trace_no, src_coords)

        print(f"{mse=:.1f}")

        def make_raft2(raft_name):
            pieces = dict([(i.piece.label, i.piece) for i in self.pieces])
            
            f = puzzler.raft.RaftFactory(pieces)
            (dst_piece, dst_tab_no), (src_piece, src_tab_no) = frob_raft_name(raft_name)
            dst_raft = f.make_raft_for_piece(dst_piece)
            src_raft = f.make_raft_for_piece(src_piece)

            print(f"{dst_raft=}")
            print(f"{src_raft=}")

            dst_feature = puzzler.raft.Feature(dst_piece, 'tab', dst_tab_no)
            src_feature = puzzler.raft.Feature(src_piece, 'tab', src_tab_no)

            a = puzzler.raft.RaftAligner(pieces, dst_raft)
            src_coord = a.rough_align_single_tab(src_raft, (dst_feature, src_feature))
            print(f"{src_coord=}")

            alignment = puzzler.raft.RaftAlignment(dst_raft, src_raft, src_coord)

            ret = f.merge_rafts(alignment)
            print(f"{ret=}")
            return ret

        def trace_to_features(raft, trace_no):
            trace = raft.traces[trace_no][1]
            return [puzzler.raft.Feature(p, 'tab', i) for p, i in trace]

        def merge_rafts(dst_raft, src_raft, feature_pairs):
            
            pieces = dict([(i.piece.label, i.piece) for i in self.pieces])
            a = puzzler.raft.RaftAligner(pieces, dst_raft)
            
            src_coord = a.rough_align_multiple_tabs(src_raft, feature_pairs)
            print(f"{src_coord=}")
            
            f = puzzler.raft.RaftFactory(pieces)
            alignment = puzzler.raft.RaftAlignment(dst_raft, src_raft, src_coord)

            ret = f.merge_rafts(alignment)
            print(f"{ret=}")
            return ret

        def get_seams(raft):

            pieces = dict([(i.piece.label, i.piece) for i in self.pieces])
            s = puzzler.raft.RaftSeamstress(pieces)
            seams = s.seams_within_raft(raft)
            for seam in seams:
                n_points = len(seam.src.indices)
                mse = seam.error / n_points
                print(f"seam: dst={seam.dst.piece} src={seam.src.piece} {n_points=} {mse=:.1f}")
                # print(f"  src_indices={seam.src.indices}")
                # print(f"  dst_indices={seam.dst.indices}")

            print("TRIMMED:")
            seams = s.trim_seams(seams)
            for seam in seams:
                n_points = len(seam.src.indices)
                mse = seam.error / n_points
                print(f"seam: dst={seam.dst.piece} src={seam.src.piece} {n_points=} {mse=:.1f}")
                # print(f"  src_indices={seam.src.indices}")
                # print(f"  dst_indices={seam.dst.indices}")
            return seams

        def refine_alignment_within_raft(raft, seams):

            pieces = dict([(i.piece.label, i.piece) for i in self.pieces])
            a = puzzler.raft.RaftAligner(pieces, dst_raft)

            return a.refine_alignment_within_raft(raft, seams)

        def cumulative_seam_error(seams):
            n_points = 0
            error = 0.
            for s in seams:
                n_points += len(s.src.indices)
                error += s.error
            return error / n_points
            
        print("-------------------")
        dst_raft2 = make_raft2(dst_raft_name)
        print("-------------------")
        src_raft2 = make_raft2(src_raft_name)
        print("-------------------")
        
        feature_pairs = list(zip(trace_to_features(dst_raft, dst_trace_no),
                                 reversed(trace_to_features(src_raft, src_trace_no))))
        print(f"{feature_pairs=}")

        uber_raft = merge_rafts(dst_raft2, src_raft2, feature_pairs)
        
        print("-------------------")

        uber_seams = get_seams(uber_raft)
        print(f"MSE={cumulative_seam_error(uber_seams):.1f}")

        print("-------------------")

        uber_raft2 = refine_alignment_within_raft(uber_raft, uber_seams)
        
        uber_seams2 = get_seams(uber_raft2)
        print(f"MSE={cumulative_seam_error(uber_seams2):.1f}")
        
        if self.var_refine_raft_alignment.get():
            print("Taking *REFINED* alignment!")
            uber_raft = uber_raft2
            uber_seams = uber_seams2
        else:
            print("Taking *ORIGINAL* alignment!")

        print("-------------------")

        dst_coords = puzzler.align.AffineTransform(0., (0., 2000.))

        pieces = dict([(i.piece.label, i) for i in self.pieces])

        if uber_raft is not None:
            
            for label, coord in uber_raft.coords.items():
                curr_m = dst_coords.get_transform().matrix
                prev_m = coord.matrix
                pieces[label].coords = puzzler.align.AffineTransform.invert_matrix(curr_m @ prev_m)

            self.render_vertexes = dict()
            self.render_normals = dict()
            self.render_seams = uber_seams
                
        else:

            for label, coords in dst_raft.coords.items():
                curr_m = dst_coords.get_transform().matrix
                prev_m = coords.get_transform().matrix
                pieces[label].coords = puzzler.align.AffineTransform.invert_matrix(curr_m @ prev_m)
            
            for label, coords in src_raft.coords.items():
                curr_m = puzzler.align.AffineTransform(
                    src_coords.angle, src_coords.dxdy + dst_coords.dxdy).get_transform().matrix
                prev_m = coords.get_transform().matrix
                pieces[label].coords = puzzler.align.AffineTransform.invert_matrix(curr_m @ prev_m)

        self.scenegraph = None
        self.render()
        
def load_buddies(path):
    
    buddies = dict()
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dst = (row['dst_label'], int(row['dst_tab_no']))
            src = (row['src_label'], int(row['src_tab_no']))
            rank = int(row['rank'])
            if rank == 1:
                buddies[dst] = src

    return buddies

def test_buddies(pieces, buddies_path):

    # the buddies file is a csv of piece pairs and the rank for the
    # quality of fit for them on each possible tab pairing, generated
    # by the "tabs" command
    buddies = load_buddies(buddies_path)
    buddies = [(a, b) for a, b in buddies.items() if buddies.get(b) == a and a > b]
    
    bb = puzzler.align.BosomBuddies(pieces, buddies)

    def format_raft(raft):
        return '_'.join(f"{a}:{b}={c}:{d}" for (a, b), (c, d) in raft.joints)

    def are_traces_compatible(a, b):
        sigA = a[0]
        sigB = b[0]
        return sigA[0] == (not sigB[2]) and sigA[2] == (not sigB[0])

    f = open('ranked_quads.csv', 'w', newline='')
    writer = csv.DictWriter(f, fieldnames='dst_raft dst_trace_no src_raft src_trace_no mse rank'.split())
    writer.writeheader()

    for dst_raft in bb.rafts:
        for dst_trace_no, dst_trace in enumerate(dst_raft.traces):
            align_rafts = puzzler.align.RaftAligner(pieces, dst_raft, dst_trace_no)

            rows = []
            for src_raft in bb.rafts:
                if src_raft == dst_raft:
                    continue
                for src_trace_no, src_trace in enumerate(src_raft.traces):
                    # shouldn't this be dst_trace, not dst_raft.traces[0]???
                    if not are_traces_compatible(dst_trace, src_trace):
                        continue
                    src_coords = align_rafts.compute_alignment_for_trace(src_raft, src_trace_no)
                    mse = align_rafts.measure_fit_for_trace(src_raft, src_trace_no, src_coords)

                    rows.append({'dst_raft': format_raft(dst_raft),
                                 'dst_trace_no': dst_trace_no,
                                 'src_raft': format_raft(src_raft),
                                 'src_trace_no': src_trace_no,
                                 'mse': mse,
                                 'rank': 0})

            rows.sort(key=operator.itemgetter('mse'))                    
            for i, row in enumerate(rows, start=1):
                row['rank'] = i

            writer.writerows(rows)

class RaftHelper:

    def __init__(self, pieces):
        self.pieces = pieces
        self.raft_factory = puzzler.raft.RaftFactory(pieces)
        self.raft_aligner = puzzler.raft.RaftAligner(pieces, None)
        self.raft_seamstress = puzzler.raft.RaftSeamstress(pieces)

    def make_raft(self, dst_piece, dst_tab_no, src_piece, src_tab_no):
    
        dst_raft = self.raft_factory.make_raft_for_piece(dst_piece)
        src_raft = self.raft_factory.make_raft_for_piece(src_piece)

        dst_feature = puzzler.raft.Feature(dst_piece, 'tab', dst_tab_no)
        src_feature = puzzler.raft.Feature(src_piece, 'tab', src_tab_no)

        self.raft_aligner.dst_raft = dst_raft
        src_coord = self.raft_aligner.rough_align_single_tab(src_raft, (dst_feature, src_feature))

        alignment = puzzler.raft.RaftAlignment(dst_raft, src_raft, src_coord)

        return self.raft_factory.merge_rafts(alignment)

    def trace_to_features(self, legacy_raft, trace_no):
        trace = legacy_raft.traces[trace_no][1]
        return [puzzler.raft.Feature(p, 'tab', i) for p, i in trace]

    def merge_rafts(self, dst_raft, src_raft, feature_pairs):
    
        self.raft_aligner.dst_raft = dst_raft
        src_coord = self.raft_aligner.rough_align_multiple_tabs(src_raft, feature_pairs)
    
        alignment = puzzler.raft.RaftAlignment(dst_raft, src_raft, src_coord)

        return self.raft_factory.merge_rafts(alignment)

    def get_seams(self, raft):

        return self.raft_seamstress.trim_seams(self.raft_seamstress.seams_within_raft(raft))

    def refine_alignment_within_raft(self, raft, seams):

        self.raft_aligner.dst_raft = raft
        return self.raft_aligner.refine_alignment_within_raft(raft, seams)

    def cumulative_seam_error(self, seams):
        n_points = 0
        error = 0.
        for s in seams:
            n_points += len(s.src.indices)
            error += s.error
        return error / n_points
            
def test_buddies_v2(pieces, buddies_path):

    # the buddies file is a csv of piece pairs and the rank for the
    # quality of fit for them on each possible tab pairing, generated
    # by the "tabs" command
    buddies = load_buddies(buddies_path)
    buddies = [(a, b) for a, b in buddies.items() if buddies.get(b) == a and a > b]
    
    bb = puzzler.align.BosomBuddies(pieces, buddies)
    
    def format_raft(raft):
        return '_'.join(f"{a}:{b}={c}:{d}" for (a, b), (c, d) in raft.joints)

    def are_traces_compatible(a, b):
        sigA = a[0]
        sigB = b[0]
        return sigA[0] == (not sigB[2]) and sigA[2] == (not sigB[0])

    raft_helper = RaftHelper(pieces)

    new_rafts = [raft_helper.make_raft(*dst, *src) for dst, src in buddies]

    f = open('ranked_quads.csv', 'w', newline='')
    writer = csv.DictWriter(f, fieldnames='dst_raft dst_trace_no src_raft src_trace_no mse rank'.split())
    writer.writeheader()

    n_rows = 0
    for i, dst_raft in enumerate(tqdm(bb.rafts)):

        for dst_trace_no, dst_trace in enumerate(dst_raft.traces):

            rows = []
            
            for j, src_raft in enumerate(bb.rafts):
                
                if i == j:
                    continue

                for src_trace_no, src_trace in enumerate(src_raft.traces):

                    if not are_traces_compatible(dst_trace, src_trace):
                        continue

                    dst_raft2 = new_rafts[i]
                    src_raft2 = new_rafts[j]

                    feature_pairs = list(zip(raft_helper.trace_to_features(dst_raft, dst_trace_no),
                                             reversed(raft_helper.trace_to_features(src_raft, src_trace_no))))

                    uber_raft = raft_helper.merge_rafts(dst_raft2, src_raft2, feature_pairs)

                    uber_seams = raft_helper.get_seams(uber_raft)

                    uber_raft2 = raft_helper.refine_alignment_within_raft(uber_raft, uber_seams)
        
                    uber_seams2 = raft_helper.get_seams(uber_raft2)

                    mse = raft_helper.cumulative_seam_error(uber_seams2)
                    
                    rows.append({'dst_raft': format_raft(dst_raft),
                                 'dst_trace_no': dst_trace_no,
                                 'src_raft': format_raft(src_raft),
                                 'src_trace_no': src_trace_no,
                                 'mse': decimal.Decimal(f"{mse:.6f}"),
                                 'rank': 0})

            rows.sort(key=operator.itemgetter('mse'))
            for k, row in enumerate(rows, start=1):
                row['rank'] = k

            writer.writerows(rows)
            n_rows += len(rows)

def quadmaster(pieces, quad):

    raftinator = puzzler.raft.Raftinator(pieces)

    Feature = puzzler.raft.Feature

    def format_feature(f):
        return f"{f.piece}:{f.index}" if f.kind == 'tab' else f"{f.piece}/{f.index}"

    def format_feature_pair(p):
        return format_feature(p[0]) + '=' + format_feature(p[1])

    def format_feature_pairs(pairs):
        return ','.join(format_feature_pair(i) for i in pairs)

    def format_frontier(frontier):
        return ','.join(format_feature(i) for i in frontier)

    def make_rafts(dst_label, src_label):

        dst = pieces[dst_label]
        src = pieces[src_label]
        
        dst_raft = raftinator.factory.make_raft_for_piece(dst_label)
        src_raft = raftinator.factory.make_raft_for_piece(src_label)

        retval = []
        for dst_tab_no, dst_tab in enumerate(dst.tabs):
            for src_tab_no, src_tab in enumerate(src.tabs):
                if dst_tab.indent == src_tab.indent:
                    continue

                feature_pairs = [(Feature(dst_label,'tab',dst_tab_no), Feature(src_label,'tab',src_tab_no))]
                desc = format_feature_pairs(feature_pairs)

                new_raft = raftinator.align_and_merge_rafts_with_feature_pairs(dst_raft, src_raft, feature_pairs)

                retval.append((desc, new_raft))

        if False:
            print(f"make_rafts: dst={dst_label} src={src_label} n_rafts={len(retval)}")
            for i, (desc, raft) in enumerate(retval):
                print(f"  raft[{i}]: {desc=} frontier[0]={format_frontier(raft.frontiers[0])}")

        return retval

    def fnord(frontier):
        
        retval = []
        for i in range(len(frontier)):
            l = frontier[i-1]
            r = frontier[i]
            if l.piece != r.piece and l.kind == 'tab' and r.kind == 'tab':
                retval.append((l, r))

        return retval

    def are_compatible_tabs(a, b):

        if a.kind != 'tab' or b.kind != 'tab':
            return False

        tab_a = pieces[a.piece].tabs[a.index]
        tab_b = pieces[b.piece].tabs[b.index]
        return tab_a.indent != tab_b.indent

    def are_good_feature_pairs_for_quad(fps, quad):

        ul, ur, ll, lr = quad
        
        return (len(fps) == 2 and
                fps[0][0].piece == ur and
                fps[0][1].piece == lr and
                fps[1][0].piece == ul and
                fps[1][1].piece == ll and
                are_compatible_tabs(*fps[0]) and
                are_compatible_tabs(*fps[1]))
        
    def join_rafts(dst_raft, src_raft, quad):

        dst_features = fnord(dst_raft.frontiers[0])
        src_features = fnord(src_raft.frontiers[0][::-1])

        retval = []
        for a, b in dst_features:
            for c, d in src_features:

                feature_pairs = [(a, c), (b, d)]
                if are_good_feature_pairs_for_quad(feature_pairs, quad):
                    new_raft = raftinator.align_and_merge_rafts_with_feature_pairs(dst_raft, src_raft, feature_pairs)
                    new_raft2 = raftinator.refine_alignment_within_raft(new_raft)
                    seams2 = raftinator.get_seams_for_raft(new_raft2)
                    mse = raftinator.get_total_error_for_raft_and_seams(new_raft2, seams2)
                    desc = format_feature_pairs(feature_pairs)
                    retval.append((desc, mse))

        return retval

    ul, ur, ll, lr = quad
    
    upper_rafts = make_rafts(ul, ur)
    lower_rafts = make_rafts(ll, lr)

    retval = []
    for upper in upper_rafts:
        for lower in lower_rafts:
            # print(f"upper={upper[0]} lower={lower[0]}")
            for desc, mse in join_rafts(upper[1], lower[1], quad):
                s = ','.join([upper[0], lower[0], desc])
                retval.append({'ul_piece':ul, 'ur_piece':ur, 'll_piece':ll, 'lr_piece':lr,
                               'upper_raft':upper[0], 'lower_raft':lower[0], 'quad_raft':s, 'mse':mse, 'rank':None})

    retval.sort(key=operator.itemgetter('mse'))
    for i, row in enumerate(retval, start=1):
        row['rank'] = i

    return retval
        
def quadrophenia(pieces):

    assert len(pieces) == 1026

    rows = [chr(ord('A')+i) for i in range(26)] + ['AA']
    cols = [str(i) for i in range(1,39)]
    # print(f"{rows=} {cols=} n_pieces={len(rows)*len(cols)}")

    quads = []
    for row_no in range(len(rows)-1):
        for col_no in range(len(cols)-1):
            quads.append((row_no, col_no))

    f = open('quadrophenia.csv', 'w', newline='')
    writer = csv.DictWriter(f, fieldnames='col_no row_no ul_piece ur_piece ll_piece lr_piece upper_raft lower_raft quad_raft mse rank'.split())
    writer.writeheader()

    for row_no, col_no in tqdm(quads):
        
        row0, row1 = rows[row_no], rows[row_no+1]
        col0, col1 = cols[col_no], cols[col_no+1]
        quad = (row0+col0, row0+col1, row1+col0, row1+col1)
        therows = quadmaster(pieces, quad)
        for i in therows:
            i['row_no'] = row_no
            i['col_no'] = col_no
        writer.writerows(therows)
    
def align_ui(args):

    puzzle = puzzler.file.load(args.puzzle)

    by_label = dict()
    for p in puzzle.pieces:
        by_label[p.label] = p

    if args.quadrophenia:
        quadrophenia(by_label)
        return

    if 'I1' in by_label:
        p = by_label['I1']
        if len(p.edges) == 2:
            p.edges = p.edges[::-1]

    if args.buddies:
        test_buddies_v2(by_label, args.buddies)
        return

    labels = set(args.labels)

    if not labels:
        labels |= set(by_label.keys())

    # approx (simplified polygon) only needed for immediate mode rendering
    epsilon = 10 if args.mode == 'immediate' else None
    pieces = [Piece(by_label[l], epsilon) for l in sorted(labels)]

    root = Tk()
    ui = AlignTk(root, pieces, renderer=args.renderer, mode=args.mode)
    root.bind('<Key-Escape>', lambda e: root.destroy())
    root.title("Puzzler: align")

    ui.reset_layout()
    if args.input:
        ui.load_solver(args.input)

    root.mainloop()

def add_parser(commands):

    parser_align = commands.add_parser("align", help="UI to experiment with aligning pieces")
    parser_align.add_argument("labels", nargs='*')
    parser_align.add_argument("-q", "--quadrophenia", help="quadrophenia", action='store_true')
    parser_align.add_argument("-i", "--input", help="initialize solver")
    parser_align.add_argument("-b", "--buddies", help="buddy file")
    parser_align.add_argument("-r", "--renderer", choices=['tk', 'cairo'], default='cairo',
                               help="renderer (default: %(default)s)")
    parser_align.add_argument("-m", "--mode", choices=['scenegraph', 'immediate'], default='scenegraph',
                               help="mode (default: %(default)s)")
    parser_align.set_defaults(func=align_ui)
