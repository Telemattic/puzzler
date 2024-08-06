import collections
import csv
import decimal
import itertools
import math
import numpy as np
import operator
import palettable.tableau
import re
import time
import traceback
import puzzler.feature
import puzzler
import puzzler.raft
import puzzler.solver
from tqdm import tqdm

from tkinter import *
from tkinter import ttk

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
        self.init_piece_xy = piece.coords.xy.copy()
        super().__init__(camera_matrix)

    def start(self, xy):
        self.origin = self.transform(xy)

    def drag(self, xy):
        self.piece.coords.xy = self.init_piece_xy + (self.transform(xy) - self.origin)

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
        dx, dy = xy - self.piece.coords.xy
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

Coord = puzzler.align.Coord

class Piece:

    def __init__(self, piece):

        self.piece = piece
        self.coords = Coord()

class PuzzleSGFactory:

    # HACK: initialized as singleton on first construction of PuzzleSGFactory
    piece_factory = None

    def __init__(self, pieces):
        self.pieces = pieces
        self.selection = None
        self.frontiers = []
        self.render_frontier_details = False
        self.render_vertex_details = False
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
                
            sgb.add_translate(p.coords.xy)

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
            if self.render_vertex_details:
                draw_vertexes(stitches.points, color)
            if normals_flag:
                normals = list(zip(stitches.points, stitches.normals))
                draw_normals(normals, color)

        def draw_index_range(stitches, color):

            seamstress = puzzler.raft.RaftSeamstress({p.piece.label: p.piece})
            a, b = seamstress.get_index_range_for_stitches(stitches)

            with puzzler.sgbuilder.insert_sequence(sgb):
                sgb.add_translate(p.coords.xy)
                sgb.add_rotate(p.coords.angle)
            
                va = p.piece.points[a]
                vb = p.piece.points[b]
                
                sgb.add_points([va], radius=6, fill='', outline=color, width=1)
                sgb.add_text(va, "a", font=('Courier New', 12))
                
                sgb.add_points([vb], radius=6, fill='', outline=color, width=1)
                sgb.add_text(vb, "b", font=('Courier New', 12))

        if self.seams:
            for seam in self.seams:
                if seam.src.piece == p.piece.label:
                    draw_stitches(seam.src, color, False)
                if seam.dst.piece == p.piece.label:
                    draw_stitches(seam.dst, color, True)
                    draw_index_range(seam.dst, color)
        
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

            sgb.add_translate(p.coords.xy)
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
                sgb.add_translate(p.coords.xy)
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
                sgb.add_translate(p.coords.xy)
                sgb.add_rotate(p.coords.angle)
                sgb.add_points([p.piece.points[b]], radius=10, outline='purple')
                
        for i, (l, (a, b)) in enumerate(frontier):
            p = piece_dict[l]
            with puzzler.sgbuilder.insert_sequence(sgb):
                sgb.add_translate(p.coords.xy)
                sgb.add_rotate(p.coords.angle)
                v = p.piece.points[a]
                sgb.add_points([v], radius=8, fill='pink')
                label = str(i)
                sgb.add_text(v, label, font=('Courier New', 12))

class CanvasHitTester:

    def __init__(self, canvas):
        self.canvas = canvas

    def __call__(self, xy):
        x, y = xy
        oids = self.canvas.find('overlapping', x-1, y-1, x+1, y+1)
        return [self.canvas.gettags(i) for i in oids]

class AlignTk:

    def __init__(self, parent, pieces, *, expected=None, puzzle_path=None):
        
        self.pieces = pieces

        pieces_dict = {i.piece.label: i.piece for i in pieces}
        self.solver = puzzler.solver.PuzzleSolver(pieces_dict, expected=expected, puzzle_path=puzzle_path)

        self.draggable = None
        self.selection = None
        self.render_normals = None
        self.render_vertexes = None
        self.render_seams = None
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
            
            m = re.fullmatch(r"piece_(\d+)", tag)
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

        r = puzzler.renderer.cairo.CairoRenderer(self.canvas)

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

    def do_tab_alignment(self):

        self.solver.solve_field()
        if self.solver.seams:
            self.render_seams = self.solver.seams
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

        if not self.var_refine_raft_alignment.get():
            for p in self.pieces:
                if c := g.coords.get(p.piece.label):
                    p.coords = c
            return

        pieces = dict((i.piece.label, i.piece) for i in self.pieces)
        raftinator = puzzler.raft.Raftinator(pieces)

        coords = dict()
        for p, c in self.solver.geometry.coords.items():
            coords[p] = puzzler.raft.Coord(c.angle, c.xy)

        raft = puzzler.raft.Raft(coords)
        mse = raftinator.get_total_error_for_raft_and_seams(raft)
            
        raft2 = raftinator.refine_alignment_within_raft(raft)
        mse2 = raftinator.get_total_error_for_raft_and_seams(raft2)

        rfc = puzzler.raft.RaftFeaturesComputer(pieces)
        rfc.compute_features(raft.coords)
        
        print(f"raft: {mse=:.3f} {mse2=:.3f}")

        for p in self.pieces:
            if c := raft2.coords.get(p.piece.label):
                g.coords[p.piece.label] = p.coords = Coord(c.angle, c.xy)

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
            m = re.fullmatch(r"([a-zA-Z]+)(\d+)", label)
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
            # make space around the origin
            y = (rows[r] + 2) * -1000.
            p.coords.angle = 0.
            p.coords.xy = np.array((x, y))

        self.scenegraph = None
        self.render_normals = None
        self.render_vertexes = None
        self.render_seams = None
        self.render()

    def show_edge_alignment(self):
        s = self.var_show_edge_alignment.get().strip()
        m = re.fullmatch(r"([a-zA-Z]+\d+):(\d+),(\d+)=([a-zA-Z]+\d+):(\d+),(\d+)", s)
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

        dst.coords = Coord(0., (0., 0.))
        src.coords = Coord(src_coords.angle, src_coords.xy)

        self.scenegraph = None
        self.render()
        
    def show_tab_alignment(self):
        s = self.var_show_tab_alignment.get().strip()
        m = re.fullmatch(r"([a-zA-Z]+\d+):(\d+)=([a-zA-Z]+\d+):(\d+)", s)
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

        dst_corner_normals = tab_aligner.get_outside_normals(dst.piece, dfp[0], dfp[1])
        src_corner_normals = src_coords.xform.apply_n2(tab_aligner.get_outside_normals(src.piece, sfp[0], sfp[1]))

        corner_dp_0 = np.dot(dst_corner_normals[1], src_corner_normals[0])
        corner_dp_1 = np.dot(dst_corner_normals[0], src_corner_normals[1])

        with np.printoptions(precision=3):
            print(f"  {dst_corner_normals=}")
            print(f"  {src_corner_normals=}")
            print(f"  {corner_dp_0=} {corner_dp_1=}")
        
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

        dst.coords = Coord(0., (0., 0.))
        src.coords = Coord(src_coords.angle, src_coords.xy)

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

        # HACK: 2000.
        dst_coord = Coord(0., (0., 0.))
        for label, coord in raft.coords.items():
            curr_m = dst_coord.xform.matrix
            prev_m = coord.matrix
            pieces[label].coords = Coord.from_matrix(curr_m @ prev_m)

        pieces = dict([(i.piece.label, i.piece) for i in self.pieces])

        rfc = puzzler.raft.RaftFeaturesComputer(pieces)
        rfc.compute_features(raft.coords)

        self.render_vertexes = dict()
        self.render_normals = dict()
        self.render_seams = seams

        self.scenegraph = None
        self.render()
            
def read_expected_tab_matches(path):

    Feature = puzzler.raft.Feature

    expected = dict()
    
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dst = Feature(row['dst_piece'], 'tab', int(row['dst_tab_no']))
            src = Feature(row['src_piece'], 'tab', int(row['src_tab_no']))
            expected[dst] = src

    return expected

def cross_check_expected_and_pieces(expected, pieces):

    piece_tabs = set()
    for p in pieces.values():
        for i in range(len(p.tabs)):
            piece_tabs.add(puzzler.raft.Feature(p.label, 'tab', i))

    dst_tabs = set(expected.keys())
    src_tabs = set(expected.values())

    assert dst_tabs == src_tabs

    if piece_tabs != dst_tabs or dst_tabs != src_tabs:
        print(f"WARNING: {len(piece_tabs)=} {len(dst_tabs)=} {len(src_tabs)=}")
        a, b = piece_tabs, dst_tabs
        c = a & b
        print(f"    no. common tabs: {len(c)}")
        if a != c:
            s = ', '.join(str(i) for i in (a-c))
            print(f"    tabs not referenced in expected: {s}")
        if b != c:
            s = ', '.join(str(i) for i in (b-c))
            print(f"    unknown tabs in expected: {s}")
    
def align_ui(args):

    if args.num_workers:
        ps = puzzler.psolve.ParallelSolver(args.puzzle, args.num_workers)
        while ps.solve():
            pass
        n_pieces = len(ps.pieces)
        n_placed = len(ps.raft.coords) if ps.raft else 0
        print(f"all done! {n_pieces=} {n_placed=}")
        return

    puzzle = puzzler.file.load(args.puzzle)

    by_label = dict()
    for p in puzzle.pieces:
        by_label[p.label] = p

    # HACK: 100.json
    if 'I1' in by_label:
        p = by_label['I1']
        if len(p.edges) == 2:
            p.edges = puzzler.commands.ellipse.clean_edges(p.label, p.edges)

    labels = set(args.labels)

    if not labels:
        labels |= set(by_label.keys())

    expected = None
    if args.expected:
        expected = read_expected_tab_matches(args.expected)
        cross_check_expected_and_pieces(expected, by_label)

    pieces = [Piece(by_label[l]) for l in sorted(labels)]

    root = Tk()
    ui = AlignTk(root, pieces, expected=expected, puzzle_path=args.puzzle)
    root.bind('<Key-Escape>', lambda e: root.destroy())
    root.title("Puzzler: align")

    ui.reset_layout()
    if args.input:
        ui.load_solver(args.input)

    root.mainloop()

def add_parser(commands):

    parser_align = commands.add_parser("align", help="UI to experiment with aligning pieces")
    parser_align.add_argument("labels", nargs='*')
    parser_align.add_argument("-i", "--input", help="initialize solver")
    parser_align.add_argument("-e", "--expected", help="expected tab matches csv file")
    parser_align.add_argument("-n", "--num-workers", help="number of workers for parallel solve", default=0, type=int)
    parser_align.set_defaults(func=align_ui)
