import cv2 as cv
import math
import numpy as np
import puzzler
import puzzler.scenegraph
import puzzler.sgbuilder
import puzzler.renderer.canvas
import puzzler.renderer.cairo
import puzzler.renderer.opengl
import re

from tkinter import *
from tkinter import ttk

# Camera = puzzler.commands.align.Camera
# MoveCamera = puzzler.commands.align.MoveCamera

def simplify_polygon(points, epsilon):
    approx  = cv.approxPolyDP(points, epsilon, True)
    poly = np.squeeze(approx)
    return np.concatenate((poly, poly[:1,:]))

class Browser:

    class Outline:

        def __init__(self, piece, epsilon=5):

            assert piece.points is not None
            approx  = cv.approxPolyDP(piece.points, epsilon, True)
            self.poly = np.squeeze(approx)
            self.poly = np.concatenate((self.poly, self.poly[:1,:]))

            ll = np.min(self.poly, 0)
            ur = np.max(self.poly, 0)
            self.bbox   = tuple(ll.tolist() + ur.tolist())
            self.piece  = piece

    def make_scenegraph(outlines, num_rows, bbox_w, bbox_h):

        builder = puzzler.sgbuilder.SceneGraphBuilder()

        pieces = dict((o.piece.label, o.piece) for o in outlines)
        factory = puzzler.sgbuilder.PieceSceneGraphFactory(pieces)

        for o in outlines:
            x = o.col
            y = num_rows - 1 - o.row
            tx = (x + .5) * bbox_w
            ty = (y + .5) * bbox_h
            with puzzler.sgbuilder.insert_sequence(builder):
                builder.add_translate((tx, ty))
                # want the corners of the outline bbox centered within the tile
                bbox_center = np.array((o.bbox[0]+o.bbox[2], o.bbox[1]+o.bbox[3])) / 2
                builder.add_translate(-bbox_center)
                builder.add_node(factory(o.piece.label))
                
        sg = builder.commit(None, None)
        
        lodf = puzzler.sgbuilder.LevelOfDetailFactory()
        sg.root_node = lodf.visit_node(sg.root_node)
        
        return sg

    def __init__(self, puzzle, renderer, use_scenegraph, screensize=None):

        self.renderer = renderer
        
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
            return (to_row(m[1]), to_col(m[2])) if m else (label, None)

        self.outlines = []
        for p in puzzle.pieces:
            r, c = to_row_col(p.label)
            o = Browser.Outline(p)
            o.row = r - 1
            o.col = c - 1
            self.outlines.append(o)

        bbox_w = max(o.bbox[2] - o.bbox[0] for o in self.outlines)
        bbox_h = max(o.bbox[3] - o.bbox[1] for o in self.outlines)

        if screensize is not None:
            max_w, max_h = screensize
            max_w -= 100
            max_h -= 200
        else:
            max_w = 2500 # 1400
            max_h = 1500 # 800

        if False:

            def compute_rows(cols):
                tile_w = max_w // cols
                tile_h = tile_w * bbox_h // bbox_w
                rows   = max_h // tile_h
                print(f"{cols=} {tile_w=} {tile_h=} {rows=}")
                return rows
                
            cols = 1
            while cols * compute_rows(cols) < len(self.outlines):
                cols += 1
    
            self.cols = cols
            self.rows = compute_rows(cols)
        else:
            self.rows = 1 + max(o.row for o in self.outlines)
            self.cols = 1 + max(o.col for o in self.outlines)
            
        self.bbox_w = bbox_w
        self.bbox_h = bbox_h
        self.scale  = min(max_w / (bbox_w * self.cols), max_h / (bbox_h * self.rows))
        self.width  = int(bbox_w * self.scale * self.cols)
        self.height = int(bbox_h * self.scale * self.rows)

        self.font = None
        self.font2 = None

        if use_scenegraph:
            self.scenegraph = Browser.make_scenegraph(self.outlines, self.rows, self.bbox_w, self.bbox_h)
            # with open('scenegraph_foo.json','w') as f:
            #     f.write(puzzler.scenegraph.to_json(self.scenegraph))
        else:
            self.scenegraph = None
        self.hittester = None

    def render(self, canvas, camera):

        if self.renderer == 'opengl':
            r = puzzler.renderer.opengl.OpenGLRenderer(canvas)
        elif self.renderer == 'cairo':
            r = puzzler.renderer.cairo.CairoRenderer(canvas)
        else:
            canvas.delete('all')
            r = puzzler.renderer.canvas.CanvasRenderer(canvas)

        r.transform(camera.matrix)

        if self.scenegraph:
            viewport = (canvas.winfo_width(), canvas.winfo_height())
            
            self.hittester = puzzler.scenegraph.BuildHitTester()(self.scenegraph.root_node)

            sgr = puzzler.scenegraph.SceneGraphRenderer(r, viewport, scale=camera.zoom)
            self.scenegraph.root_node.accept(sgr)
            return r.commit()

        self.font = r.make_font("Courier New", 18)
        self.font2 = r.make_font("Courier New", 12)

        for i, o in enumerate(self.outlines):
            x = o.col
            y = self.rows - 1 - o.row
            tx = (x + .5) * self.bbox_w
            ty = (y + .5) * self.bbox_h
            with puzzler.render.save(r):
                r.translate((tx, ty))
                self.render_outline(r, o)

        return r.commit()

    def render_outline(self, r, o):

        p = o.piece

        # want the corners of the outline bbox centered within the tile
        bbox_center = np.array((o.bbox[0]+o.bbox[2], o.bbox[1]+o.bbox[3])) / 2

        with puzzler.render.save(r):
            r.translate(-bbox_center)

            if p.tabs is not None:
                for i, tab in enumerate(p.tabs):
                    e = tab.ellipse
                    r.draw_ellipse(e.center, e.semi_major, e.semi_minor, e.phi, fill='cyan', outline='')
                    r.draw_text(e.center, text=str(i), font=self.font2, fill='darkblue')

            if p.edges is not None:
                for edge in p.edges:
                    r.draw_lines(edge.line.pts, width=4, fill='pink')

            r.draw_polygon(o.poly, outline='black', fill='', width=1)
            # r.draw_polygon(p.points, outline='black', fill='', width=1)

        r.draw_text(np.zeros(2), text=p.label, font=self.font, fill='black')
        
class BrowseTk:

    def __init__(self, parent, puzzle, renderer, use_scenegraph, screensize):

        self.browser = Browser(puzzle, renderer, use_scenegraph, screensize)

        w, h = self.browser.width, self.browser.height

        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)
        
        self.frame = ttk.Frame(parent, padding=5)
        self.frame.grid(column=0, row=0, sticky=(N, W, E, S))
        
        self.frame.grid_rowconfigure(0, weight=1)
        self.frame.grid_columnconfigure(0, weight=1)

        if renderer == 'opengl':
            self.canvas = puzzler.renderer.opengl.OpenGLFrame(
                self.frame, width=w, height=h, background='white', highlightthickness=0)
        else:
            self.canvas = Canvas(
                self.frame, width=w, height=h, background='white', highlightthickness=0)
            
        self.canvas.grid(column=0, row=0, rowspan=2, sticky=(N, W, E, S))

        self.canvas.bind("<MouseWheel>", self.mouse_wheel)
        self.canvas.bind("<Button-1>", self.canvas_press)
        self.canvas.bind("<B1-Motion>", self.canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.canvas_release)
        self.canvas.bind("<Configure>", self.canvas_configure)
        self.canvas.bind("<Motion>", self.canvas_motion)

        self.controls = ttk.Frame(self.frame)
        self.controls.grid(row=1, sticky=(W,E))

        label0 = ttk.Label(self.controls, text="Scale:")
        label0.grid(column=0, row=0, sticky=(E,))

        self.var_scale = StringVar(value="scale")
        label1 = ttk.Label(self.controls, textvariable=self.var_scale, width=12)
        label1.grid(column=1, row=0, sticky=(E,))

        label2 = ttk.Label(self.controls, text="Mouse:")
        label2.grid(column=2, row=0, sticky=(E,))

        self.var_mouse = StringVar(value="this is my label")
        label3 = ttk.Label(self.controls, textvariable=self.var_mouse, width=80)
        label3.grid(column=3, row=0, sticky=(E,))

        center = np.array((w / 2, h / 2), dtype=np.float64)
        zoom = self.browser.scale
        viewport = (w, h)
        
        self.camera = puzzler.commands.align.Camera(center, zoom, viewport)
        self.draggable = None

    def canvas_configure(self, event):
        self.camera.viewport = (event.width, event.height)
        if self.browser.renderer != 'opengl':
            self.render()
        self.update_scale()

    def mouse_wheel(self, event):
        f = pow(1.2, 1 if event.delta > 0 else -1)
        xy = (event.x, event.y)
        self.camera.fixed_point_zoom(f, xy)
        self.render()
        self.update_scale()

    def update_scale(self):
        scale = self.camera.zoom
        if scale < 1:
            scale = f"1/{1/scale:.1f}x"
        else:
            scale = f"{scale:.2f}x"
        self.var_scale.set(scale)

    def canvas_press(self, event):

        self.draggable = puzzler.commands.align.MoveCamera(self.camera)
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

    def device_to_user(self, xy):
        xy = np.array((*xy, 1)) @ np.linalg.inv(self.camera.matrix).T
        return xy[:2]
    
    def canvas_motion(self, event):

        if self.browser.hittester:
            xy = self.device_to_user((event.x, event.y))
            hits = self.browser.hittester(xy)
            s = '; '.join(str(id)+':'+','.join(tags) for id, tags in hits)
            self.var_mouse.set(s)

    def render(self):

        self.displayed_image = self.browser.render(self.canvas, self.camera)

def browse(args):

    puzzle = puzzler.file.load(args.puzzle)
    
    root = Tk()
    screensize = (root.winfo_screenwidth(), root.winfo_screenheight())
    ui = BrowseTk(root, puzzle, args.renderer, args.mode == 'scenegraph', screensize)
    root.bind('<Key-Escape>', lambda e: root.destroy())
    root.title("Puzzler: browse")
    root.mainloop()

def add_parser(commands):
    parser_browse = commands.add_parser("browse", help="browse pieces")
    parser_browse.add_argument("-r", "--renderer", choices=['tk', 'cairo', 'opengl'], default='cairo',
                               help="renderer (default: %(default)s)")
    parser_browse.add_argument("-m", "--mode", choices=['scenegraph', 'immediate'], default='scenegraph',
                               help="mode (default: %(default)s)")
    parser_browse.set_defaults(func=browse)
