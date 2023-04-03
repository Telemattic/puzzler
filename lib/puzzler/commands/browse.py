import cv2 as cv
import math
import numpy as np
import puzzler
import puzzler.renderer.canvas
import puzzler.renderer.cairo
import re

from tkinter import *
from tkinter import ttk

# Camera = puzzler.commands.align.Camera
# MoveCamera = puzzler.commands.align.MoveCamera

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

    def __init__(self, puzzle, use_cairo):

        self.use_cairo = use_cairo
        
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
    
        pieces = []
        for p in puzzle.pieces:
            label = p.label
            r, c = to_row_col(label)
            pieces.append((r, c, p))
        pieces.sort()

        self.outlines = [Browser.Outline(p[2]) for p in pieces]

        bbox_w = max(o.bbox[2] - o.bbox[0] for o in self.outlines)
        bbox_h = max(o.bbox[3] - o.bbox[1] for o in self.outlines)

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
            self.rows = pieces[-1][0]
            self.cols = pieces[-1][1]
            
        self.bbox_w = bbox_w
        self.bbox_h = bbox_h
        self.scale  = min(max_w / (bbox_w * self.cols), max_h / (bbox_h * self.rows))
        self.width  = int(bbox_w * self.scale * self.cols)
        self.height = int(bbox_h * self.scale * self.rows)

        self.font = None

        print(f"scale={self.scale:.3f}")

    def render(self, canvas, camera):

        if self.use_cairo:
            r = puzzler.renderer.cairo.CairoRenderer(canvas)
        else:
            r = puzzler.renderer.canvas.CanvasRenderer(canvas)

        r.transform(camera.matrix)

        self.font = r.make_font("Courier New", 18)

        for i, o in enumerate(self.outlines):
            x = (i %  self.cols)
            y = (self.rows - 1 - (i // self.cols))
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
                for tab in p.tabs:
                    e = tab.ellipse
                    r.draw_ellipse(e.center, e.semi_major, e.semi_minor, e.phi, fill='cyan', outline='')

            if p.edges is not None:
                for edge in p.edges:
                    r.draw_lines(edge.line.pts, width=4, fill='pink')

            r.draw_polygon(o.poly, outline='black', fill='', width=1)

        r.draw_text(np.zeros(2), text=p.label, font=self.font, fill='black')

class BrowseTk:

    def __init__(self, parent, puzzle, use_cairo=False):

        self.browser = Browser(puzzle, use_cairo)

        w, h = self.browser.width, self.browser.height

        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)
        
        self.frame = ttk.Frame(parent, padding=5)
        self.frame.grid(column=0, row=0, sticky=(N, W, E, S))
        
        self.frame.grid_rowconfigure(0, weight=1)
        self.frame.grid_columnconfigure(0, weight=1)
        
        self.canvas = Canvas(self.frame, width=w, height=h, background='white', highlightthickness=0)
        self.canvas.grid(column=0, row=0, rowspan=2, sticky=(N, W, E, S))

        self.canvas.bind("<MouseWheel>", self.mouse_wheel)
        self.canvas.bind("<Button-1>", self.canvas_press)
        self.canvas.bind("<B1-Motion>", self.canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.canvas_release)
        self.canvas.bind("<Configure>", self.canvas_configure)

        center = np.array((w / 2, h / 2), dtype=np.float64)
        zoom = self.browser.scale
        viewport = (w, h)
        
        self.camera = puzzler.commands.align.Camera(center, zoom, viewport)
        self.draggable = None

    def canvas_configure(self, event):
        self.camera.viewport = (event.width, event.height)
        self.render()

    def mouse_wheel(self, event):
        f = pow(1.2, 1 if event.delta > 0 else -1)
        xy = (event.x, event.y)
        self.camera.fixed_point_zoom(f, xy)
        self.render()

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

    def render(self):

        self.canvas.delete('all')

        self.displayed_image = self.browser.render(self.canvas, self.camera)

def browse(args):

    puzzle = puzzler.file.load(args.puzzle)
    
    root = Tk()
    ui = BrowseTk(root, puzzle, args.cairo)
    root.bind('<Key-Escape>', lambda e: root.destroy())
    root.title("Puzzler: browse")
    root.mainloop()

def add_parser(commands):
    parser_browse = commands.add_parser("browse", help="browse pieces")
    parser_browse.add_argument("-c", "--cairo", action='store_const', const=True, default=False,
                               help="use cairo rendering (default: tk)")
    parser_browse.set_defaults(func=browse)
