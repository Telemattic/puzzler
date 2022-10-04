import cv2 as cv
import numpy as np
import puzzler
import re

from tkinter import *
from tkinter import font
from tkinter import ttk

class Browser:

    class Outline:

        def __init__(self, piece, epsilon=10):

            assert piece.points is not None
            approx  = cv.approxPolyDP(piece.points, epsilon, True)
            self.poly = np.squeeze(approx)
            self.poly = np.concatenate((self.poly, self.poly[:1,:]))

            ll = np.min(self.poly, 0)
            ur = np.max(self.poly, 0)
            self.bbox   = tuple(ll.tolist() + ur.tolist())
            self.piece  = piece

    def __init__(self, puzzle):

        pieces = []
        for p in puzzle.pieces:
            label = p.label
            m = re.fullmatch("^(\w+)(\d+)", label)
            if m:
                pieces.append((m[1], int(m[2]), p))
            else:
                pieces.append((label, None, p))
        pieces.sort()

        self.outlines = [Browser.Outline(p[2]) for p in pieces]

        bbox_w = max(o.bbox[2] - o.bbox[0] for o in self.outlines)
        bbox_h = max(o.bbox[3] - o.bbox[1] for o in self.outlines)

        max_w = 1400
        max_h = 800

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

        self.bbox_w = bbox_w
        self.bbox_h = bbox_h
        self.tile_w = max_w // cols
        self.tile_h = self.tile_w * bbox_h // bbox_w
        self.scale  = min(self.tile_w / bbox_w, self.tile_h / bbox_h)
        self.width  = self.tile_w * self.cols
        self.height = self.tile_h * self.rows

    def render(self, canvas):

        h = self.height
        camera_matrix = np.array(
            ((1,  0,   0),
             (0, -1, h-1),
             (0,  0,   1)), dtype=np.float64)

        r = puzzler.render.Renderer(canvas)
        r.transform.multiply(camera_matrix)
        r.transform.scale(self.scale)
        
        self.font = font.Font(family='Courier', name='pieceLabelFont', size=12)
        for i, o in enumerate(self.outlines):
            x = (i %  self.cols)
            y = (self.rows - 1 - (i // self.cols))
            tx = (x + .5) * self.bbox_w
            ty = (y + .5) * self.bbox_h
            r.transform.push()
            r.transform.translate(tx, ty)
            self.render_outline(r, o)
            r.transform.pop()

    def render_outline(self, r, o):

        p = o.piece

        # want the corners of the outline bbox centered within the tile
        bbox_center = np.array((o.bbox[0]+o.bbox[2], o.bbox[1]+o.bbox[3])) / 2

        r.transform.push()
        r.transform.translate(*-bbox_center)

        if p.tabs is not None:
            for tab in p.tabs:
                pts = puzzler.geometry.get_ellipse_points(tab.ellipse, npts=40)
                r.draw_polygon(pts, fill='cyan', outline='')

        if p.edges is not None:
            for edge in p.edges:
                r.draw_lines(edge.line.pts, width=4, fill='pink')

        r.draw_polygon(o.poly, outline='black', fill='', width=1)

        r.transform.pop()

        r.draw_text((0, 0), text=p.label, font=self.font, fill='black')
        
class BrowseTk:

    def __init__(self, parent, puzzle):

        self.browser = Browser(puzzle)

        w, h = self.browser.width, self.browser.height

        self.frame = ttk.Frame(parent, padding=5)
        self.frame.grid(column=0, row=0, sticky=(N, W, E, S))
        
        self.canvas = Canvas(self.frame, width=w, height=h, background='white', highlightthickness=0)
        self.canvas.grid(column=0, row=0, rowspan=2, sticky=(N, W, E, S))

        self.render()

    def render(self):

        self.canvas.delete('all')
        self.browser.render(self.canvas)

def browse(args):

    puzzle = puzzler.file.load(args.puzzle)
    
    root = Tk()
    ui = BrowseTk(root, puzzle)
    root.bind('<Key-Escape>', lambda e: root.destroy())
    root.title("Puzzler: browse")
    root.mainloop()

def add_parser(commands):
    parser_browse = commands.add_parser("browse", help="browse pieces")
    parser_browse.set_defaults(func=browse)
