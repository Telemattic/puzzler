import cv2 as cv
import numpy as np
import puzzler
import re
import scipy

from tkinter import *
from tkinter import ttk

class Normals:

    def __init__(self, piece):

        self.piece = piece
        self.kdtree = scipy.spatial.KDTree(self.piece.points)

        bbox_w, bbox_h = 1.2 * (piece.bbox[1] - piece.bbox[0])

        max_w = 1400
        max_h = 800

        self.scale  = min(max_w / bbox_w, max_h / bbox_h)
        self.width  = bbox_w * self.scale
        self.height = bbox_h * self.scale
        self.ii     = None
        self.bl     = 10

    def get_camera_matrix(self):
        w = self.width
        h = self.height
        s = self.scale
        
        m1 = np.array(
            ((1,  0, w/2),
             (0, -1, h/2),
             (0,  0,   1)), dtype=np.float64)
        m2 = np.array(
            ((s, 0, 0),
             (0, s, 0),
             (0, 0, 1)), dtype=np.float64)

        return m1 @ m2

    def render(self, canvas):

        r = puzzler.renderer.canvas.CanvasRenderer(canvas)
        r.transform(self.get_camera_matrix())

        p = self.piece
        
        r.draw_polygon(p.points, outline='black', fill='', width=1)

        if self.ii is not None:
            self.render_normals(r)

    def render_normals(self, r):

        p = self.piece
        n = len(p.points)
        i = self.ii

        p0 = self.get_point(i-self.bl)
        p1 = self.get_point(i)
        p2 = self.get_point(i+self.bl)
        
        n1 = self.compute_normal(p0, p1)
        n2 = self.compute_normal(p1, p2)

        r.draw_points(p0, fill='red', radius=5)
        r.draw_points(p2, fill='green', radius=5)

        r.draw_lines(np.array((p1, p1+n1*35)), fill='red', width=2, arrow='last')
        r.draw_lines(np.array((p1, p1+n2*35)), fill='green', width=2, arrow='last')

    def get_point(self, i):
        p = self.piece
        n = len(p.points)
        if i < 0:
            i += n
        elif i >= n:
            i -= n
        return p.points[i]

    def compute_normal(self, p1, p2):
        n = puzzler.math.unit_vector(p2 - p1)
        return np.array((-n[1], n[0]))
        
class BrowseTk:

    def __init__(self, parent, piece):

        self.normals = Normals(piece)

        w, h = self.normals.width, self.normals.height

        self.frame = ttk.Frame(parent, padding=5)
        self.frame.grid(column=0, row=0, sticky=(N, W, E, S))
        
        self.canvas = Canvas(self.frame, width=w, height=h, background='white', highlightthickness=0)
        self.canvas.grid(column=0, row=0, rowspan=2, sticky=(N, W, E, S))
        self.canvas.bind("<Motion>", self.motion)

        self.render()

    def motion(self, event):
        xy0 = np.array((event.x, event.y, 1))
        xy1 = xy0 @ np.linalg.inv(self.normals.get_camera_matrix()).T
        xy1 = xy1[:2]

        dd, ii = self.normals.kdtree.query(xy1)

        if ii != self.normals.ii:
            self.normals.ii = ii
            self.render()
        
    def render(self):

        self.canvas.delete('all')
        self.normals.render(self.canvas)

def normals(args):

    puzzle = puzzler.file.load(args.puzzle)
    
    pieces = {i.label: i for i in puzzle.pieces}
    
    root = Tk()
    ui = BrowseTk(root, pieces[args.piece])
    root.bind('<Key-Escape>', lambda e: root.destroy())
    root.title("Puzzler: normals")
    root.wm_resizable(0, 0)
    root.mainloop()

def add_parser(commands):
    parser_normals = commands.add_parser("normals", help="explore normals")
    parser_normals.add_argument("piece")
    parser_normals.set_defaults(func=normals)
