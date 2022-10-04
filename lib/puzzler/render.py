import puzzler
import numpy as np

from contextlib import contextmanager

class Transform:

    def __init__(self):
        self.matrix = np.identity(3)

    def multiply(self, m):
        self.matrix = self.matrix @ m
        return self

    def scale(self, s):
        m = np.array(((s, 0, 0),
                      (0, s, 0),
                      (0, 0, 1)))
        return self.multiply(m)

    def translate(self, x, y):
        m = np.array(((1, 0, x),
                      (0, 1, y),
                      (0, 0, 1)))
        return self.multiply(m)
        
    def rotate(self, rad):
        c, s = np.cos(rad), np.sin(rad)
        m = np.array(((c, -s, 0),
                      (s,  c, 0),
                      (0,  0, 1)))
        return self.multiply(m)

    def apply_v2(self, points):
        if points.ndim == 1:
            points = np.hstack((points, np.ones(1)))
            points = points @ self.matrix.T
            return points[:2]
        
        n = len(points)
        points = np.hstack((points, np.ones((n,1))))
        points = points @ self.matrix.T
        return points[:,:2]
        
@contextmanager
def save_matrix(m):

    saved_matrix = m.matrix.copy()
    try:
        yield m
    finally:
        m.matrix = saved_matrix

class Renderer:

    def __init__(self, canvas=None):
        self.canvas    = canvas
        self.transform = Transform()

    def to_canvas(self, pts):
        return self.transform.apply_v2(pts).tolist()

    def draw_points(self, points, radius, **kw):
        r = np.array((radius, radius))
        for xy in self.transform.apply_v2(np.atleast_2d(points)):
            bbox = np.array((xy-r, xy+r))
            self.canvas.create_oval(bbox.tolist(), **kw)
            
    def draw_lines(self, points, **kw):
        self.canvas.create_line(self.to_canvas(points), **kw)

    def draw_circle(self, points, radius, **kw):
        r = np.array((radius, radius)) / 3
        for xy in self.transform.apply_v2(np.atleast_2d(points)):
            bbox = np.array((xy-r, xy+r))
            self.canvas.create_oval(bbox.tolist(), **kw)

    def draw_polygon(self, points, **kw):
        self.canvas.create_polygon(self.to_canvas(points), **kw)

    def draw_text(self, xy, text, **kw):
        self.canvas.create_text(self.to_canvas(xy), text=text, **kw)

