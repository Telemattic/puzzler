import numpy as np

class Renderer:

    def __init__(self, canvas=None):
        self.canvas = canvas
        self.stack  = []
        self.matrix = np.identity(3)

    def push(self):
        self.stack.append(self.matrix)

    def pop(self):
        self.matrix = self.stack.pop()

    def multiply(self, m):
        self.matrix = self.matrix @ m

    def scale(self, s):
        m = np.array(((s, 0, 0),
                      (0, s, 0),
                      (0, 0, 1)))
        self.multiply(m)

    def translate(self, x, y):
        m = np.array(((1, 0, x),
                      (0, 1, y),
                      (0, 0, 1)))
        self.multiply(m)
        
    def rotate(self, rad):
        c, s = np.cos(rad), np.sin(rad)
        m = np.array(((c, -s, 0),
                      (s,  c, 0),
                      (0,  0, 1)))
        self.multiply(m)

    def to_v3(self, pts, w):
        assert pts.ndim == 2 and pts.shape[1] == 2
        return np.hstack((pts, np.full((len(pts),1), w, dtype=pts.dtype)))

    def to_v2(self, pts):
        assert pts.ndim == 2 and pts.shape[1] == 3
        return pts[:,:2]

    def to_device(self, pts):
        return self.to_v2(self.to_v3(pts, 1) @ self.matrix.T)

    def to_canvas(self, pts):
        return np.int32(self.to_device(pts)).tolist()

    def draw_lines(self, points, **kw):
        device_points = np.int32(self.to_device(points)).tolist()
        self.canvas.create_line(self.to_canvas(points), **kw)

    def draw_circle(self, points, radius, **kw):
        r = np.array((radius, radius)) / 3
        for xy in self.to_device(np.atleast_2d(points)):
            bbox = np.array((xy-r, xy+r))
            self.canvas.create_oval(bbox.tolist(), **kw)

