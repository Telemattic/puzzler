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

    def translate(self, xy):
        x, y = xy
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
        
    def apply_n2(self, normals):
        if normals.ndim == 1:
            normals = np.hstack((normals, np.zeros(1)))
            normals = normals @ self.matrix.T
            return normals[:2]
        
        n = len(normals)
        normals = np.hstack((normals, np.zeros((n,1))))
        normals = normals @ self.matrix.T
        return normals[:,:2]
    
@contextmanager
def save(r):

    r.save()
    try:
        yield r
    finally:
        r.restore()

class Renderer:

    def __init__(self):
        pass

    def save(self):
        pass

    def restore(self):
        pass

    def transform(self, m):
        raise NotImplementedError

    def translate(self, xy):
        raise NotImplementedError

    def rotate(self, rad):
        raise NotImplementedError

    def user_to_device(self, points):
        raise NotImplementedError

    def draw_points(self, points, radius, **kw):
        raise NotImplementedError
            
    def draw_lines(self, points, **kw):
        raise NotImplementedError

    def draw_circle(self, points, radius, **kw):
        raise NotImplementedError

    def draw_ellipse(self, center, semi_major, semi_minor, phi, **kw):
        raise NotImplementedError

    def draw_polygon(self, points, **kw):
        raise NotImplementedError

    def draw_text(self, xy, text, **kw):
        raise NotImplementedError
