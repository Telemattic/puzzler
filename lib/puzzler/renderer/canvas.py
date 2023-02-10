import puzzler.render
import numpy as np

class CanvasRenderer(puzzler.render.Renderer):

    def __init__(self, canvas=None):
        self.canvas    = canvas
        self._transform = puzzler.render.Transform()
        self.save_stack = []

    def transform(self, m):
        self._transform.multiply(m)

    def translate(self, xy):
        self._transform.translate(xy)

    def rotate(self, rad):
        self._transform.rotate(rad)

    def user_to_device(self, points):
        return self._transform.apply_v2(points)

    def save(self):
        m = self._transform.matrix.copy()
        self.save_stack.append(m)

    def restore(self):
        m = self.save_stack.pop()
        self._transform.matrix = m

    def to_canvas(self, pts):
        return self._transform.apply_v2(pts).tolist()

    def draw_points(self, points, radius, **kw):
        r = np.array((radius, radius))
        for xy in self._transform.apply_v2(np.atleast_2d(points)):
            bbox = np.array((xy-r, xy+r))
            self.canvas.create_oval(bbox.tolist(), **kw)
            
    def draw_lines(self, points, **kw):
        self.canvas.create_line(self.to_canvas(points), **kw)

    def draw_circle(self, points, radius, **kw):
        r = np.linalg.norm(np.array((radius, 0, 0)) @ self._transform.matrix.T)
        rr = np.array((r, r))
        for xy in self._transform.apply_v2(np.atleast_2d(points)):
            bbox = np.array((xy-rr, xy+rr))
            self.canvas.create_oval(bbox.tolist(), **kw)

    def draw_ellipse(self, center, semi_major, semi_minor, phi, **kw):
        ellipse = puzzler.geometry.Ellipse(center, semi_major, semi_minor, phi)
        points = puzzler.geometry.get_ellipse_points(ellipse, npts=40)
        self.canvas.create_polygon(self.to_canvas(points), **kw)

    def draw_polygon(self, points, **kw):
        self.canvas.create_polygon(self.to_canvas(points), **kw)

    def draw_text(self, xy, text, **kw):
        self.canvas.create_text(self.to_canvas(xy), text=text, **kw)


