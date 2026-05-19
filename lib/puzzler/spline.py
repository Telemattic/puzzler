import numpy as np
from scipy.interpolate import make_splrep

class BSpline:

    def __init__(self, n, spl_x, spl_y):
        self.n = n
        self.spl_x = spl_x
        self.spl_y = spl_y

    def eval(self, x):
        return np.vstack((self.spl_x(x), self.spl_y(x))).T

    def curvature(self, x):

        dx = self.spl_x.derivative()
        ddx = dx.derivative()

        dy = self.spl_y.derivative()
        ddy = dy.derivative()

        dx = dx(x)
        ddx = ddx(x)
        dy = dy(x)
        ddy = ddy(x)

        return (dx * ddy - dy * ddx) / np.pow(dx*dx + dy*dy, 1.5)

    def path_length(self):
        i = np.arange(self.n)
        xy = self.eval(i)
        d = np.diff(xy, axis=0, prepend=xy[-1:])
        return np.sum(np.linalg.norm(d, axis=1))

def make_spline_for_points(points, s=None):
    n = len(points)
    s = s if s else n
    x = np.arange(n+1)
    y = np.vstack((points, points[:1]))
    spl_x = make_splrep(x, y[:,0], s=s, bc_type='periodic')
    spl_y = make_splrep(x, y[:,1], s=s, bc_type='periodic')
    return BSpline(n, spl_x, spl_y)
