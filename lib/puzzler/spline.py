import bezier
import numpy as np

# pip install bezier==2023.7.28

# Great resource for bezier curves:
#
# https://pomax.github.io/bezierinfo/#derivatives

class Spline:

    def __init__(self, data):
        nodes = np.array(data)
        degree = nodes.shape[0]-1
        # each row is the weights for one dimension
        self.curve = bezier.curve.Curve(nodes.T, degree)

    @property
    def degree(self):
        return self.curve.degree

    @property
    def dimension(self):
        return self.curve.dimension

    @property
    def arclength(self):
        return self.curve.length

    def evaluate(self, t):
        t = np.array(t)
        return self.curve.evaluate_multi(t).T

    def derivative(self):
        nodes = self.degree * np.diff(self.curve.nodes)
        return Spline(nodes.T)

class Curvature:

    def __init__(self, spline):
        self.d = spline.derivative()
        self.dd = self.d.derivative()

    def evaluate(self, t):
        # kappa(t) = (x'y" - x"y') / (x'^2 + y'^2)^(3/2)
        t = np.array(t)
        d = self.d.curve.evaluate_multi(t)
        dd = self.dd.curve.evaluate_multi(t)

        dx, dy = d[0], d[1]
        ddx, ddy = dd[0], dd[1]
        numer = dx * ddy - ddx * dy
        denom = np.float_power(np.square(dx) + np.square(dy), 1.5)
        return numer / denom
